"""
LongMemEval evaluation for CortexGPT.

Accuracy vs. conversation history size on LongMemEval.
The model reads multi-turn conversation history in chunks, carrying M_cross
across chunks, then GENERATES a short answer scored by containment (the gold
answer string appearing in the generation).  The old single-greedy-token
exact-match scoring could never match LongMemEval's multi-word answers, so
every model scored 0 — uninformative.

Default split is longmemeval_s (~115k-token real haystack, the actual memory
test).  The oracle split (relevant sessions only, tiny context) is available
via --split oracle as an upper-bound control.

Dataset: xiaowu0162/LongMemEval  (HuggingFace, test split)

Usage:
    python evals/eval_longmemeval.py \
        --checkpoint runs/cortex-5b/checkpoint_0154441/checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model_utils import (load_checkpoint, has_cross_state, to_num_steps,
                         prime_cross_state, greedy_generate, ccot_prime)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LongMemEval evaluation for CortexGPT")
    p.add_argument("--checkpoint",   type=str, default=None,
                   help="Optional train.py .pt (finetuned weights overlaid strict=False).")
    p.add_argument("--model_name",   default="EleutherAI/pythia-160m")
    p.add_argument("--memory_slots", type=int, default=None,
                   help="Override K; default reads memory_slots from the checkpoint config")
    p.add_argument("--T",            type=int, default=None,
                   help="Recurrence depth at eval (None = use checkpoint mean_recurrence)")
    p.add_argument("--seq_len",      type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=32,
                   help="Greedy tokens generated for the answer (containment-scored)")
    p.add_argument("--num_chunks",   type=int, default=0,
                   help="Target a FIXED number of equal subwindows per example "
                        "(4 = the trained cross_chunks regime). Window capped "
                        "at --seq_len; 0 = fixed-size seq_len chunks")
    p.add_argument("--passes_per_chunk", type=int, default=1,
                   help="Full-model passes per priming chunk (M_cross carried "
                        "pass-to-pass; >1 = multi-pass buffer fill)")
    p.add_argument("--ccot_passes", type=int, default=0,
                   help="Extra silent full passes over the final question chunk "
                        "before generation (latent CCoT thinking); 0 = off")
    p.add_argument("--split",        default="s", choices=["s", "m", "oracle"],
                   help="LongMemEval split: s = ~115k-token haystack (default, "
                        "the real memory test), m = ~1.5M tokens, oracle = "
                        "relevant sessions only (upper-bound control)")
    p.add_argument("--max_examples", type=int, default=200,
                   help="Max examples per session-depth bucket (0 = all)")
    p.add_argument("--depth_buckets", nargs="+", type=int, default=[5, 10, 20, 50])
    p.add_argument("--out_dir",      default="eval_results/longmemeval")
    p.add_argument("--dataset_path", default=None,
                   help="Local path to pre-downloaded LongMemEval (load_from_disk). "
                        "Required on nodes without internet access.")
    p.add_argument("--dtype",        default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Conversation formatting and chunking
# ---------------------------------------------------------------------------

def format_history(turns: list[dict]) -> str:
    return "\n".join(f"{t.get('role','user').capitalize()}: {t.get('content','')}"
                     for t in turns)


SUFFIX_TEMPLATE = ("\n\nBased on the conversation above, answer the question "
                   "briefly and directly.\nQuestion: {q}\nAnswer:")


def split_history(tokenizer, history: str, suffix: str, seq_len: int,
                  max_new_tokens: int, num_chunks: int = 0):
    """Priming chunks + final prediction chunk; the suffix (question) stays
    intact in the final chunk with room reserved for generation.

    num_chunks > 0 targets a FIXED number of equal subwindows (training used
    cross_chunks=4), window capped at seq_len — long histories fall back to
    more, seq_len-sized chunks."""
    ids = tokenizer(history, add_special_tokens=False).input_ids
    sfx = tokenizer(suffix, add_special_tokens=False).input_ids
    if num_chunks > 0:
        total = len(ids) + len(sfx) + max_new_tokens
        seq_len = min(seq_len, max(-(-total // num_chunks),
                                   len(sfx) + max_new_tokens + 1))
    room = max(seq_len - len(sfx) - max_new_tokens, 0)
    if len(ids) > room:
        head, tail = ids[: len(ids) - room], ids[len(ids) - room:]
    else:
        head, tail = [], ids
    prime_chunks = [torch.tensor(head[s: s + seq_len], dtype=torch.long).unsqueeze(0)
                    for s in range(0, len(head), seq_len)]
    # Hard cap: if the suffix alone exceeds the window (tiny seq_len), keep
    # the END (question + "Answer:").
    final = (tail + sfx)[-max(seq_len - max_new_tokens, 1):]
    final_ids = torch.tensor(final, dtype=torch.long).unsqueeze(0)
    return prime_chunks, final_ids


def contains_answer(pred: str, gold: str) -> bool:
    gold = str(gold).strip().lower()
    return bool(gold) and gold in pred.lower()


# ---------------------------------------------------------------------------
# Single-example evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_one(model, tokenizer, turns, question, answer, T, seq_len,
             max_new_tokens, passes_per_chunk=1, ccot_passes=0, num_chunks=0):
    history = format_history(turns)
    suffix  = SUFFIX_TEMPLATE.format(q=question)
    prime_chunks, final_ids = split_history(tokenizer, history, suffix,
                                            seq_len, max_new_tokens,
                                            num_chunks=num_chunks)
    num_steps = to_num_steps(T)
    m_cross = prime_cross_state(model, prime_chunks, num_steps,
                                passes_per_chunk=passes_per_chunk)
    m_cross = ccot_prime(model, final_ids, num_steps, ccot_passes,
                         m_cross_init=m_cross)
    pred = greedy_generate(model, tokenizer, final_ids, max_new_tokens,
                           num_steps, m_cross=m_cross, stop_on_newline=True)
    return contains_answer(pred, answer), pred


# ---------------------------------------------------------------------------
# Dataset evaluation loop
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, T, seq_len, max_examples, depth_buckets,
             max_new_tokens, split="s", passes_per_chunk=1, ccot_passes=0,
             num_chunks=0, dataset_path=None):
    # Preference order: requested split first, then fallbacks (a warning is
    # printed if we fall back — the splits are NOT comparable).
    order = {"s":      ("longmemeval_s", "longmemeval_oracle", "longmemeval_m"),
             "m":      ("longmemeval_m", "longmemeval_s", "longmemeval_oracle"),
             "oracle": ("longmemeval_oracle", "longmemeval_s", "longmemeval_m")}[split]

    bucket_labels = [f"≤{d}sess" for d in depth_buckets] + [f">{depth_buckets[-1]}sess"]
    results = {lbl: {"correct": 0, "total": 0} for lbl in bucket_labels}
    samples = []

    if dataset_path is not None:
        # snapshot_download saves files without extensions (longmemeval_s, etc.)
        import json as _json
        local = Path(dataset_path)
        for candidate in order:
            p = local / candidate
            if p.exists():
                if candidate != order[0]:
                    print(f"WARNING: requested split '{order[0]}' not found; "
                          f"falling back to '{candidate}' — results are NOT "
                          f"comparable across splits.")
                with open(p) as f:
                    raw = _json.load(f)
                # file is either a list of examples or {"test": [...]} / {"data": [...]}
                if isinstance(raw, list):
                    ds = raw
                elif isinstance(raw, dict):
                    ds = next(iter(raw.values()))
                break
        else:
            raise FileNotFoundError(
                f"Could not find longmemeval_s/m/oracle in {dataset_path}. "
                f"Run `python evals/download_datasets.py` on a login node "
                f"(it downloads to <repo>/data/LongMemEval) and pass that path "
                f"via --dataset_path."
            )
    else:
        # The hub repo stores extensionless JSON files at the root
        # (longmemeval_oracle / _m / _s), which load_dataset cannot
        # auto-resolve — download the file and parse it directly.
        import json as _json
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id="xiaowu0162/LongMemEval",
                            repo_type="dataset", filename=order[0])
        with open(p) as f:
            raw = _json.load(f)
        ds = raw if isinstance(raw, list) else next(iter(raw.values()))
    seen = 0
    for ex in ds:
        question = ex.get("question", "")
        answer   = ex.get("answer", "")
        # haystack_sessions is stored as a JSON string of [[{role,content},...],...]
        raw_sessions = ex.get("haystack_sessions", "[]")
        try:
            turns = json.loads(raw_sessions) if isinstance(raw_sessions, str) else raw_sessions
        except Exception:
            continue
        depth = len(turns)
        if not turns or not question or not answer:
            continue

        bucket = bucket_labels[-1]
        for i, thresh in enumerate(depth_buckets):
            if depth <= thresh:
                bucket = bucket_labels[i]
                break

        if max_examples > 0 and results[bucket]["total"] >= max_examples:
            continue

        # Flatten all sessions into a single turn list for the model
        all_turns = [turn for session in turns for turn in session]
        ok, pred = eval_one(model, tokenizer, all_turns, question, answer, T,
                            seq_len, max_new_tokens,
                            passes_per_chunk=passes_per_chunk,
                            ccot_passes=ccot_passes, num_chunks=num_chunks)
        if ok:
            results[bucket]["correct"] += 1
        results[bucket]["total"] += 1
        seen += 1
        if seen <= 20:   # first 20 predictions, for debuggability
            samples.append({"bucket": bucket, "question": question,
                            "gold": answer, "pred": pred, "correct": ok})

        if seen % 50 == 0:
            print(f"  {seen} examples processed...")

        if max_examples > 0 and all(r["total"] >= max_examples for r in results.values()):
            break

    for r in results.values():
        r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0.0
    return results, samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint, args.model_name,
                                 args.memory_slots, dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    T = args.T
    print(f"T={T if T is not None else cfg.mean_recurrence}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, samples = run_eval(model, tokenizer, T, args.seq_len,
                                args.max_examples, args.depth_buckets,
                                args.max_new_tokens, split=args.split,
                                passes_per_chunk=args.passes_per_chunk,
                                ccot_passes=args.ccot_passes,
                                num_chunks=args.num_chunks,
                                dataset_path=args.dataset_path)
    # fall back to the model dir when no overlay checkpoint was given
    # (--checkpoint defaults to None) so we don't do Path(None).
    label   = (Path(args.checkpoint).parent.parent.name
               if args.checkpoint else Path(args.model_name).name)

    print(f"\n  {'Bucket':<16} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"  {'-'*44}")
    for bucket, r in results.items():
        if r["total"] > 0:
            print(f"  {bucket:<16} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>8.3f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({label: results}, f, indent=2)

    with open(out_dir / "samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    with open(out_dir / "summary.csv", "w") as f:
        f.write("model,bucket,correct,total,accuracy\n")
        for bucket, r in results.items():
            f.write(f"{label},{bucket},{r['correct']},{r['total']},{r['accuracy']:.4f}\n")

    print(f"\nResults saved → {out_dir}")

    # Guard against silently-empty evals — fail the job loudly instead of
    # leaving an all-zero results file that looks like a (bad) result.
    if sum(r["total"] for r in results.values()) == 0:
        print("ERROR: 0 examples were evaluated — results are empty. "
              "Check --dataset_path / network access.")
        sys.exit(1)


if __name__ == "__main__":
    main()
