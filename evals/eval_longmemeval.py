"""
LongMemEval evaluation for CortexGPT.

Accuracy vs. conversation turn depth on LongMemEval.
The model reads multi-turn conversation history in chunks, carrying M_cross
across chunks, then predicts the answer token.

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

from model_utils import load_checkpoint, has_cross_state, to_num_steps


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
    p.add_argument("--max_examples", type=int, default=200,
                   help="Max examples per turn-depth bucket (0 = all)")
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

def format_conversation(turns: list[dict], question: str) -> str:
    lines = [f"{t.get('role','user').capitalize()}: {t.get('content','')}" for t in turns]
    lines += [f"Question: {question}", "Answer:"]
    return "\n".join(lines)


def encode_and_chunk(tokenizer, text: str, seq_len: int):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    chunks = []
    for start in range(0, max(len(ids), 1), seq_len):
        chunk = ids[start : start + seq_len]
        chunks.append(torch.tensor(chunk, dtype=torch.long).unsqueeze(0))
    return chunks


# ---------------------------------------------------------------------------
# Single-example evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_one(model, tokenizer, turns, question, answer, T, seq_len) -> bool:
    text   = format_conversation(turns, question)
    chunks = encode_and_chunk(tokenizer, text, seq_len)
    device = next(model.parameters()).device
    num_steps = to_num_steps(T)

    # Prime the cross state on every chunk except the final (prediction) one.
    # Models without cross state skip priming entirely — nothing is carried.
    m_cross = None
    if has_cross_state(model):
        for chunk in chunks[:-1]:
            chunk = chunk.to(device)
            out   = model(input_ids=chunk, num_steps=num_steps,
                          m_cross_in=m_cross, return_m_cross=True)
            m_cross = out.get("m_cross")

    chunk = chunks[-1].to(device)
    out   = model(input_ids=chunk, num_steps=num_steps,
                  m_cross_in=m_cross, return_m_cross=False)
    pred  = tokenizer.decode([out["logits"][0, -1].argmax(dim=-1).item()]).strip().lower()
    return pred == str(answer).strip().lower()


# ---------------------------------------------------------------------------
# Dataset evaluation loop
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, T, seq_len, max_examples, depth_buckets, dataset_path=None):
    from datasets import load_dataset

    bucket_labels = [f"≤{d}turns" for d in depth_buckets] + [f">{depth_buckets[-1]}turns"]
    results = {lbl: {"correct": 0, "total": 0} for lbl in bucket_labels}

    if dataset_path is not None:
        # snapshot_download saves files without extensions (longmemeval_oracle, etc.).
        # Use the oracle split; fall back to _m then _s.
        import json as _json
        local = Path(dataset_path)
        for candidate in ("longmemeval_oracle", "longmemeval_m", "longmemeval_s"):
            p = local / candidate
            if p.exists():
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
                f"Could not find longmemeval_oracle/m/s in {dataset_path}. "
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
                            repo_type="dataset", filename="longmemeval_oracle")
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
        if eval_one(model, tokenizer, all_turns, question, answer, T, seq_len):
            results[bucket]["correct"] += 1
        results[bucket]["total"] += 1
        seen += 1

        if seen % 50 == 0:
            print(f"  {seen} examples processed...")

        if max_examples > 0 and all(r["total"] >= max_examples for r in results.values()):
            break

    for r in results.values():
        r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0.0
    return results


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

    results = run_eval(model, tokenizer, T, args.seq_len, args.max_examples, args.depth_buckets,
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
