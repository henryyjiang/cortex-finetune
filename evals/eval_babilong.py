"""
BABILong evaluation for CortexGPT.

Accuracy vs. context length on BABILong QA1/QA2/QA3.
The model reads the context in seq_len-token chunks, carrying M_cross across
chunks, then GENERATES a short answer which is scored by containment (the
gold word appearing in the generation).  Few-shot demos + an instruction are
appended just before the real question so base (non-instruct) models know the
expected format — the old single-greedy-token exact-match scoring pinned every
model (including the base) to 0% and was uninformative.

Dataset: RMT-team/BABILong  (HuggingFace)

Usage:
    python evals/eval_babilong.py \
        --checkpoint runs/cortex-5b/checkpoint_0154441/checkpoint.pt \
        --tasks qa1 qa2 qa3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import re

import torch
from transformers import AutoTokenizer

from model_utils import (load_checkpoint, has_cross_state, to_num_steps,
                         prime_cross_state, greedy_generate, ccot_prime)


# ---------------------------------------------------------------------------
# Few-shot prompt template (per task)
# ---------------------------------------------------------------------------
# BABILong answers are single bAbI words; the official harness prompts with an
# instruction + examples and scores generations leniently.  Demos sit in the
# FINAL chunk right before the real question, so every model (with or without
# memory) sees them.

INSTRUCTION = ("You will read a story with facts scattered in it. "
               "Answer the question using only those facts. "
               "Answer with a single word.")

TASK_DEMOS = {
    "qa1": [("Mary moved to the bathroom. John went to the hallway.",
             "Where is Mary?", "bathroom"),
            ("Daniel journeyed to the office. Sandra travelled to the garden.",
             "Where is Sandra?", "garden")],
    "qa2": [("John took the apple. John went to the office.",
             "Where is the apple?", "office"),
            ("Mary got the football. Mary travelled to the kitchen.",
             "Where is the football?", "kitchen")],
    "qa3": [("Mary got the milk. Mary went to the bedroom. Mary went to the garden.",
             "Where was the milk before the garden?", "bedroom"),
            ("John took the football. John journeyed to the hallway. John went to the office.",
             "Where was the football before the office?", "hallway")],
}


def build_suffix(task: str, question: str) -> str:
    parts = [f"\n\n{INSTRUCTION}\n"]
    for story, q, a in TASK_DEMOS.get(task, []):
        parts.append(f"\nExample:\n{story}\nQuestion: {q}\nAnswer: {a}\n")
    parts.append(f"\nQuestion: {question}\nAnswer:")
    return "".join(parts)


def contains_answer(pred: str, gold: str) -> bool:
    gold = str(gold).strip().lower()
    if not gold:
        return False
    return re.search(rf"(?<![a-z]){re.escape(gold)}(?![a-z])", pred.lower()) is not None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("BABILong evaluation for CortexGPT")
    p.add_argument("--checkpoint",    type=str, default=None,
                   help="Optional train.py .pt (finetuned weights overlaid strict=False). "
                        "Omit to eval the graft-prepared --model_name dir as-is.")
    p.add_argument("--model_name",    default="EleutherAI/pythia-160m")
    p.add_argument("--memory_slots",  type=int, default=None,
                   help="Override K; default reads memory_slots from the checkpoint config")
    p.add_argument("--T",             type=int, default=None,
                   help="Recurrence depth at eval (None = use checkpoint mean_recurrence)")
    p.add_argument("--tasks",         nargs="+", default=["qa1", "qa2", "qa3"],
                   choices=["qa1", "qa2", "qa3"])
    p.add_argument("--seq_len",       type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=12,
                   help="Greedy tokens generated for the answer (containment-scored)")
    p.add_argument("--num_chunks", type=int, default=0,
                   help="Target a FIXED number of equal subwindows per example "
                        "(4 = the trained cross_chunks regime: 3 buffer updates "
                        "before the final read). Window capped at --seq_len; "
                        "0 = fixed-size seq_len chunks (count varies)")
    p.add_argument("--passes_per_chunk", type=int, default=1,
                   help="Full-model passes per priming chunk (M_cross carried "
                        "pass-to-pass; >1 = multi-pass buffer fill)")
    p.add_argument("--ccot_passes", type=int, default=0,
                   help="Extra silent full passes over the final question chunk "
                        "before generation (latent CCoT thinking); 0 = off")
    p.add_argument("--max_examples",  type=int, default=500,
                   help="Max examples per task/length bucket (0 = all)")
    p.add_argument("--length_buckets", nargs="+", type=int,
                   default=[1000, 2000, 4000, 8000, 16000, 32000])
    p.add_argument("--out_dir",       default="eval_results/babilong")
    p.add_argument("--dataset_path",  default=None,
                   help="Local path to pre-downloaded BABILong snapshot (snapshot_download). "
                        "Required on nodes without internet access.")
    p.add_argument("--dtype",         default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Chunked context encoding
# ---------------------------------------------------------------------------

def split_context(tokenizer, context: str, suffix: str, seq_len: int,
                  max_new_tokens: int, num_chunks: int = 0):
    """Split the context into priming chunks + a final prediction chunk.

    The final chunk holds the suffix (instruction + demos + question) intact
    plus as much trailing context as fits, reserving room for generation; the
    rest of the context goes to the priming chunks so nothing is dropped.

    num_chunks > 0 targets a FIXED number of equal subwindows (training used
    cross_chunks=4, i.e. 3 buffer updates before the final read), sizing the
    window as ceil(total/num_chunks) — capped at seq_len (the model's trained
    window; longer contexts fall back to more, seq_len-sized chunks)."""
    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    sfx_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    if num_chunks > 0:
        total = len(ctx_ids) + len(sfx_ids) + max_new_tokens
        seq_len = min(seq_len, max(-(-total // num_chunks),
                                   len(sfx_ids) + max_new_tokens + 1))
    room = max(seq_len - len(sfx_ids) - max_new_tokens, 0)
    if len(ctx_ids) > room:
        head, tail = ctx_ids[: len(ctx_ids) - room], ctx_ids[len(ctx_ids) - room:]
    else:
        head, tail = [], ctx_ids
    prime_chunks = [torch.tensor(head[s: s + seq_len], dtype=torch.long).unsqueeze(0)
                    for s in range(0, len(head), seq_len)]
    # Hard cap: if the suffix alone exceeds the window (tiny seq_len), keep
    # the END (question + "Answer:") — degrade by dropping instruction/demos.
    final = (tail + sfx_ids)[-max(seq_len - max_new_tokens, 1):]
    final_ids = torch.tensor(final, dtype=torch.long).unsqueeze(0)
    return prime_chunks, final_ids


# ---------------------------------------------------------------------------
# Single-example evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_one(model, tokenizer, context, question, answer, T, seq_len,
             max_new_tokens, task, passes_per_chunk=1, ccot_passes=0,
             num_chunks=0):
    suffix = build_suffix(task, question)
    prime_chunks, final_ids = split_context(tokenizer, context, suffix,
                                            seq_len, max_new_tokens,
                                            num_chunks=num_chunks)
    num_steps = to_num_steps(T)
    # Prime the cross state on every context chunk before the final one — the
    # final chunk is the prediction pass.  Models without cross state carry
    # nothing and see only the final chunk (the no-memory control).
    m_cross = prime_cross_state(model, prime_chunks, num_steps,
                                passes_per_chunk=passes_per_chunk)
    # Optional latent CCoT: extra silent passes over the question chunk,
    # seeded with the context-primed buffer, before answering.
    m_cross = ccot_prime(model, final_ids, num_steps, ccot_passes,
                         m_cross_init=m_cross)
    pred = greedy_generate(model, tokenizer, final_ids, max_new_tokens,
                           num_steps, m_cross=m_cross, stop_on_newline=True)
    return contains_answer(pred, answer), pred


# ---------------------------------------------------------------------------
# Task loop
# ---------------------------------------------------------------------------

def run_task(task_name, model, tokenizer, T, seq_len, max_examples, length_buckets,
             max_new_tokens, passes_per_chunk=1, ccot_passes=0, num_chunks=0,
             dataset_path=None):
    from datasets import load_dataset

    # BABILong uses config name for context length (e.g. '1k', '4k') and
    # split for the task (e.g. 'qa1'). Load each bucket config separately.
    config_names = [f"{b // 1000}k" for b in length_buckets]
    results = {cfg: {"correct": 0, "total": 0} for cfg in config_names}
    samples = []

    for cfg in config_names:
        if dataset_path is not None:
            # Verified hub layout: data/<task>/<length>.json
            local = Path(dataset_path) / "data" / task_name / f"{cfg}.json"
            if not local.exists():
                # A missing local file is a setup error, not a skippable bucket —
                # silent skipping here is how an entire eval once produced
                # all-zero "results" without anyone noticing.
                raise FileNotFoundError(
                    f"BABILong file not found: {local}\n"
                    f"Expected snapshot layout data/<task>/<length>.json. "
                    f"Run `python evals/download_datasets.py` on a login node "
                    f"(it downloads to <repo>/data/BABILong) and pass that path "
                    f"via --dataset_path."
                )
            ds = load_dataset("json", data_files=str(local), split="train",
                              streaming=True)
        else:
            try:
                # Load the task/length file directly. Config auto-resolution on
                # this repo is unreliable (it tries to materialise every
                # task x length combination, some of which don't exist).
                ds = load_dataset(
                    "json",
                    data_files=f"hf://datasets/RMT-team/BABILong/data/{task_name}/{cfg}.json",
                    split="train",
                    streaming=True,
                )
            except Exception as e:
                # Network flake on one bucket shouldn't kill the whole job; the
                # all-zero guard in main() still fails the run if nothing loads.
                print(f"  [{task_name}/{cfg}] ERROR loading from hub — {e}")
                continue

        seen = 0
        for ex in ds:
            ctx      = ex.get("input", ex.get("context", ex.get("text", "")))
            question = ex.get("question", "")
            answer   = str(ex.get("target", ex.get("answer", "")))
            if not ctx or not question or not answer:
                continue

            ok, pred = eval_one(model, tokenizer, ctx, question, answer, T,
                                seq_len, max_new_tokens, task_name,
                                passes_per_chunk=passes_per_chunk,
                                ccot_passes=ccot_passes,
                                num_chunks=num_chunks)
            if ok:
                results[cfg]["correct"] += 1
            results[cfg]["total"] += 1
            seen += 1
            if seen <= 5:   # first 5 per length bucket, for debuggability
                samples.append({"bucket": cfg, "question": question,
                                "gold": answer, "pred": pred, "correct": ok})

            if seen % 50 == 0:
                print(f"  [{task_name}/{cfg}] {seen} examples processed...")

            if max_examples > 0 and seen >= max_examples:
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
    print(f"T={T if T is not None else cfg.mean_recurrence}  tasks={args.tasks}  "
          f"num_chunks={args.num_chunks}  passes_per_chunk={args.passes_per_chunk}  "
          f"ccot_passes={args.ccot_passes}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    # run dir name as label; fall back to the model dir when no overlay
    # checkpoint was given (--checkpoint defaults to None: eval the prepared dir
    # as-is) so we don't do Path(None).
    label = (Path(args.checkpoint).parent.parent.name
             if args.checkpoint else Path(args.model_name).name)
    all_results[label] = {}

    all_samples: dict = {}
    for task in args.tasks:
        print(f"\n--- {task} ---")
        task_results, task_samples = run_task(
            task, model, tokenizer, T, args.seq_len,
            args.max_examples, args.length_buckets,
            args.max_new_tokens, passes_per_chunk=args.passes_per_chunk,
            ccot_passes=args.ccot_passes, num_chunks=args.num_chunks,
            dataset_path=args.dataset_path)
        all_results[label][task] = task_results
        all_samples[task] = task_samples

        print(f"  {'Bucket':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        print(f"  {'-'*40}")
        for bucket, r in task_results.items():
            if r["total"] > 0:
                print(f"  {bucket:<12} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>8.3f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(out_dir / "samples.json", "w") as f:
        json.dump(all_samples, f, indent=2)

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("model,task,bucket,correct,total,accuracy\n")
        for lbl, task_dict in all_results.items():
            for task, bucket_dict in task_dict.items():
                for bucket, r in bucket_dict.items():
                    f.write(f"{lbl},{task},{bucket},{r['correct']},{r['total']},{r['accuracy']:.4f}\n")

    print(f"\nResults saved → {out_dir}")

    # Guard against silently-empty evals: an all-zero results file looks like
    # a (bad) result; fail the job loudly instead.
    total_examples = sum(
        r["total"]
        for task_dict in all_results.values()
        for bucket_dict in task_dict.values()
        for r in bucket_dict.values()
    )
    if total_examples == 0:
        print("ERROR: 0 examples were evaluated across all tasks/buckets — "
              "results are empty. Check --dataset_path / network access.")
        sys.exit(1)


if __name__ == "__main__":
    main()
