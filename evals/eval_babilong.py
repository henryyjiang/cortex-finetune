"""
BABILong evaluation for CortexGPT.

Accuracy vs. context length on BABILong QA1/QA2/QA3.
The model reads the context in seq_len-token chunks, carrying M_cross across
chunks, then predicts the answer token at the end.

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

import torch
from transformers import AutoTokenizer

from model_utils import load_checkpoint, has_cross_state, to_num_steps


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

def encode_and_chunk(tokenizer, context: str, question: str, seq_len: int):
    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    q_ids   = tokenizer(f"\nQuestion: {question}\nAnswer:", add_special_tokens=False).input_ids
    all_ids = ctx_ids + q_ids
    chunks  = []
    for start in range(0, len(all_ids), seq_len):
        chunk = all_ids[start : start + seq_len]
        chunks.append(torch.tensor(chunk, dtype=torch.long).unsqueeze(0))
    return chunks


# ---------------------------------------------------------------------------
# Single-example evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_one(model, tokenizer, context, question, answer, T, seq_len) -> bool:
    chunks = encode_and_chunk(tokenizer, context, question, seq_len)
    device = next(model.parameters()).device
    num_steps = to_num_steps(T)

    # Prime the cross state on every chunk EXCEPT the final one — the final
    # chunk is the prediction pass and must not read a summary of itself.
    # Models without cross state skip priming: nothing is carried, so the
    # prediction depends only on the final chunk.
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
# Task loop
# ---------------------------------------------------------------------------

def run_task(task_name, model, tokenizer, T, seq_len, max_examples, length_buckets,
             dataset_path=None):
    from datasets import load_dataset

    # BABILong uses config name for context length (e.g. '1k', '4k') and
    # split for the task (e.g. 'qa1'). Load each bucket config separately.
    config_names = [f"{b // 1000}k" for b in length_buckets]
    results = {cfg: {"correct": 0, "total": 0} for cfg in config_names}

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

            if eval_one(model, tokenizer, ctx, question, answer, T, seq_len):
                results[cfg]["correct"] += 1
            results[cfg]["total"] += 1
            seen += 1

            if seen % 50 == 0:
                print(f"  [{task_name}/{cfg}] {seen} examples processed...")

            if max_examples > 0 and seen >= max_examples:
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
    print(f"T={T if T is not None else cfg.mean_recurrence}  tasks={args.tasks}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    # run dir name as label; fall back to the model dir when no overlay
    # checkpoint was given (--checkpoint defaults to None: eval the prepared dir
    # as-is) so we don't do Path(None).
    label = (Path(args.checkpoint).parent.parent.name
             if args.checkpoint else Path(args.model_name).name)
    all_results[label] = {}

    for task in args.tasks:
        print(f"\n--- {task} ---")
        task_results = run_task(task, model, tokenizer, T, args.seq_len,
                                args.max_examples, args.length_buckets,
                                dataset_path=args.dataset_path)
        all_results[label][task] = task_results

        print(f"  {'Bucket':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        print(f"  {'-'*40}")
        for bucket, r in task_results.items():
            if r["total"] > 0:
                print(f"  {bucket:<12} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>8.3f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

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
