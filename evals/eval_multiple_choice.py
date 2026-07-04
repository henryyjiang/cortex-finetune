"""
Multiple-choice evaluation for CortexGPT.

Covers: HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, PIQA.

Scoring: for each choice, compute the mean per-token log-probability of the
choice continuation given its context, then select the highest-scoring choice.

Usage:
    python evals/eval_multiple_choice.py \
        --checkpoint runs/cortex-5b/checkpoint_0154441/checkpoint.pt \
        --tasks hellaswag winogrande arc_easy arc_challenge piqa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model_utils import load_checkpoint, has_cross_state, to_num_steps


TASK_CHOICES = ["hellaswag", "winogrande", "arc_easy", "arc_challenge", "piqa"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Multiple-choice evaluation for CortexGPT")
    p.add_argument("--checkpoint",   type=str, default=None,
                   help="Optional train.py .pt (finetuned weights overlaid strict=False).")
    p.add_argument("--model_name",   default="EleutherAI/pythia-160m")
    p.add_argument("--memory_slots", type=int, default=None,
                   help="Override K; default reads memory_slots from the checkpoint config")
    p.add_argument("--T",            type=int, default=None,
                   help="Recurrence depth at eval (None = use checkpoint mean_recurrence)")
    p.add_argument("--tasks",        nargs="+", default=TASK_CHOICES, choices=TASK_CHOICES)
    p.add_argument("--max_examples", type=int, default=0, help="0 = all")
    p.add_argument("--seq_len",      type=int, default=2048)
    p.add_argument("--out_dir",      default="eval_results/multiple_choice")
    p.add_argument("--dtype",        default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Log-likelihood scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_prob_of_completion(
    model,
    context_ids: list[int],
    completion_ids: list[int],
    T: Optional[int],
    seq_len: int,
    device: torch.device,
) -> float:
    """Mean per-token log-prob of completion_ids given context_ids."""
    all_ids      = (context_ids + completion_ids)[-seq_len:]
    n_completion = min(len(completion_ids), len(all_ids))
    input_ids    = torch.tensor(all_ids, dtype=torch.long).unsqueeze(0).to(device)

    num_steps = to_num_steps(T)
    logits    = model(input_ids=input_ids, num_steps=num_steps)["logits"][0]  # [S, V]

    log_probs      = F.log_softmax(logits[:-1], dim=-1)          # [S-1, V]
    target         = input_ids[0, 1:]                             # [S-1]
    comp_log_probs = log_probs[-n_completion:].gather(
        1, target[-n_completion:].unsqueeze(1)
    ).squeeze(1)

    return comp_log_probs.mean().item()


# ---------------------------------------------------------------------------
# Dataset loaders → list of (context_str, [choice_str, ...], correct_idx)
# ---------------------------------------------------------------------------

def load_hellaswag(max_examples: int):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    return [
        (ex["activity_label"] + ": " + ex["ctx"], ex["endings"], int(ex["label"]))
        for ex in ds
    ]


def load_winogrande(max_examples: int):
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    examples = []
    for ex in ds:
        sentence  = ex["sentence"]
        blank_pos = sentence.find("_")
        context   = sentence[:blank_pos]
        rest      = sentence[blank_pos + 1:]
        completions = [ex["option1"] + rest, ex["option2"] + rest]
        examples.append((context, completions, int(ex["answer"]) - 1))
    return examples


def load_arc(config: str, max_examples: int):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", config, split="validation")
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    examples = []
    for ex in ds:
        labels = ex["choices"]["label"]
        try:
            correct = labels.index(ex["answerKey"])
        except ValueError:
            continue
        examples.append((ex["question"], ex["choices"]["text"], correct))
    return examples


def load_piqa(max_examples: int):
    from datasets import load_dataset
    # The canonical "piqa" repo ships a loading script (piqa.py), which datasets
    # >= 4.x refuses to execute ("Dataset scripts are no longer supported").
    # Load the auto-generated parquet conversion instead — same goal/sol1/sol2/
    # label schema, no remote code.
    ds = load_dataset("ybisk/piqa", "default", split="validation",
                      revision="refs/convert/parquet")
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    return [(ex["goal"], [ex["sol1"], ex["sol2"]], int(ex["label"])) for ex in ds]


LOADERS = {
    "hellaswag":     lambda n: load_hellaswag(n),
    "winogrande":    lambda n: load_winogrande(n),
    "arc_easy":      lambda n: load_arc("ARC-Easy", n),
    "arc_challenge": lambda n: load_arc("ARC-Challenge", n),
    "piqa":          lambda n: load_piqa(n),
}


# ---------------------------------------------------------------------------
# Task evaluation
# ---------------------------------------------------------------------------

def run_task(task_name, examples, model, tokenizer, T, seq_len, device):
    correct = 0
    for i, (context, choices, label) in enumerate(examples):
        ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
        scores  = []
        for choice in choices:
            comp_ids = tokenizer(choice, add_special_tokens=False).input_ids
            if not comp_ids:
                scores.append(float("-inf"))
                continue
            scores.append(log_prob_of_completion(model, ctx_ids, comp_ids, T, seq_len, device))
        if max(range(len(scores)), key=lambda j: scores[j]) == label:
            correct += 1
        if (i + 1) % 200 == 0:
            print(f"  [{task_name}] {i+1}/{len(examples)}  acc={correct/(i+1):.4f}")
    accuracy = correct / len(examples) if examples else 0.0
    return {"correct": correct, "total": len(examples), "accuracy": accuracy}


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

    all_results = {}
    for task in args.tasks:
        print(f"\n{'='*55}\nTask: {task}\n{'='*55}")
        examples = LOADERS[task](args.max_examples)
        result   = run_task(task, examples, model, tokenizer, T, args.seq_len, device)
        all_results[task] = result
        print(f"  {task}: {result['correct']}/{result['total']} = {result['accuracy']:.4f}")

    print(f"\n{'Task':<18} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("-" * 45)
    for task, r in all_results.items():
        print(f"{task:<18} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>8.4f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(out_dir / "summary.csv", "w") as f:
        f.write("task,correct,total,accuracy\n")
        for task, r in all_results.items():
            f.write(f"{task},{r['correct']},{r['total']},{r['accuracy']:.4f}\n")

    print(f"\nResults saved → {out_dir}")


if __name__ == "__main__":
    main()
