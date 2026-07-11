"""
GSM8K evaluation for CortexGPT.

8-shot chain-of-thought prompting on the GSM8K test set (1319 problems).
The model generates up to 256 tokens; we extract the final numeric answer
after "####" or fall back to the last number in the response.

Dataset: openai/gsm8k  (main config, test split)

Usage:
    python evals/eval_gsm8k.py \
        --checkpoint runs/cortex-5b/checkpoint_0154441/checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from model_utils import (load_checkpoint, has_cross_state, to_num_steps,
                         ccot_prime)


# ---------------------------------------------------------------------------
# 8-shot prompt
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. #### 6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. #### 39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. #### 8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is "
        "4 more toys. 5 + 4 = 9. #### 9",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 = 29. #### 29",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 * 3 = 15 dollars. "
        "23 - 15 = 8. #### 8",
    ),
]


def build_prompt(question: str) -> str:
    shots = [f"Question: {q}\nAnswer: {a}" for q, a in FEW_SHOT_EXAMPLES]
    shots.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(shots)


def extract_answer(text: str) -> Optional[str]:
    m = re.search(r"####\s*([\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "").strip() if numbers else None


def normalize(ans: str) -> str:
    return ans.replace(",", "").strip().lstrip("0") or "0"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GSM8K evaluation for CortexGPT")
    p.add_argument("--checkpoint",     type=str, default=None,
                   help="Optional train.py .pt (finetuned weights overlaid strict=False).")
    p.add_argument("--model_name",     default="EleutherAI/pythia-160m")
    p.add_argument("--memory_slots",   type=int, default=None,
                   help="Override K; default reads memory_slots from the checkpoint config")
    p.add_argument("--T",              type=int, default=None,
                   help="Recurrence depth at eval (None = use checkpoint mean_recurrence)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--ccot_passes",    type=int, default=0,
                   help="Mixed CCoT+CoT: run N silent full forward passes over "
                        "the prompt first, carrying M_cross between passes "
                        "(latent 'thinking'), then generate the CoT with the "
                        "primed buffer as read-only context. Requires a model "
                        "with cross state (K>0 or ccot_direct); 0 = off.")
    p.add_argument("--max_examples",   type=int, default=0, help="0 = all")
    p.add_argument("--out_dir",        default="eval_results/gsm8k")
    p.add_argument("--dtype",          default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int,
             T: Optional[int], device: torch.device, seq_len: int = 2048,
             ccot_passes: int = 0) -> str:
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    max_prompt = seq_len - max_new_tokens
    if input_ids.shape[1] > max_prompt:
        input_ids = input_ids[:, -max_prompt:]
    input_ids = input_ids.to(device)

    num_steps = to_num_steps(T)
    # Mixed CCoT+CoT: latent multi-pass 'thinking' over the prompt before
    # explicit CoT generation.  m_cross stays fixed during generation.
    m_cross   = ccot_prime(model, input_ids, num_steps, ccot_passes)
    generated = input_ids
    eos_id    = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        out      = model(input_ids=generated, num_steps=num_steps,
                         m_cross_in=m_cross, return_m_cross=False)
        next_tok = out["logits"][0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
        generated = torch.cat([generated, next_tok], dim=1)
        if eos_id is not None and next_tok.item() == eos_id:
            break
        decoded_new = tokenizer.decode(generated[0, input_ids.shape[1]:])
        if "####" in decoded_new and "\n" in decoded_new.split("####")[-1]:
            break

    return tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)


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
    print(f"T={T if T is not None else cfg.mean_recurrence}  "
          f"ccot_passes={args.ccot_passes}")
    if args.ccot_passes > 0 and not has_cross_state(model):
        print("WARNING: --ccot_passes set but the model has no cross state "
              "(no M_cross / DirectCCoT) — the passes would be identical "
              "no-ops. Running as plain CoT.")

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    correct  = 0
    total    = 0
    failures = []

    for i, ex in enumerate(ds):
        gold_ans = extract_answer(ex["answer"])
        if gold_ans is None:
            continue
        response = generate(model, tokenizer, build_prompt(ex["question"]),
                            args.max_new_tokens, T, device,
                            ccot_passes=args.ccot_passes)
        pred_ans   = extract_answer(response)
        is_correct = pred_ans is not None and normalize(pred_ans) == normalize(gold_ans)
        if is_correct:
            correct += 1
        else:
            failures.append({"question": ex["question"], "gold": gold_ans, "pred": pred_ans})
        total += 1
        if total % 100 == 0:
            print(f"  {total}/{len(ds)}  acc={correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nGSM8K accuracy: {correct}/{total} = {accuracy:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump({"correct": correct, "total": total, "accuracy": accuracy}, f, indent=2)

    with open(out_dir / "summary.csv", "w") as f:
        f.write("correct,total,accuracy\n")
        f.write(f"{correct},{total},{accuracy:.4f}\n")

    with open(out_dir / "failures.json", "w") as f:
        json.dump(failures[:50], f, indent=2)

    print(f"Results saved → {out_dir}")


if __name__ == "__main__":
    main()
