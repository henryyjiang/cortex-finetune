"""
GSM8K evaluation for CortexGPT.

Few-shot chain-of-thought prompting on the GSM8K test set (1319 problems).
The model generates up to 256 tokens; we extract the final numeric answer
after "####" or fall back to the last number in the response.

--n_shot controls the exemplar count (default 8, the harness standard).  It is
a PROBE, not a tuning knob: sweeping it varies prompt length across the
max_length/cross_chunks bound the recipe ever trained on.

Every item is written to records.json with its index and correct flag, so two
runs can be compared pairwise rather than as two independent binomials.  s0 is
pinned per example (seed_example), so a rerun of the same checkpoint reproduces
-- it did not before 2026-09-16.

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
                         ccot_prime, greedy_generate, seed_example,
                         score_continuation)


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


def build_prompt(question: str, n_shot: int = 8) -> str:
    shots = [f"Question: {q}\nAnswer: {a}"
             for q, a in FEW_SHOT_EXAMPLES[:n_shot]]
    shots.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(shots)


_CALC = re.compile(r"<<[^>]*>>")


def gold_continuation(answer: str) -> str:
    """The gold CoT in the format the few-shot prompt demonstrates: GSM8K's
    <<43-5=38>> calculator annotations stripped, one leading space to follow
    "Answer:".  Stripping is applied identically at every --n_shot, so it
    cannot manufacture a trend across the sweep."""
    return " " + _CALC.sub("", answer).strip()


def extract_answer(text: str) -> Optional[str]:
    m = re.search(r"####\s*([\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "").strip() if numbers else None


def normalize(ans: str) -> str:
    """Canonicalise a numeric answer for exact comparison.

    The rstrip(".") is load-bearing.  extract_answer's fallback regex captures
    the sentence-final period out of "the answer is 8.", and without this the
    result scored as a miss against gold "8".  Three such cases turned up in
    800 sampled P0.5 failures -- roughly two questions per 500-item run, the
    same order as the heal-to-anneal trend the ladder was trying to measure.
    """
    ans = ans.replace(",", "").strip().rstrip(".")
    return ans.lstrip("0") or "0"


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
    p.add_argument("--n_shot",         type=int, default=8,
                   choices=range(0, len(FEW_SHOT_EXAMPLES) + 1),
                   metavar="[0-8]",
                   help="Few-shot exemplars in the prompt (default 8, the "
                        "harness standard). Lower values exist to test the "
                        "chunk-length hypothesis: the retrofit recipe never "
                        "shows the model more than max_length/cross_chunks "
                        "contiguous tokens (512 on the B2 arm), and the "
                        "8-shot prompt crosses that bound while the 4-shot one "
                        "does not. Accuracy that RECOVERS as the prompt drops "
                        "under the bound localises the GSM8K "
                        "floor to the chunk geometry rather than to memory or "
                        "to conversion budget. Sweep it; do not tune it.")
    p.add_argument("--score_only",     action="store_true",
                   help="Skip generation entirely: teacher-force the gold CoT "
                        "and report its NLL in nats/token. ONE forward per "
                        "example against a 256-token decode -- measured at "
                        "0.025 s/forward vs 23.8 s/example, ~950x -- and the "
                        "metric is continuous, so it resolves at an n where "
                        "3%-accuracy cannot. This is the powered and the cheap "
                        "way to run the --n_shot sweep: accuracy pinned at the "
                        "floor cannot show a break against prompt length, NLL "
                        "can.")
    p.add_argument("--ccot_passes",    type=int, default=0,
                   help="Mixed CCoT+CoT: run N silent full forward passes over "
                        "the prompt first, carrying M_cross between passes "
                        "(latent 'thinking'), then generate the CoT with the "
                        "primed buffer as read-only context. Requires a model "
                        "with cross state (K>0 or ccot_direct); 0 = off.")
    p.add_argument("--no_cache",       action="store_true",
                   help="Decode by re-forwarding the whole prefix each step "
                        "(the pre-2026-08-03 behaviour). ~2 orders of magnitude "
                        "slower; kept as the reference path for debugging.")
    p.add_argument("--max_examples",   type=int, default=0, help="0 = all")
    p.add_argument("--out_dir",        default="eval_results/gsm8k")
    p.add_argument("--dtype",          default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _answer_complete(new_text: str) -> bool:
    """CoT is done once a '####' line has been closed."""
    return "####" in new_text and "\n" in new_text.split("####")[-1]


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int,
             T: Optional[int], device: torch.device, seq_len: int = 2048,
             ccot_passes: int = 0, use_cache: bool = True) -> str:
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    max_prompt = seq_len - max_new_tokens
    if input_ids.shape[1] > max_prompt:
        input_ids = input_ids[:, -max_prompt:]
    input_ids = input_ids.to(device)

    num_steps = to_num_steps(T)
    # Mixed CCoT+CoT: latent multi-pass 'thinking' over the prompt before
    # explicit CoT generation.  m_cross stays fixed during generation — the CoT
    # tokens stay inside the window here (8-shot prompt + 256 new < seq_len), so
    # the model reads them from context and a frozen buffer costs nothing.  That
    # stops being true once prompt + CoT overflows the window, which is where a
    # write at segment boundaries becomes necessary.
    m_cross = ccot_prime(model, input_ids, num_steps, ccot_passes)
    return greedy_generate(model, tokenizer, input_ids, max_new_tokens,
                           num_steps, m_cross=m_cross, use_cache=use_cache,
                           stop_fn=_answer_complete)


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

    correct = 0
    total   = 0
    records = []
    nll_sum = 0.0
    n_tok   = 0

    for i, ex in enumerate(ds):
        gold_ans = extract_answer(ex["answer"])
        if gold_ans is None:
            continue
        prompt = build_prompt(ex["question"], args.n_shot)
        n_prompt_tok = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        # s0 is drawn fresh per forward (initialize_state), so the SAME weights
        # decode differently run to run: the P0.5 replicate of chkpt 305,000
        # moved 16/500 -> 19/500, with 22 of the top 50 predictions changed.
        # Pinning the draw from the example id makes a rerun reproducible and
        # makes arm-vs-arm contrasts paired, as eval_babilong.py already does.
        seed_example(f"gsm8k:{i}")
        if args.score_only:
            prompt_ids = tokenizer(prompt, add_special_tokens=False,
                                   return_tensors="pt").input_ids
            cont_ids   = tokenizer(gold_continuation(ex["answer"]),
                                   add_special_tokens=False,
                                   return_tensors="pt").input_ids
            sc = score_continuation(model, prompt_ids, cont_ids,
                                    to_num_steps(T))
            nll_sum += sc["nll_sum"]
            n_tok   += sc["n_tok"]
            total   += 1
            records.append({"idx": i, "gold": gold_ans,
                            "nll_sum": sc["nll_sum"], "n_tok": sc["n_tok"],
                            "entropy_first": sc.get("entropy_first"),
                            "prompt_tokens": n_prompt_tok})
            if total % 100 == 0:
                print(f"  {total}/{len(ds)}  "
                      f"nll/tok={nll_sum / max(n_tok, 1):.4f}")
            continue
        response = generate(model, tokenizer, prompt,
                            args.max_new_tokens, T, device,
                            ccot_passes=args.ccot_passes,
                            use_cache=not args.no_cache)
        pred_ans   = extract_answer(response)
        is_correct = pred_ans is not None and normalize(pred_ans) == normalize(gold_ans)
        correct += int(is_correct)
        total   += 1
        records.append({"idx": i, "question": ex["question"], "gold": gold_ans,
                        "pred": pred_ans, "correct": is_correct,
                        "prompt_tokens": n_prompt_tok})
        if total % 100 == 0:
            print(f"  {total}/{len(ds)}  acc={correct/total:.4f}")

    accuracy     = correct / total if total > 0 else 0.0
    nll_per_tok  = nll_sum / n_tok if n_tok > 0 else None
    tok          = sorted(r["prompt_tokens"] for r in records)
    median_tok   = tok[len(tok) // 2] if tok else 0
    if args.score_only:
        print(f"\nGSM8K gold-CoT NLL: {nll_per_tok:.4f} nats/token "
              f"over {total} examples, {n_tok} scored tokens")
    else:
        print(f"\nGSM8K accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"  n_shot={args.n_shot}  prompt tokens: median {median_tok}, "
          f"max {tok[-1] if tok else 0}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump({"mode": "score" if args.score_only else "generate",
                   "correct": correct, "total": total, "accuracy": accuracy,
                   "nll_per_token": nll_per_tok, "scored_tokens": n_tok,
                   "n_shot": args.n_shot,
                   "T": T if T is not None else cfg.mean_recurrence,
                   "ccot_passes": args.ccot_passes,
                   "median_prompt_tokens": median_tok}, f, indent=2)

    with open(out_dir / "summary.csv", "w") as f:
        f.write("correct,total,accuracy,nll_per_token,n_shot,"
                "median_prompt_tokens\n")
        f.write(f"{correct},{total},{accuracy:.4f},"
                f"{'' if nll_per_tok is None else f'{nll_per_tok:.6f}'},"
                f"{args.n_shot},{median_tok}\n")

    # EVERY item, not a truncated failure list.  Without a per-example id and a
    # correct flag nothing downstream can be PAIRED, and the deltas this
    # instrument has to resolve -- arm vs control, carry on vs off -- are a few
    # points at n=500, where McNemar on the same items is several times tighter
    # than two independent binomials.  failures.json stays for reading by eye.
    with open(out_dir / "records.json", "w") as f:
        json.dump(records, f, indent=2)

    with open(out_dir / "failures.json", "w") as f:
        json.dump([{k: r[k] for k in ("question", "gold", "pred")}
                   for r in records if not r.get("correct", True)][:50],
                  f, indent=2)

    print(f"Results saved → {out_dir}")


if __name__ == "__main__":
    main()
