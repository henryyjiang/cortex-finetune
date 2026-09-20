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
                         prime_cross_state, greedy_generate, ccot_prime,
                         score_continuation, seed_example, rank_candidates,
                         buffer_geometry)
from model_utils import parse_config_overrides  # noqa: E402


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
                   choices=[f"qa{i}" for i in range(1, 21)],
                   help="BABILong tasks. Only qa1-qa3 have few-shot demos "
                        "here; qa4+ get the instruction alone (build_suffix "
                        "falls back to no demos), which is a DIFFERENT prompt "
                        "regime -- do not pool qa4+ with qa1-qa3 unless demos "
                        "have been added for them.")
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
    p.add_argument("--no_carry", action="store_true",
                   help="Paired no-memory control: chunk the context exactly as "
                        "usual, but never build the cross-chunk buffer, so the "
                        "model sees the SAME final window with nothing carried "
                        "into it. The only difference from the normal run is the "
                        "memory, on the same weights -- unlike a base-model "
                        "comparison, which also varies the lineage.")
    p.add_argument("--max_examples",  type=int, default=500,
                   help="Max examples per task/length bucket (0 = all). "
                        "RMT-team/BABILong holds only 100 rows per "
                        "(task, length); ask for more than a bucket has and "
                        "the run WARNS and reports the short count rather "
                        "than silently capping. --dataset_repo "
                        "RMT-team/babilong-1k-samples has ~1000 rows.")
    p.add_argument("--length_buckets", nargs="+", type=int,
                   default=[1000, 2000, 4000, 8000, 16000, 32000])
    p.add_argument("--out_dir",       default="eval_results/babilong")
    p.add_argument("--dataset_path",  default=None,
                   help="Local path to pre-downloaded BABILong snapshot (snapshot_download). "
                        "Required on nodes without internet access. Both known "
                        "layouts are auto-detected: data/<task>/<length>.json "
                        "(RMT-team/BABILong) and <length>/<task>-*.parquet "
                        "(RMT-team/babilong-1k-samples).")
    p.add_argument("--dataset_repo",  default="RMT-team/BABILong",
                   help="Hub repo used when --dataset_path is absent.")
    p.add_argument("--accum_max",    type=int, default=None,
                   help="Override the carry buffer's FIFO cap (config.accum_max). "
                        "The buffer keeps only the newest accum_max/accum_vecs "
                        "chunks; B2 trained at 256/32 = 8 chunks = 4096 tokens, "
                        "and train.py asserts the cap is never exceeded, so the "
                        "FIFO trim branch NEVER fires in training and always "
                        "fires at eval past that horizon. Raising it here is an "
                        "out-of-distribution sequence length for the prefix -- "
                        "that is the point of the probe, not an oversight.")
    p.add_argument("--records",       default=None,
                   help="Write one JSON line per example (id, bucket, correct, "
                        "gold NLL, prediction) to this path. REQUIRED for the "
                        "paired carry-on/carry-off analysis: summary.csv keeps "
                        "counts only, and counts cannot say WHICH examples "
                        "flipped. tools/analyze_longcontext_pairs.py reads it.")
    p.add_argument("--rank_answers",  action="store_true",
                   help="Also score every candidate answer and rank them, "
                        "instead of relying only on free generation + "
                        "containment. The candidate set is derived from the "
                        "cell's own target column, so chance is 1/|set| and is "
                        "reportable. Immune to the EOS/length collapse that "
                        "containment scoring is not.")
    p.add_argument("--pmi",           action="store_true",
                   help="With --rank_answers, also score every candidate with "
                        "the story stripped (demos + question only), so the "
                        "analysis can rank on log P(a|context) - log P(a|no "
                        "context). Removes the answer prior, which is what the "
                        "base model's flatness across a 32x context range says "
                        "these tasks are actually being answered from.")
    p.add_argument("--blank_context", action="store_true",
                   help="Answer-prior control: same demos and question, no "
                        "story. Skips priming entirely (minutes, not hours) "
                        "and gives the floor that every accuracy number in "
                        "this eval should be read against.")
    p.add_argument("--score_nll",     action="store_true",
                   help="Also score the gold answer's teacher-forced NLL "
                        "(one extra forward per example). Continuous and "
                        "paired, so it resolves effects far below what 0/1 "
                        "containment accuracy can at any affordable n.")
    p.add_argument("--no_seed_per_example", action="store_true",
                   help="Disable per-example RNG seeding. Seeding (the "
                        "default) makes the s0 draw identical across "
                        "conditions so the paired contrast isolates the "
                        "carry; it also makes runs reproducible, but it does "
                        "NOT reproduce pre-2026-09 runs, which were unseeded.")
    p.add_argument("--set", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="force a graft-building config flag, e.g. "
                        "--set use_memory=true --set prefix_memory=gated.  "
                        "REQUIRED on an OVERLAY checkpoint: train.py's "
                        "checkpoint_<step> dirs, and the _w16 branch dirs cut "
                        "from them, hold chkpt.pt and NO config.json, so "
                        "--model_name loads the BASE dir whose config carries "
                        "no cortex flags at all (use_memory is literally "
                        "'<absent>' on ckpts/olmo-retrofit-cortex) and without "
                        "these the graft builds with no buffer.  Mirror the "
                        "arm's PROBE_SETS in pace/p1_arms.sbatch.  RED 12.")
    p.add_argument("--dtype",         default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
# Two hub repos, two layouts, and the choice between them decides how much
# statistical power the eval can reach:
#
#   RMT-team/BABILong            data/<task>/<length>.json      100 rows/cell
#   RMT-team/babilong-1k-samples <length>/<task>-*.parquet     ~1000 rows/cell
#
# The 100-row cap is why every long-context table so far is n=100 per bucket:
# nothing was truncating, the file simply ran out.  Anything asking for more
# than 100 per bucket must use the 1k-samples repo.

def load_bucket(task: str, cfg: str, dataset_path: Optional[str],
                dataset_repo: str):
    """Return a streaming iterable of rows for one (task, length) cell."""
    from datasets import load_dataset

    if dataset_path is not None:
        local = Path(dataset_path)
        parquet = sorted(local.glob(f"{cfg}/{task}-*.parquet"))
        if parquet:
            return load_dataset("parquet",
                                data_files=[str(p) for p in parquet],
                                split="train", streaming=True)
        js = local / "data" / task / f"{cfg}.json"
        if js.exists():
            return load_dataset("json", data_files=str(js), split="train",
                                streaming=True)
        # A missing local file is a setup error, not a skippable bucket --
        # silent skipping here is how an entire eval once produced all-zero
        # "results" without anyone noticing.
        raise FileNotFoundError(
            f"BABILong cell not found for {task}/{cfg} under {local}.\n"
            f"Looked for {local}/{cfg}/{task}-*.parquet (babilong-1k-samples) "
            f"and {js} (BABILong).\n"
            f"Run `python evals/download_datasets.py` on a login node and pass "
            f"the directory it reports via --dataset_path."
        )

    if "1k-samples" in dataset_repo:
        return load_dataset(
            "parquet",
            data_files=f"hf://datasets/{dataset_repo}/{cfg}/{task}-00000-of-00001.parquet",
            split="train", streaming=True)
    # Load the task/length file directly. Config auto-resolution on this repo
    # is unreliable (it tries to materialise every task x length combination,
    # some of which don't exist).
    return load_dataset(
        "json",
        data_files=f"hf://datasets/{dataset_repo}/data/{task}/{cfg}.json",
        split="train", streaming=True)


def collect_candidates(rows) -> list:
    """The distinct gold answers in one (task, length) cell.

    Derived from the data rather than hardcoded so this works for any qa task
    (qa1-qa3 are the six bAbI rooms; qa4+ are different sets entirely).  Both
    conditions get the identical set, and |set| is the chance level, which is
    the number the containment scores should have been read against all along:
    qa2 at 12.5% and qa3 at 6.7% are BELOW uniform guessing over six rooms.
    """
    seen = []
    for ex in rows:
        t = str(ex.get("target", ex.get("answer", ""))).strip()
        if t and t not in seen:
            seen.append(t)
    return sorted(seen)


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
             num_chunks=0, no_carry=False, example_id=None, score_nll=False,
             candidates=None, pmi=False, blank_context=False):
    suffix = build_suffix(task, question)
    if blank_context:
        # Answer-prior control: the model sees the demos and the question and
        # nothing else, so there is no context to prime on and no carry to
        # build. Anything it scores here it knew before reading the story.
        context = ""
    prime_chunks, final_ids = split_context(tokenizer, context, suffix,
                                            seq_len, max_new_tokens,
                                            num_chunks=num_chunks)
    num_steps = to_num_steps(T)
    # Prime the cross state on every context chunk before the final one — the
    # final chunk is the prediction pass.  Models without cross state carry
    # nothing and see only the final chunk (the no-memory control).
    # no_carry drops the buffer while leaving split_context untouched, so the
    # final window is byte-identical to the carry run and the delta isolates
    # the memory.  Priming is skipped entirely (not zeroed) -- the carry-off
    # condition in eval_carry_ablation.py is the same: m_cross_in=None.
    m_cross = (None if no_carry else
               prime_cross_state(model, prime_chunks, num_steps,
                                 passes_per_chunk=passes_per_chunk))
    # Optional latent CCoT: extra silent passes over the question chunk,
    # seeded with the context-primed buffer, before answering.
    m_cross = ccot_prime(model, final_ids, num_steps, ccot_passes,
                         m_cross_init=m_cross)
    # Seed AFTER priming: the carry-on condition has consumed RNG that the
    # carry-off condition has not, so seeding earlier would not align the s0
    # draws that generation and scoring depend on.
    if example_id is not None:
        seed_example(example_id)
    scores = None
    if score_nll:
        gold_ids = torch.tensor(
            tokenizer(" " + str(answer).strip(),
                      add_special_tokens=False).input_ids,
            dtype=torch.long).unsqueeze(0)
        scores = score_continuation(model, final_ids, gold_ids, num_steps,
                                    m_cross=m_cross,
                                    eos_id=tokenizer.eos_token_id)
        if example_id is not None:
            seed_example(example_id)   # scoring consumed a draw; realign
    ranked = ranked_nc = None
    if candidates:
        ranked = rank_candidates(model, tokenizer, final_ids, candidates,
                                 num_steps, m_cross=m_cross)
        if example_id is not None:
            seed_example(example_id)
        if pmi:
            # The prior denominator: same demos and question, no story and no
            # carry. Identical in the carry-on and carry-off runs by
            # construction -- it shifts both conditions, and changes the argmax,
            # which is the point.
            nc_ids = torch.tensor(
                tokenizer(suffix, add_special_tokens=False).input_ids,
                dtype=torch.long).unsqueeze(0)
            ranked_nc = rank_candidates(model, tokenizer, nc_ids, candidates,
                                        num_steps, m_cross=None)
            if example_id is not None:
                seed_example(example_id)
    # Can the answer be read straight off the window the model predicts from?
    # If so the carry is irrelevant on this example and it only dilutes the
    # contrast; the paired analysis stratifies on this.
    final_text = tokenizer.decode(final_ids[0], skip_special_tokens=True)
    gold_in_window = contains_answer(final_text, answer)
    pred = greedy_generate(model, tokenizer, final_ids, max_new_tokens,
                           num_steps, m_cross=m_cross, stop_on_newline=True)
    return (contains_answer(pred, answer), pred, scores, len(prime_chunks),
            ranked, ranked_nc, gold_in_window)


# ---------------------------------------------------------------------------
# Task loop
# ---------------------------------------------------------------------------

def run_task(task_name, model, tokenizer, T, seq_len, max_examples, length_buckets,
             max_new_tokens, passes_per_chunk=1, ccot_passes=0, num_chunks=0,
             no_carry=False, dataset_path=None, dataset_repo="RMT-team/BABILong",
             records_fh=None, score_nll=False, seed_per_example=True,
             rank_answers=False, pmi=False, blank_context=False, geom=None):
    # BABILong uses config name for context length (e.g. '1k', '4k') and
    # split for the task (e.g. 'qa1'). Load each bucket config separately.
    config_names = [f"{b // 1000}k" for b in length_buckets]
    results = {cfg: {"correct": 0, "total": 0} for cfg in config_names}
    samples = []
    short = []

    for cfg in config_names:
        try:
            ds = load_bucket(task_name, cfg, dataset_path, dataset_repo)
        except FileNotFoundError:
            raise
        except Exception as e:
            # Network flake on one bucket shouldn't kill the whole job; the
            # all-zero guard in main() still fails the run if nothing loads.
            print(f"  [{task_name}/{cfg}] ERROR loading — {e}")
            continue

        candidates = None
        if rank_answers:
            # One extra read of the cell to collect its answer set. The files
            # are 100-1000 rows, so this is cheap; streaming means re-opening.
            try:
                candidates = collect_candidates(
                    load_bucket(task_name, cfg, dataset_path, dataset_repo))
            except Exception as e:
                print(f"  [{task_name}/{cfg}] could not build a candidate set — {e}")
            if candidates:
                print(f"  [{task_name}/{cfg}] {len(candidates)} candidates "
                      f"(chance {100.0 / len(candidates):.1f}%): "
                      f"{', '.join(candidates)}")

        seen = 0
        for row_idx, ex in enumerate(ds):
            ctx      = ex.get("input", ex.get("context", ex.get("text", "")))
            question = ex.get("question", "")
            answer   = str(ex.get("target", ex.get("answer", "")))
            if not ctx or not question or not answer:
                continue

            # Stable across runs and across the carry-on/carry-off conditions:
            # the row index within the cell, NOT the count of accepted rows, so
            # a row skipped in one run cannot shift the ids of everything after
            # it and silently mis-pair the two conditions.
            example_id = f"{task_name}/{cfg}/{row_idx}"
            ok, pred, scores, n_prime, ranked, ranked_nc, gold_in_window = eval_one(
                model, tokenizer, ctx, question, answer, T,
                seq_len, max_new_tokens, task_name,
                passes_per_chunk=passes_per_chunk,
                ccot_passes=ccot_passes,
                num_chunks=num_chunks,
                no_carry=no_carry,
                example_id=example_id if seed_per_example else None,
                score_nll=score_nll, candidates=candidates, pmi=pmi,
                blank_context=blank_context)
            if ok:
                results[cfg]["correct"] += 1
            results[cfg]["total"] += 1
            seen += 1

            if records_fh is not None:
                rec = {"id": example_id, "task": task_name, "bucket": cfg,
                       "correct": bool(ok), "gold": answer, "pred": pred,
                       "n_prime_chunks": n_prime,
                       "pred_words": len(pred.split()),
                       "stopped_on_newline": "\n" in pred,
                       "gold_in_final_window": bool(gold_in_window)}
                if geom is not None:
                    # chunks_kept < chunks_written = the FIFO dropped writes on
                    # this example, i.e. the carry never contained the earlier
                    # context at all.  A column, not an afterthought.
                    n_vec, max_vecs, held = geom
                    rec["buffer_chunks_held"] = held
                    rec["chunks_evicted"] = max(0, n_prime - held)
                if ranked is not None:
                    rec["candidates"] = [c["text"] for c in ranked]
                    rec["cand_nll"] = [c["nll_sum"] for c in ranked]
                    rec["cand_ntok"] = [c["n_tok"] for c in ranked]
                    if ranked_nc is not None:
                        rec["cand_nll_nocontext"] = [c["nll_sum"] for c in ranked_nc]
                if scores and scores.get("n_tok"):
                    rec["gold_nll_sum"] = scores["nll_sum"]
                    rec["gold_n_tok"] = scores["n_tok"]
                    rec["gold_nll_per_tok"] = scores["nll_sum"] / scores["n_tok"]
                    rec["p_eos_first"] = scores.get("p_eos_first")
                    rec["entropy_first"] = scores.get("entropy_first")
                records_fh.write(json.dumps(rec) + "\n")

            if seen <= 5:   # first 5 per length bucket, for debuggability
                samples.append({"bucket": cfg, "question": question,
                                "gold": answer, "pred": pred, "correct": ok})

            if seen % 50 == 0:
                print(f"  [{task_name}/{cfg}] {seen} examples processed...")

            if max_examples > 0 and seen >= max_examples:
                break

        if max_examples > 0 and seen < max_examples:
            short.append((cfg, seen))

    if short:
        # The n=100 tables were produced by a --max_examples 500 job: nothing
        # truncated them, RMT-team/BABILong simply holds 100 rows per cell.
        # Say so instead of leaving it to be rediscovered from the totals.
        cells = ", ".join(f"{cfg}={n}" for cfg, n in short)
        print(f"  WARNING [{task_name}]: asked for {max_examples}/bucket, "
              f"dataset ran out at: {cells}")
        print(f"           RMT-team/BABILong holds 100 rows per (task, length). "
              f"Use --dataset_repo RMT-team/babilong-1k-samples for more.")

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
    overrides = parse_config_overrides(args.set)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name,
                                 args.memory_slots, dtype, device,
                                 accum_max=args.accum_max,
                                 config_overrides=overrides or None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    T = args.T
    print(f"T={T if T is not None else cfg.mean_recurrence}  tasks={args.tasks}  "
          f"num_chunks={args.num_chunks}  passes_per_chunk={args.passes_per_chunk}  "
          f"ccot_passes={args.ccot_passes}  no_carry={args.no_carry}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    # run dir name as label; fall back to the model dir when no overlay
    # checkpoint was given (--checkpoint defaults to None: eval the prepared dir
    # as-is) so we don't do Path(None).
    label = (Path(args.checkpoint).parent.parent.name
             if args.checkpoint else Path(args.model_name).name)
    all_results[label] = {}

    records_path = Path(args.records) if args.records else out_dir / "records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_fh = open(records_path, "w", encoding="utf-8")

    all_samples: dict = {}
    for task in args.tasks:
        print(f"\n--- {task} ---")
        task_results, task_samples = run_task(
            task, model, tokenizer, T, args.seq_len,
            args.max_examples, args.length_buckets,
            args.max_new_tokens, passes_per_chunk=args.passes_per_chunk,
            ccot_passes=args.ccot_passes, num_chunks=args.num_chunks,
            no_carry=args.no_carry, dataset_path=args.dataset_path,
            dataset_repo=args.dataset_repo, records_fh=records_fh,
            score_nll=args.score_nll,
            seed_per_example=not args.no_seed_per_example,
            rank_answers=args.rank_answers, pmi=args.pmi,
            blank_context=args.blank_context, geom=buffer_geometry(model))
        all_results[label][task] = task_results
        all_samples[task] = task_samples

        print(f"  {'Bucket':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        print(f"  {'-'*40}")
        for bucket, r in task_results.items():
            if r["total"] > 0:
                print(f"  {bucket:<12} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>8.3f}")

    records_fh.close()
    print(f"\nPer-example records → {records_path}")

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
