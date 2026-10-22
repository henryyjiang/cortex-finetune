# Final pair — the read-out lines (drafted 2026-10-22)

Everything to run once the last link lands, in order.  Steps 0–2 are minutes
and gate the rest; step 4 is the benchmarks.

`STEP` below is **38146** if the last link was submitted with
`STOP_AT_STEP=38146`, and **37800** if it was left to end on `max_steps`.
Step 0 tells you which.  Every later line takes it as `STEP=<that>`.

---

## Step 0 — did the link land, and on what (login node, seconds)

```bash
cd ~/cortex-finetune && squeue -u $USER
```

```bash
cd ~/cortex-finetune && for R in c-chunked cortex-final; do
  echo "=== $R"
  echo -n "  newest resumable: "
  for d in cortex-5b/$R/checkpoint_*; do [ -f "$d/chkpt.pt" ] && echo "${d##*/}"; done \
    | sort -t_ -k2 -n | tail -3 | tr '\n' ' '; echo
  ls -ld cortex-5b/$R/final_checkpoint/config.json 2>/dev/null \
    || echo "  final_checkpoint: ABSENT"
  ls -ld "$(ls -t cortex-5b/$R/checkpoint_*/chkpt.pt | head -1)"
done
```

**`final_checkpoint` IS A TRAP, and this is the line that catches it.**  It is
written by `save_model_only` after `train()` returns (train.py:2636), so it
exists on BOTH arms already — from the **heal link**, which ended cleanly at
step 11,444.  A link killed by SIGBUS never returns and never refreshes it.  So
if the last link also died, `final_checkpoint` is silently the **11,444-step
model**, every benchmark below would score it, and nothing in those jobs would
say so.  The two timestamps above must be within minutes of each other.

If `final_checkpoint` is stale, the benchmarks cannot run as written: they take
an HF dir and there is no tool here that exports one from a `chkpt.pt`.  Resume
one more short link with `STOP_AT_STEP` instead — reaching it is what writes it.

The mid-run check (step 1) does not care: it loads `checkpoint_$STEP/chkpt.pt`
directly and is unaffected by any of this.

---

## Step 1 — the final mid-run check, both arms

```bash
cd ~/cortex-finetune && ARM=control STEP=38146 sbatch pace/midrun_check.sbatch
```

```bash
cd ~/cortex-finetune && ARM=cortex STEP=38146 ZNORM_TARGET=78.9 sbatch pace/midrun_check.sbatch
```

- `STEP` must be one the run wrote a `chkpt.pt` for — step 0's listing.  The
  job refuses with the available list rather than failing inside an eval.
- `ZNORM_TARGET=78.9` is checked against `cortex_diag.jsonl` at that step
  before anything loads, so a wrong value is a refusal, not a bad number.
- `BASELINE_STEP` defaults to 11,444, the heal boundary, which is the pairing
  every post-switch number is quoted against.  Leave it.
- `N_CHUNKS` defaults to 8 and that is correct for BOTH arms: the common recipe
  sets `CROSS_CHUNKS=8` and neither arm overrides it.
- ~4 h wall, H200, one GPU each.  They can run at the same time.

Expect the numbers to move in the third decimal, not the second: `train/lr` is
at 0.29% of peak and the remaining lr-integral is 0.003% of the run's total.

---

## Step 2 — the logs off the cluster (login node, one trip)

```bash
cd ~/cortex-finetune && module load anaconda3 && conda activate cortex-retro && \
  python tools/pull_wandb_metrics.py --project cortex-5b --all_attempts
```

```bash
cd ~/cortex-finetune && cp cortex-5b/cortex-final/cortex_diag.jsonl \
  logs/final5b_diag/cortex-final_at_38146.jsonl && \
  git add wandb_exports/cortex-5b logs/final5b_diag && \
  git commit -m "logs: cortex-5b wandb and cortex diag at the horizon" && git push
```

- `--project cortex-5b` is **required**: the tool defaults to `cortex-retro-ft`
  and a bare run refreshes that one and leaves cortex-5b stale.
- `--all_attempts` is **required**: every link is its own run id and the
  crashed ones (which is most of them) are only included with it.
- The diag copy is **cortex only** — the control's is empty by design.
- `git pull` on the cluster first if you want the `created_at` column the
  summary now carries; without it the stitcher orders by first step, which is
  correct for this chain (checked — no rollbacks) but cannot prove it is.

Running the pull **locally** instead works and skips the git round trip for
`wandb_exports/` — the credentials come from `wandb login` / `~/.netrc`:

```bash
cd "/Users/paff/Documents/Projects/Cortex Refs/cortex-finetune" && \
  /opt/miniconda3/envs/huginn/bin/python tools/pull_wandb_metrics.py \
  --project cortex-5b --all_attempts
```

---

## Step 3 — one contiguous run out of the links (local, seconds)

```bash
cd "/Users/paff/Documents/Projects/Cortex Refs/cortex-finetune" && \
  python3 tools/stitch_wandb_links.py --project cortex-5b \
  --pair c-chunked cortex-final \
  --diag logs/final5b_diag/cortex-final_at_38146.jsonl
```

Writes, under `wandb_exports/cortex-5b/stitched/`:

| file | what it is |
|---|---|
| `c-chunked.csv`, `cortex-final.csv` | the contiguous per-step series, same columns as the exports plus `link_id`/`link_index` |
| `<run>__links.csv` | per link: claimed range, kept range, rows rolled back |
| `paired_c-chunked_vs_cortex-final.csv` | per-step `cortex − control`, joined on step |
| `cortex-final_at_38146__dedup.jsonl` | the diag with the re-walked steps removed |

Both arms stitch to **no gaps** as of the 2026-10-21 export (control 1..36,612,
cortex 1..37,517), after dropping 936 and 1,047 rolled-back steps respectively.
The paired mix-phase lead reproduces at **−0.0623 nats** (n=25,169, t=−906).

Why a tool and not `cat`: the links overlap, and the overlap is not duplicate
data.  A link that died at 17,017 was resumed from checkpoint_16,800, so
16,801..17,017 were trained twice and only the second traversal is on the
lineage that reached the horizon.  Concatenating re-traverses them;
de-duplicating on first-seen keeps the orphaned branch.  The tool gives each
step to the link that resumed last and prints what it dropped.

The diag needs the same treatment for the same reason — `cortex_diag.jsonl`
appends across links, so it holds 20 duplicated steps and is **not monotone**
in `step`.  Anything that reads it as a time series reads those twice.

---

## Step 4 — the benchmarks, both arms

### 4a. Prepare the checkpoints once, and READ what it prints (login node)

```bash
cd ~/cortex-finetune && module load anaconda3 && conda activate cortex-retro && \
  python tools/prepare_eval_checkpoint.py --base ckpts/olmo-retrofit-cortex \
  --dirs cortex-5b/c-chunked/final_checkpoint cortex-5b/cortex-final/final_checkpoint
```

Both sbatch scripts run this themselves, but running it here first is the only
chance to see the `flags:` line before spending the GPU.  The eval jobs take
**no `--set` overrides** — they rebuild the graft from `config.json` alone, so
that line is the whole contract:

- `cortex-final` must show `latent_read_znorm_target: 78.9`,
  `latent_depth_lo: 1`, `latent_depth_hi: 9`, `latent_embed_frozen: true`,
  `latent_encoding: staggered_state`, `accum_vecs: 16`, `gate_slots: 64`,
  `accum_max: 128`, `e_carry_read: true`, `latent_carry_read: true`.
- `c-chunked` must show `use_memory: false` and **no `latent_*` keys at all**.
- Both will show `mean_recurrence: 32`, inherited from the untrained retrofit
  base.  That is why `T_OVERRIDE=8` is on every line below and is not optional:
  without it the evals run at 4× the depth these arms trained at, for 4× the
  wall clock, and report it as the arm.

Also confirm the packs are there (they are, from the b2 round):

```bash
cd ~/cortex-finetune && ls -d data/babilong-1k data/LongMemEval
```

### 4b. Basic — MC suite + GSM8K

```bash
cd ~/cortex-finetune && OUT_ROOT=cortex-5b PREP_BASE=ckpts/olmo-retrofit-cortex \
  RUN=c-chunked T_OVERRIDE=8 EVAL_TAG=5b-final sbatch pace/eval_basic.sbatch
```

```bash
cd ~/cortex-finetune && OUT_ROOT=cortex-5b PREP_BASE=ckpts/olmo-retrofit-cortex \
  RUN=cortex-final T_OVERRIDE=8 EVAL_TAG=5b-final sbatch pace/eval_basic.sbatch
```

hellaswag / winogrande / arc_easy / arc_challenge / piqa, then GSM8K at 500
examples, 8-shot.  ~4 h each on an A100 of a 16 h wall; GSM8K is ~90% of it.
GSM8K is dropped from the claim map — these lines collect it anyway, as asked.
`SKIP_GSM8K=true` on both lines cuts them to ~25 minutes if you change your
mind.

### 4c. Long memory — BABILong qa1–qa3 + LongMemEval

```bash
cd ~/cortex-finetune && OUT_ROOT=cortex-5b PREP_BASE=ckpts/olmo-retrofit-cortex \
  RUN=c-chunked T_OVERRIDE=8 SEQ_LEN=512 NUM_CHUNKS=8 EVAL_TAG=5b-final \
  sbatch pace/eval_longcontext.sbatch
```

```bash
cd ~/cortex-finetune && OUT_ROOT=cortex-5b PREP_BASE=ckpts/olmo-retrofit-cortex \
  RUN=cortex-final T_OVERRIDE=8 SEQ_LEN=512 NUM_CHUNKS=8 EVAL_TAG=5b-final \
  sbatch pace/eval_longcontext.sbatch
```

`SEQ_LEN=512 NUM_CHUNKS=8` is the arms' own training geometry
(`max_length=4096 / cross_chunks=8`) and is **not** the script's default — it
defaults to Track-A's 1024/4, which would write the buffer at twice the trained
window.  Both arms get the same chunking: that is what makes the contrast the
memory and nothing else.

Defaults that are already right and should be left alone: `BABILONG_MAX=250`
with `babilong-1k-samples`, `LME_MAX=0` (all 500), `SCORE_NLL=true`,
`RANK_ANSWERS=true`, `PMI=true`.  Those are the four knobs that make the
carry effect resolvable at all — n=100/cell unpaired has an MDE of ~3.1 pt.

~2.5 h each at this geometry (measured: job 12691489, the b2 T8/nc8/sl512 run,
2:27:41), so no sharding and no `-t` override.  The control is cheaper still.

Labels land at `eval_results/longcontext_5b-final/{c-chunked,cortex-final}-T8-nc8-sl512/`
and `eval_results/basic_5b-final/{c-chunked,cortex-final}/`.

### 4d. Two optional additions — not run above

Neither is in the "both models" ask; both are cheap and both answer a question
the four jobs above raise.

```bash
# the within-model carry-off control: same weights, same chunking, buffer never
# built.  The only no-memory reference that does not also vary the lineage, and
# what tools/analyze_longcontext_pairs.py runs McNemar on.  ~2.5 h.
cd ~/cortex-finetune && OUT_ROOT=cortex-5b PREP_BASE=ckpts/olmo-retrofit-cortex \
  RUN=cortex-final T_OVERRIDE=8 SEQ_LEN=512 NUM_CHUNKS=8 NO_CARRY=true \
  EVAL_TAG=5b-final sbatch pace/eval_longcontext.sbatch
```

```bash
# the answer-prior floor: no priming at all, so every accuracy above has a
# chance line to sit on.  Minutes, not hours.
cd ~/cortex-finetune && OUT_ROOT=cortex-5b PREP_BASE=ckpts/olmo-retrofit-cortex \
  RUN=cortex-final T_OVERRIDE=8 SEQ_LEN=512 NUM_CHUNKS=8 BLANK_CONTEXT=true \
  EVAL_TAG=5b-final sbatch pace/eval_longcontext.sbatch
```

---

## What to expect, written down first

- **The control will score near the floor on BABILong.** It has no carry, so
  under 8-chunk priming it answers from the final 512-token chunk alone. That
  is the contrast working, not the job failing — but it means the headline is
  a floor-vs-signal comparison, and 4d's blank-context line is what turns it
  into a number.
- **`accum_max=128` holds exactly 8 chunks.** Examples under 4,096 tokens fit
  the trained regime exactly; longer ones chunk into more 512-token windows and
  the FIFO keeps the newest 8. That fires on 3 of 6 BABILong buckets and all of
  LongMemEval. It is the trained geometry, and raising it is a separate sweep
  (the b2 round ran am1024 / am2048 at 37 and 58 minutes).
- **Clipping is not a caveat against cortex.** Cortex clips on ~56% of steps to
  the control's 0.4%, but at the median update the clip coefficient is 0.923
  and the effective step cortex applies is 1.8× the control's. Report the
  median, not the tail-dominated mean.
- **Do not pool the paired loss across step 11,444.** The corpus moves there.
  Step 3's report splits it for that reason.
