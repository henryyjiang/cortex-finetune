# Final pair — the last link (drafted 2026-10-21)

Both arms died on SIGBUS (`rc=135`) at Oct-20 13:31, ~20h short of the 48h
wall, at 95.7% (control) and 98.0% (cortex) of the 38,147-update horizon.
Nothing is wrong with the runs. Each needs one more short link.

| arm | resume from | updates left | est. train | + fast-fwd | total |
|---|---|---:|---:|---:|---:|
| control | `checkpoint_36600` † | 1,547 | 14.3h | ~3h | **~17h** |
| cortex  | `checkpoint_37200`   |   947 |  7.1h | ~2.5h | **~10h** |

† needs verifying first — see step 1.

---

## Step 1 — verify the control's `checkpoint_36600` (login node, seconds)

Its save was never logged: the last `Saved checkpoint` line is 36,000 and the
logged tail stops at update **36,494**, *below* 36,600. It is most likely real
(all three SIGBUS logs are truncated **mid-line** — a block-buffer flush
boundary — so the real stop runs past the logged tail, and 106 updates of lag
is inside the ~150 seen before). But `watch_run.py:528` only counts
directories, and `train.py:916-918` creates the dir *before* `torch.save` and
prints only *after* it, so a crash mid-save leaves exactly this state.

```bash
cd ~/cortex-finetune && python3 - <<'EOF'
import os, zipfile
run = "cortex-5b/c-chunked"
for s in (36000, 36600):
    p = f"{run}/checkpoint_{s}/chkpt.pt"
    if not os.path.isfile(p):
        print(f"checkpoint_{s}: NO chkpt.pt  <- unusable")
        continue
    sz = os.path.getsize(p)
    try:
        n = len(zipfile.ZipFile(p).namelist())
        print(f"checkpoint_{s}: {sz/2**30:.2f} GiB, zip OK ({n} entries)")
    except Exception as e:
        print(f"checkpoint_{s}: {sz/2**30:.2f} GiB, TRUNCATED -> {type(e).__name__}")
EOF
```

`torch.save` writes a zip, so a truncated file throws on the central
directory. Stdlib only — no torch import, no conda env, no allocation, and
none of the 13 GB `torch.load` that the login node's cgroup kills.

- **Both `zip OK` at about the same size** → use `checkpoint_36600`.
- **36600 missing or TRUNCATED** → use `checkpoint_36000`. Costs 600 updates
  (~5.6h), nothing else.

A bad checkpoint fails loudly at `train.py:922` (bare `torch.load`, no
existence check, no fallback), so the downside of guessing wrong is a wasted
queue slot, not a silent corruption.

## Step 2 — resubmit (control first)

```bash
cd ~/cortex-finetune && PHASE=mix \
  RESUME_PATH=cortex-5b/c-chunked/checkpoint_36600 \
  STOP_AT_STEP=38146 \
  sbatch pace/control_5b.sbatch
```

```bash
cd ~/cortex-finetune && PHASE=mix \
  RESUME_PATH=cortex-5b/cortex-final/checkpoint_37200 \
  ZNORM_TARGET=78.9 \
  STOP_AT_STEP=38146 \
  sbatch pace/cortex_final_5b.sbatch
```

- `RESUME_PATH` **must** be non-empty. It is a submit-time env var
  (`final_5b_common.sh:118`) with no newest-checkpoint fallback — if it is
  empty, `CHAIN_ARGS=""` and train.py trains from the base graft at step 0
  into the live run dir.
- `ZNORM_TARGET=78.9` carried forward, as measured on this lineage. ‖E‖
  finished the link at 81.6, i.e. +3.4% and converging toward it.
- The cortex script prints a warning if a `c5b` job is in the queue. It is a
  warning, not a refusal — the documented overlap is fine.
- `squeue -u $USER` before concluding anything ended. An epilog in the log does
  not mean the job is gone; these links have been requeued under the same job
  ID before, and a silent requeue re-runs with whatever `RESUME_PATH` the
  environment had.

### Why `STOP_AT_STEP=38146` (my addition, not from the log's next-link line)

Left at `0`, the link ends via `train.py:2540` — `max_steps` reached, which
breaks **without** calling `save_checkpoint` and goes straight to
`save_model_only("final_checkpoint")`. That is an HF dir (config.json +
safetensors, **no chkpt.pt**), so:

- the newest *resumable* checkpoint at the end of the run would be **37,800**,
  and the last 347 updates would exist only in a model-only dir;
- `pace/midrun_check.sbatch:146` scores `checkpoint_$STEP/chkpt.pt` and
  explicitly cannot load a model-only dir (this is the `STEP=23100` failure,
  every eval `rc=1`).

`STOP_AT_STEP` takes the other branch (`train.py:2549`), which *does* write a
resumable checkpoint — the code comment says that is the point — and prints
`[stop_at_step] reached N`, the only clean "exited on its own terms" signal.
`max_steps` stays 38,147, so the LR cosine and the recurrence ramp are
untouched.

It must be **below** `max_steps`: the `max_steps` check runs first, so
`STOP_AT_STEP=38147` would be preempted and save nothing. 38,146 is one update
short of the horizon and gives a scorable, branchable `checkpoint_38146`.

Drop it if you would rather read out at 37,800 — a 0.9% shortfall, immaterial
either way. The gap is real, the fix is optional.

## Step 3 — read out

```bash
ARM=control STEP=38146 sbatch pace/midrun_check.sbatch
ARM=cortex  STEP=38146 ZNORM_TARGET=78.9 sbatch pace/midrun_check.sbatch
```

(`STEP=37800` if you drop `STOP_AT_STEP`.)

---

## Still to come off the cluster — one trip, on the login node

Neither is needed to resubmit; both are needed for the write-up. Both are
git-tracked, so they come back to the laptop by the same ff-merge as the logs.

```bash
cd ~/cortex-finetune
module load anaconda3 && conda activate cortex-retro

# 1. wandb: stale since Oct 19 08:01, which predates the final link entirely.
#    --project is REQUIRED (the tool defaults to cortex-retro-ft, and a bare
#    run refreshes that one and leaves cortex-5b stale).  --all_attempts is
#    REQUIRED too: every link is its own run id, and the crashed ones are only
#    included with it.  Takes a few minutes; writes wandb_exports/cortex-5b/.
python tools/pull_wandb_metrics.py --project cortex-5b --all_attempts

# 2. the mechanism trace.  CORTEX ONLY -- the control's diag is empty by
#    design ("no cortex, nothing to report"), so there is no c-chunked file.
cp cortex-5b/cortex-final/cortex_diag.jsonl \
   logs/final5b_diag/cortex-final_at_37500.jsonl

git add wandb_exports/cortex-5b logs/final5b_diag
git commit -m "logs: cortex-5b wandb through 14178513, cortex diag at 37500"
git push
```

`cortex_diag.jsonl` is cumulative across links (770 rows ≈ 37,500/50 at
`diag_interval=50`), so the one copy carries the whole run. The newest local
one is `cortex-final_at_23100.jsonl` (Oct 12) and only `watch_run`'s summary of
the current state is in the log — the raw rows are the only record of the
mechanism at the final step, and they are what settles the carry-rank question
below.

Worth doing **before** the last link rather than after: if anything has gone
wrong with the mechanism, better to see it now than to find out having spent
the link. Repeat both after the link finishes for the final numbers.

## Two things for the write-up

- **Clip engagement crossed the 50% line.** Cortex clipped on **55.4%** of
  steps (1,718/3,100), mean ‖g‖ **2.302**, max **264.4** — up from 10% → 23%
  on earlier links. The control over the same stretch is **0.4%**, mean 0.506,
  max 4.855, so this is arm-specific, not the setup. Still zero non-finite,
  and the LR-cooldown explanation still fits. Not worth changing
  `max_grad_norm` with 2.5% of the horizon left, but it is a caveat on the
  between-arm comparison: over this stretch cortex's updates are scaled by
  ~1/2.3 on more than half of steps while the control's are essentially
  unclipped, so the arms are not getting matched effective step sizes.
- **`watch_run`'s carry-rank warning prints the wrong numbers.**
  `tools/watch_run.py:456` triggers on `pr[-1] < 0.5 × mean(first quarter)` but
  displays `pr[0] → pr[-1]`, and `pr[0]=1.73` is a warm-up outlier. So "halved
  (1.73 → 5.43)" is self-contradictory as printed. The rank did fall — the
  quarter mean must have been above 10.9 — but confirm the direction against
  the raw `e_eff_rank_pr` series before quoting it.
