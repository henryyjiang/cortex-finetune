# Mid-run read-out ledger — the 5B pair (`cortex-final` vs `c-chunked`)

Running record of every mid-run check on the 5B pair: what was scored, what the
numbers were, and what each one is confounded by. Append a block per read-out.
Never overwrite a row — a superseded number gets a note, not a deletion.

---

## Read-out index

| # | date | STEP | jobs | arm(s) | evals | outcome |
|---|------|------|------|--------|-------|---------|
| 1 | 2026-10-03 | 11444 | — | both | val NLL + full 2x2 | **OK** — the baseline every later row pairs against |
| 2 | 2026-10-12 | 23100 | 13989859 / 13989860 | both | val NLL + full 2x2 | **FAILED, rc=1, zero evals** — bad checkpoint path (see §2) |

---

## 1. Baseline — step 11,444 (the heal boundary), T=8

Scored 2026-10-03 off `cortex-5b/<run>/checkpoint_11444`.
Trained under **`znorm_target = 98.0`** and the **heal** corpus — both change at
step 11,450. This row is the last checkpoint of the old regime.

### 1a. Arm vs arm, val NLL (nats; lower better)

`chunk1` is the no-carry chunk — neither arm has anything to carry there, so it
prices the graft itself. `chunks2plus` is where the carry exists.

| pack | slice | cortex-final | c-chunked | cortex − control |
|------|-------|-------------:|----------:|-----------------:|
| fwedu | all         | 2.5850 | 2.6533 | **−0.0683** (cortex ahead) |
| fwedu | chunk1      | 2.7088 | 2.6694 | +0.0394 (cortex behind) |
| fwedu | chunks2plus | 2.5674 | 2.6511 | **−0.0837** (cortex ahead) |
| pg19  | all         | 3.4256 | 3.4906 | **−0.0650** (cortex ahead) |
| pg19  | chunk1      | 3.5840 | 3.5348 | +0.0492 (cortex behind) |
| pg19  | chunks2plus | 3.4029 | 3.4843 | **−0.0814** (cortex ahead) |

The shape is coherent on both packs: cortex **pays ~0.04–0.05 nats on chunk1**
and **wins ~0.08 nats on chunks2+**, netting ~0.065–0.068. That is the carry
paying for the graft, and it is the headline claim of the pair.

**Caveat — these CIs are unpaired and wide** (±0.026 to ±0.055 on `all`). Both
chunk1 deficits sit inside CI overlap, so "cortex is behind on chunk1" is a
consistent sign across two packs, not yet a significant single measurement.
`--baseline` pairing is what tightens this, which is why §2b matters.

### 1b. The 2x2 — E and Z decomposed

Sign convention: **positive = that channel helps.**

| effect | carry (n=300) | pg19 (n=400) |
|--------|--------------:|-------------:|
| cells E1Z1 / E1Z0 / E0Z1 / E0Z0 | 2.4884 / 2.4860 / 2.6853 / 2.8003 | 3.4029 / 3.4750 / 3.7487 / 3.9301 |
| **E_main** (E0Z1−E1Z1, Z on) | **+0.1969** | **+0.3457** |
| **Z_main / X_off** (E1Z0−E1Z1, E on) | **−0.0024** | **+0.0721** |
| Z_alone (E0Z0−E0Z1, E off) | +0.1150 | +0.1815 |
| **interaction** | **+0.1175** | **+0.1094** |
| **X_content** (donor−own) | −0.0005 | **+0.0771** |
| **content share** (X_content/X_off) | 18.5% (of ~0 — meaningless) | **107.0%** |
| carry-local Z_main (n=100) | −0.0323 | — |

**How to read this row:**

- **E does nearly all the work.** E_main is +0.197 / +0.346; Z's marginal
  contribution with E already on is −0.002 / +0.072. Quote E_main for E, never
  the E1Z1-vs-E0Z0 together-number (+0.3119 at this step).
- **Z's contents are load-bearing on pg19, and only on pg19.** X_content
  +0.0771 at 107% share means substituting a donor Z costs as much as removing
  Z entirely — the band holds real content, not just a column. On carry both
  X_off and X_content are ~0, so Z contributes nothing measurable there.
- **The interaction is POSITIVE on both packs → E and Z SUBSTITUTE.** This is
  the sign the design did *not* want; the hypothesis was negative
  (complements: E worth more when Z is present). Z recovers +0.115/+0.182 on
  its own with E off, then is largely redundant once E is on.
- **Z hurts local tokens**: carry-local Z_main −0.0323.

---

## 2. Step 23,100 — 2026-10-12 — FAILED, no evals produced

Jobs 13989859 (control) / 13989860 (cortex). Both `rc=1`, walltime 17s / 21s.
**Every eval raised `FileNotFoundError` before loading weights. There are no new
Z-vs-E numbers from this read-out.** The only new data is the mechanism trace
(§3), which is written by training, not by the eval.

### 2a. Root cause — `model_only_chkpt_*` can never be a `--checkpoint`

`midrun_check.sbatch:130-134` prefers the weights-only dir:

```sh
CKPT="$RUN_DIR/model_only_chkpt_$STEP"
[ -d "$CKPT" ] || CKPT="$RUN_DIR/checkpoint_$STEP"
```

with the comment *"Prefer the weights-only dir — it exists at twice the
resolution and this job needs no optimizer state."* **That comment is wrong.**
`save_model_only()` (`train.py:873`) writes via `model.save_pretrained()` — an
HF dir holding `config.json` + safetensors and **no `chkpt.pt`**.
`load_checkpoint()` (`evals/model_utils.py:183-189`) requires `chkpt.pt` inside
a dir and raises otherwise. So that branch is unloadable by construction, and
`j5_joint.sbatch:168-169` already says so in as many words: *"`% save_interval`
writes save_model_only (a save_pretrained dir, NO chkpt.pt, useless to the
read-out)"*.

**Why the baseline passed and this one didn't — step-number luck.**
`SAVE_INTERVAL=300`, resumable every 600.

| STEP | % 300 | % 600 | `model_only` dir? | `checkpoint_` dir? | result |
|------|------:|------:|---|---|---|
| 11444 | 44 | 44 | no | **yes** (fault/timeout save) | fell through to the good branch → **worked** |
| 23100 | **0** | 300 | **yes** | **no** | took the bad branch → **failed** |

11,444 is not a multiple of 300, so no `model_only` dir existed and the
fallback saved it. 23,100 is 77×300, so the dir existed and was handed to
`--checkpoint`. The guard in `model_utils.py` did its job — this would
otherwise have scored the **base weights** and reported healthy numbers.

Worse, `23100 % 600 = 300`, so there is **no resumable checkpoint at 23,100
either**. Fixing the branch order alone would not rescue this step.

### 2b. Second, quieter bug — control's baseline never pairs

Control's baseline is on disk at `eval_results/midrun/c-chunked/step_11444/`,
but the sbatch computes `BASE_RES=.../step_11444_T${T}` → `step_11444_T8`. The
pairing is then gated on `[ -f "$BASE_RES/val_nll_$KEY.json" ]`
(`midrun_check.sbatch:260`) with **no else branch**, so control silently runs
unpaired and loses the paired CI. The cortex baseline dir carries the `_T8`
suffix; control's predates the T-keying change and was never renamed.

### 2c. The arms are no longer at the same step

| arm | newest resumable | % of horizon | diag last step |
|-----|------------------|-------------:|---------------:|
| cortex-final | `checkpoint_26531` | 69.5% | 26,550 |
| c-chunked | `checkpoint_23349` | 61.2% | — (no mechanism) |

Cortex is ~3,200 optimizer steps ahead — expected, it draws more updates per
link. **Any paired read-out must pick a step both arms hold**, so the ceiling
is control's 23,349.

### 2d. The fix

Score the largest multiple of 600 that both arms hold: **22,800** (38×600).

```bash
ARM=control STEP=22800 sbatch pace/midrun_check.sbatch
ARM=cortex  STEP=22800 ZNORM_TARGET=78.9 sbatch pace/midrun_check.sbatch
```

Before submitting, three repo-side fixes:

1. **Drop the `model_only` branch** in `midrun_check.sbatch:130-134` — use
   `checkpoint_$STEP` only, and fail loudly when it is absent (the existing
   `ls` error block at :149 already prints what is present).
2. **Rename** `eval_results/midrun/c-chunked/step_11444` →
   `step_11444_T8` so control pairs.
3. **Make the missing baseline loud** — add an `else echo "WARN: no baseline at
   $BASE_RES, running UNPAIRED"` at :260.

---

## 3. Mechanism trace at 23,100 / 26,550 — the one thing that did land

From `logs/final5b_diag/cortex-final_at_23100.jsonl` (547 rows, steps 50→26,550).
Architecture checks all pass: `n_depths 9`, `embed_frozen 1`,
`e_carried_norm 78.494`, `embed_ratio 1.0051`, `znorm_target 78.9`, zero
dead rows, no mix, stride 1, E read on, no e_dropout.

### 3a. The `‖E‖ −14.3%` WARN is a FALSE ALARM — resolved

The job warned *"‖E‖ has moved −14.3% across the trace… still climbing is an
instability"* and said to compare thirds before deciding. Doing that:

| third | steps | mean ‖E‖ | sd |
|-------|-------|---------:|---:|
| 1 | 50 – 9,100 | 78.541 | 3.651 |
| 2 | 9,150 – 18,100 | 77.738 | 1.301 |
| 3 | 18,150 – 26,550 | 77.736 | 1.121 |

Thirds 2 and 3 are **identical to three decimals** and the sd is collapsing
(3.65 → 1.12). The entire −14.3% is the launch transient in the first ~5
records: 92.59 (step 50) → 59.82 (step 250) → recovered to ~78 by step 600.
**Excluding steps <2,000, the whole trace moves −1.42%** (n=508, mean 78.18,
sd 1.43), and the last 20 records sit at 78.3–80.2 against the 78.9 target —
locked on, ±1.5%.

**‖E‖ is settled, not climbing.** Note the WARN is computed first-record vs
last-record over a trace that always includes launch, so **it will fire on
every future read-out forever** and look worse as the trace grows. It should
exclude the warmup window or compare thirds itself.

### 3b. What actually drifted, 11,444 → 26,550

Mean over the 1,000 steps ending at each point (n=20 / n=21 records).

| metric | @11444 | @26550 | Δ | note |
|--------|-------:|-------:|---:|------|
| loss | 2.5746 | 2.5490 | −0.0255 | **nearly flat over ~15k steps** |
| fg_at_bias | 0.3427 | 0.1822 | **−46.8%** | E forget gate closing hard |
| ig_at_bias | 0.7564 | 0.9052 | **+19.7%** | E input gate opening |
| fg_z_at_bias | 0.6047 | 0.4589 | −24.1% | Z forget gate closing |
| ig_z_at_bias | 0.7142 | 0.8458 | +18.4% | Z input gate opening |
| gate_mem_w_norm | 69.11 | 80.39 | +16.3% | E write pathway growing |
| **gate_mem_z_w_norm** | 89.55 | 119.78 | **+33.7%** | **Z's gate growing 2x faster than E's** |
| summary_emb_norm | 14.57 | 16.39 | +12.5% | |
| z_write_grad_frac | 0.8313 | 0.7560 | −9.1% | by design; J6 measured 0.76 |
| z_read_grad_frac | 1.000 | 1.000 | 0 | read fully in-graph |
| z_e_spliced_norm | 78.91 | 79.20 | +0.4% | settled |
| z_row_norm | 21.55 | 21.82 | +1.2% | |
| z_over_e_norm | 0.2723 | 0.2764 | +1.5% | Z stays ~27% of E |
| z_eff_rank_pr | 6.71 | 6.60 | −1.6% | |
| e_eff_rank_pr | 7.50 | 7.82 | +4.3% | |
| n_depths (median) | 6 | 6 | 0 | collapsed-to-1 on 5.0% → 4.8% |
| z_tape_len | 8.00 | 8.76 | +9.5% | |

**The reading:** both arms of the gate pair are moving the same direction —
forget closing, input opening — i.e. the model is turning over memory faster
and leaning on fresher writes. **Z's write gate is growing ~2x faster than E's
(+33.7% vs +16.3%)**, which says the Z pathway is still actively learning at
69% of the horizon and has not saturated. Geometry (norms, ranks, Z/E ratio,
depth map) is stable throughout.

None of this says whether Z *helps* — that is the 2x2, and it did not run.

---

## 4. Confounds on any 11,444 → 22,800 comparison

Three things change together at step 11,450, so a Z-vs-E delta across that
boundary is **not** a clean "did Z grow" read:

1. **More training** — the intended variable.
2. **Corpus switch** — heal → mix, at `HEAL_END = 11444`.
3. **`znorm_target` 98.0 → 78.9** — a **load-bearing forward-pass constant**,
   not a watch value: `rescale_rows(z, latent_read_znorm_target)` at
   `cortex_graft.py:1521, 1581, 1583`. It rescales both own and donor Z, so it
   moves X_off *and* X_content.

The re-target was deliberate (‖E‖ settled to ~78.5, so the launch value of 98.0
left the band broken upward at 1.253) and `midrun_check.sbatch` guards the
rebuild correctly — `ZNORM_TARGET` is mandatory on the cortex arm, is passed to
`--set latent_read_znorm_target`, and is cross-checked against the diag record
at the scored step. That machinery worked on job 13989860. But it means the
baseline row is scored at a **different read strength** than every later row.

**Consequence:** read a 22,800 vs 11,444 change in X_content/content share as
*regime + training*, not training alone. The clean within-regime baseline will
be the first successful post-11,450 read-out — i.e. 22,800 becomes the new
reference for everything after it.

---

## 5. Open questions carried forward

- **The interaction sign.** Positive (substitutes) on both packs at 11,444,
  against the design's hypothesis. Does it move at all under the mix corpus, or
  is substitution the stable answer for this architecture?
- **Does X_content survive the re-target?** 107% share on pg19 at
  `znorm_target=98.0`. The band is re-mapped by the lo=1..9 widening and the
  read strength has changed; the value does not transfer as-is.
- **Why is loss flat?** −0.0255 nats over ~15k steps while gate norms move
  16–34%. The mechanism is still learning; the loss is not moving with it.
- **Does the chunk1 deficit close?** Cortex paid ~0.04–0.05 nats there at
  11,444. If the carry is improving, chunk1 should hold flat while chunks2+
  pulls further ahead.
