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
| 3 | 2026-10-13 | 22800 | 13994793 / 13994794 | both | val NLL + full 2x2 | **OK, rc=0 both arms** — first complete post-switch read-out (see §3) |

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

## 3. Step 22,800 — 2026-10-13 — OK, both arms, T=8

Jobs 13994793 (control) / 13994794 (cortex), both **`rc=0`**, ~21 min each.
Scored off `cortex-5b/<run>/checkpoint_22800`, `0 missing / 0 unexpected keys`
on both arms. Trained under **`znorm_target = 78.9`** and the **mix** corpus.
Diag at the scored step: `n_depths 7`, `embed_frozen 1`,
`e_carried_norm 75.880`, `embed_ratio 1.0398` — in band.

Both repo fixes are confirmed working: the resumable checkpoint resolved, and
**control paired for the first time** ("PAIRED against c-chunked@11444/fwedu,
400 shared rows"), which it had been silently skipping.

### 3a. Within-arm, paired vs 11,444 (positive = worse now)

| pack | slice | cortex-final | c-chunked |
|------|-------|-------------:|----------:|
| fwedu | all | +0.0188 [+0.0147,+0.0228] | +0.0067 [+0.0024,+0.0106] |
| fwedu | chunk1 | +0.0272 [+0.0209,+0.0335] | +0.0062 [+0.0001,+0.0121] |
| pg19 | all | **−0.5300** [−0.5577,−0.5009] | **−0.5622** [−0.5849,−0.5390] |
| pg19 | chunk1 | −0.5915 [−0.6296,−0.5548] | −0.5970 [−0.6332,−0.5609] |

The corpus switch did exactly what the script's pre-written bar predicted:
**PG-19 down hard (~0.55 nats), FineWeb-Edu up a little.** Not degradation —
this is the mix pack arriving. Both arms show it, so it is the corpus, not the
mechanism.

### 3b. Arm vs arm — THE LEAD NARROWED

| pack | slice | cortex | control | cortex−control | @11444 | change |
|------|-------|-------:|--------:|---------------:|-------:|-------:|
| fwedu | all | 2.6038 | 2.6601 | **−0.0563** | −0.0683 | lead −0.012 |
| fwedu | chunk1 | 2.7360 | 2.6755 | +0.0604 | +0.0394 | deficit +0.021 |
| fwedu | chunks2plus | 2.5849 | 2.6578 | **−0.0729** | −0.0837 | lead −0.011 |
| pg19 | all | 2.8956 | 2.9284 | **−0.0328** | −0.0651 | **lead HALVED** |
| pg19 | chunk1 | 2.9925 | 2.9378 | +0.0547 | +0.0492 | deficit +0.006 |
| pg19 | chunks2plus | 2.8818 | 2.9271 | **−0.0453** | −0.0814 | lead −0.036 |

The qualitative shape **holds**: cortex still pays on chunk1 (no carry) and
still wins on chunks2+. But the margin is shrinking on both packs, and on PG-19
it roughly halved. The control captured more of the mix corpus's gain
(−0.5622 vs −0.5300). **This is the headline tension of this read-out** —
Z's internal contribution grew (§3c) while the between-arm advantage fell.

### 3c. The 2x2 — 11,444 → 22,800

`*` = 95% CI excludes zero. Sign convention: positive = that channel helps.

#### carry (n=300, score=carry)

| | @11444 | @22800 | Δ |
|---|---:|---:|---:|
| cells E1Z1/E1Z0/E0Z1/E0Z0 | 2.4884 / 2.4860 / 2.6853 / 2.8003 | 2.5598 / 2.6752 / 2.8639 / 3.2757 | |
| E_main | +0.1969\* | +0.3041\* | +0.1072 |
| **X_off** (= Z_main) | −0.0024\* | **+0.1154\*** | **+0.1178** |
| **X_content** | −0.0005 | **−0.0006** | −0.0002 |
| **content share** | 18.5% (of ~0) | **−0.6%** | |
| Z_alone | +0.1150\* | +0.4118\* | +0.2968 |
| interaction | +0.1175\* | +0.2965\* | +0.1790 |
| carry-local Z_main | −0.0323 | −0.0236 | +0.0087 |

**This is the pre-J7 signature, and the script named it in advance:** *"Zero
here with X_off positive is the pre-J7 signature: columns load-bearing,
contents irrelevant."* X_off appeared from nothing (−0.002 → +0.115) while
X_content stayed pinned at zero and **not significant**. On the synthetic carry
task Z's columns became worth 0.115 nats as **free capacity**, carrying nothing
retrievable. Z still hurts local tokens (−0.024).

#### pg19 (n=400, score=all)

| | @11444 | @22800 | Δ |
|---|---:|---:|---:|
| cells E1Z1/E1Z0/E0Z1/E0Z0 | 3.4029 / 3.4750 / 3.7487 / 3.9301 | 2.8818 / 3.0070 / 3.1612 / 3.3313 | |
| E_main | +0.3457\* | +0.2794\* | −0.0663 |
| **X_off** (= Z_main) | +0.0721\* | **+0.1252\*** | **+0.0531** (+74%) |
| **X_content** | +0.0771\* | **+0.1295\*** | **+0.0523** (+68%) |
| **content share** | 107.0% | **103.4%** | |
| Z_alone | +0.1815\* | +0.1702\* | −0.0113 |
| **interaction** | +0.1094\* | **+0.0449 (n.s.)** | −0.0644 |

**On natural text Z's contribution nearly doubled and it is real content.**
X_off and X_content rose together by almost exactly the same amount (+0.053 /
+0.052), holding the share at ~100% — substituting a donor Z still costs as
much as deleting Z. The band is holding more content than it did, not just more
column.

**The interaction halved and its CI now includes zero** ([−0.003, +0.089]).
**But do not read this as a finding** — the interaction is
`(E0Z0 − E1Z0) − (E0Z1 − E1Z1)`, so it uses BOTH E-off cells and sits in the
untrustworthy tier of §3d. The honest statement is: *we cannot measure whether
E and Z complement or substitute in this run.* The apparent move toward the
design's hypothesis (negative = complements) is not evidence for it.

**The two packs disagree in direction** on E_main (carry +0.107, pg19 −0.066)
and on the interaction (carry +0.179, pg19 −0.064). The synthetic carry task
and natural text are telling different stories; §6 carries that forward.

### 3d. CAVEAT — the E-off cells are out of distribution in this run

`pace/cortex_final_5b.sbatch:205` sets `--cortex.e_dropout 0.0`, and the diag
confirms it was never overridden: `z_e_dropout` is 0.0 and `z_e_dropped` is 0
across all 547 records. **This run was never trained with E switched off.**

`tools/prepare_carry_task.py` says so explicitly about its own instrument:
`Z_alone` is *"only in-distribution for a model trained with
--cortex.e_dropout > 0"*. The same applies to anything else computed from an
E-off cell. So the five effects split into two tiers:

| effect | cells used | trustworthy? |
|---|---|---|
| **X_off**, **X_content** | E1Z1, E1Z0 — **E always on** | **YES** |
| E_main | needs E0Z1 | no — E-off is OOD |
| Z_alone | needs E0Z0, E0Z1 | no — OOD, and the task doc says so |
| interaction | needs all four | no — inherits the OOD cells |

This is why carry's E0Z0 jumps to 3.2757 and `Z_alone` to +0.4118: those cells
ask the model to run in a configuration it has never seen, and the resulting
NLLs are inflated.

**Consequence for §3b's reading.** The "E_main + X_off is flat on pg19
(0.4178 → 0.4046), so Z gained what E lost" inference leans on E_main, which is
in the untrustworthy tier. Treat it as a hypothesis worth one check, not a
measured result. **The narrowing between-arm lead in §3b stands on its own** —
it is a plain NLL contrast between two independently trained models with no
ablation in it, and needs no E-off cell.

To make the E-side effects measurable at all, a future arm would need
`e_dropout > 0`. That is an architecture change and
[[cortex-final-z-architecture-frozen]] applies; it is not a mid-run fix.

### 3d-bis. "E only" vs "Z only" — the cells exist, the test does not

Asked directly: is there an E-only vs Z-only comparison? The 2x2 **does** carry
both one-channel cells — `E1Z0` is E alone, `E0Z1` is Z alone — and they can be
contrasted (computed here from the saved `per_sample` arrays, 10k bootstrap):

| contrast | carry @22800 | pg19 @22800 |
|---|---:|---:|
| E alone vs neither | +0.6006 [+0.589,+0.612] | +0.3243 [+0.287,+0.357] |
| Z alone vs neither | +0.4118 [+0.403,+0.421] | +0.1702 [+0.125,+0.209] |
| **E alone vs Z alone** | **+0.1887** [+0.180,+0.198] | **+0.1542** [+0.145,+0.164] |
| both vs E alone (= X_off) | +0.1154 [+0.110,+0.121] | +0.1252 [+0.120,+0.132] |
| both vs Z alone (= E_main) | +0.3041 [+0.296,+0.312] | +0.2794 [+0.266,+0.295] |

Taken at face value E alone beats Z alone by ~0.15–0.19 nats. **But every row
except `both vs E alone` touches an E-off cell, so only that one row is
trustworthy** (§3d).

**The real E-only vs Z-only test is a TRAINING design, not an eval.** J5 did it
properly with three separately trained limbs — `evals/score_j5.py:70-73`:

```
zonly  trained with E's spliced rows zeroed  -> its cell is E0*
eonly  trained with Z's columns zeroed       -> its cell is E1Z0
both   trained with both live                -> its cell is E1Z1
```

and that file states the governing rule outright: *"score a limb in another
limb's cell and the number is an out-of-distribution condition reported as the
arm's result."* That is exactly what the 5B mid-run check does when it switches
E off, because **the 5B pair has only two arms — `control` and `cortex`** — and
no one-channel limb was ever trained.

**So: no, there is no E-only vs Z-only test at 5B.** There are E-only and
Z-only *cells*, produced by ablating a jointly-trained model at eval time, and
they are not a substitute. Getting the real number means training a limb, which
is an architecture-level decision — see
[[cortex-final-z-architecture-frozen]].

### 3e. The ‖E‖ WARN fired again, as predicted

Cortex arm: `1 warn`, the same `‖E‖ has moved −14.3% across the trace`.
Control: `0 warn`. This is the known false alarm resolved in §4a — it compares
the first diag record to the last over a trace that always includes launch.
**Nothing to investigate.** Note also that `watch_run.py` reads the *live* diag,
so `cortex-final_at_22800.jsonl` and `_at_23100.jsonl` are byte-identical
copies of the same 547-row file (both end at step 26,550); the `_at_<step>`
name records when it was copied, not a truncation at that step.

---

## 4. Mechanism trace at 23,100 / 26,550 — the one thing that did land

From `logs/final5b_diag/cortex-final_at_23100.jsonl` (547 rows, steps 50→26,550).
Architecture checks all pass: `n_depths 9`, `embed_frozen 1`,
`e_carried_norm 78.494`, `embed_ratio 1.0051`, `znorm_target 78.9`, zero
dead rows, no mix, stride 1, E read on, no e_dropout.

### 4a. The `‖E‖ −14.3%` WARN is a FALSE ALARM — resolved

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

### 4b. What actually drifted, 11,444 → 26,550

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

## 5. Confounds on any 11,444 → 22,800 comparison

Three things change together at step 11,450, so a Z-vs-E delta across that
boundary is **not** a clean "did Z grow" read:

1. **More training** — the intended variable.
2. **Corpus switch** — heal → mix, at `HEAL_END = 11444`. **This is the
   severe one, and it is worse than "the corpus changed":**
   `final_5b_common.sh:123-124` gives
   `HEAL_DATA=data/fineweb_edu_olmo_len4096` and
   `MIX_DATA=data/pg19_fw50_olmo_len4096`. The heal phase was FineWeb-Edu
   **only** — no books at all. So at the 11,444 baseline **PG-19 was entirely
   out-of-domain**; at 22,800 it is in-domain. Every PG-19 number therefore
   crosses an out-of-domain → in-domain boundary, which on its own explains
   the ~0.55 nat improvement on both arms, and is a live alternative
   explanation for Z contributing more there: the model now trains on
   long-form books, so there is in-distribution long-range structure for the
   carry to exploit. That is NOT the same claim as "the mechanism got better
   at remembering".
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

**What this leaves standing, stated precisely.** The ABSOLUTE numbers at
22,800 are sound: Z's contents are worth **+0.1295 nats on PG-19 at ~100%
share**, measured with E on throughout, via a within-configuration content
substitution (own Z vs donor Z, matched column count) — the cleanest
instrument in the suite, and the only one touching no ablated-E cell and no
cross-regime pairing. What is NOT sound is attributing the **increase** from
+0.0771 to the mechanism learning, because the eval pack went from
never-trained-on to trained-on over the same interval.

Likewise the §3b lead narrowing is sound as a *same-checkpoint* contrast —
both arms saw the identical corpus change, so it cancels — but its comparison
to the 11,444 lead inherits the same caveat.

---

## 6. Open questions carried forward

Updated after read-out 3. **ANSWERED** items stay on the list with their answer
— the point is the trajectory, not a clean slate.

- **The interaction sign.** *NOT ANSWERABLE in this run, and that is the
  answer.* The interaction needs both E-off cells (§3d), which the model never
  trained in. The raw numbers moved in opposite directions by pack — pg19
  +0.1094 → +0.0449 (CI now spans zero), carry +0.1175 → +0.2965 — which is
  itself a sign the term is not measuring one stable thing. **To settle
  complement-vs-substitute you need separately TRAINED limbs**, the way J5 did
  it (`evals/score_j5.py:70-73`: a `zonly` limb trained with E's rows zeroed, an
  `eonly` limb trained with Z's columns zeroed). The 5B pair has no such limb —
  only `control` and `cortex` — so no eval-time ablation can substitute for
  one. See [[e-off-cells-are-ood-no-e-dropout]].
- **Does X_content survive the re-target?** **ANSWERED: YES, and it grew.**
  +0.0771 → **+0.1295** on pg19 at `znorm_target=78.9` with the lo=1..9 band,
  share held at ~100% (107.0% → 103.4%). The content read is robust to the
  re-target and the re-map. See [[lo1-band-holds-content-at-11444]].
- **Why is loss flat?** Still open, and now with a candidate. Training loss
  moved −0.0255 over ~15k steps while gate norms moved 16–34% **and** Z's
  measured contribution on pg19 nearly doubled. **Candidate (2026-10-13):**
  cortex's mean grad norm has grown 0.795 → 1.341 against a fixed
  `--max_grad_norm 1.0`, so clip engagement went 4.4% → **32.9%**, and when it
  engages the coefficient averages 0.636. A third of updates having their
  gradient cut to ~64% is a real and growing distortion. The control is flat
  at ~0.4%. No non-finite anywhere in either arm, so this is not instability —
  but it is monotone. See
  [[cortex-grad-scale-is-growing-clip-rate-rising]].
- **Does the chunk1 deficit close?** **ANSWERED: NO, it widened.** fwedu
  +0.0394 → +0.0604, pg19 +0.0492 → +0.0547. The graft's fixed cost on
  no-carry context is growing, not amortising.
- **NEW — why did the between-arm lead narrow while Z got stronger?** The
  central tension of read-out 3. pg19 `all` lead halved (−0.0651 → −0.0328)
  while pg19 X_content rose 68%. Candidates: (a) the control captured more of
  the mix corpus's gain for reasons unrelated to memory, (b) Z's gain is
  real but is being paid for by the widening chunk1 deficit, (c) E_main fell
  on pg19 (−0.066) and Z's rise did not fully cover it. **(c) is checkable
  now:** E_main + X_off went 0.4178 → 0.4046 on pg19, i.e. roughly flat — so
  Z gained approximately what E lost, which favours (c) and is consistent
  with the weakening interaction. Worth one careful look before the next
  read-out.
- **NEW — carry has become an unreliable instrument.** X_off +0.1154\* with
  X_content −0.0006 (n.s.) is the pre-J7 signature: the columns are
  load-bearing as free capacity and the contents are irrelevant. Meanwhile
  `Z_alone` jumped to +0.4118 and `E_main` rose to +0.3041. Treat carry's Z
  numbers as a capacity measurement, not a content one, and quote pg19 for
  any claim about what Z holds.
