# Codebase cleanup audit — flags and dead paths

Written 2026-09-14. Scope: the `cortex.*` flag surface and the experiment-specific code hanging off
it. Companion to `../cortex_next_phase_framework.md` §9 (which covers files and folders, not code).

**Read §0 first.** There is uncommitted work at risk that matters more than any deletion here.

## STATUS — Tiers 1 and 2 EXECUTED 2026-09-14

Test suite before: **3 failed, 230 passed.** After: **3 failed, 228 passed** — the same three
pre-existing failures (`tests/test_cortex_eval.py::TestBabilong`, where `eb.eval_one` now returns
more than the 2 values the tests unpack — unrelated test drift, still open), and 228 = 230 minus the
two `TestL2SP` tests removed with the feature. **No regressions.**

Done:
* **Tier 1** — `l2sp_coeff` and `freeze_loop_until_step` removed from `train.py` (config keys,
  asserts, the `l2sp_pairs` snapshot, the `state` entry, the DDP re-wrap guard, the staged-unfreeze
  block, and the branch in `cortex_fwd_bwd`). `TestL2SP` removed. `pace/setup_login_node.sh`
  fixed — it advertised **three** sbatch files that no longer exist.
* **Tier 2** — `cortex_memory/legacy.py` created (569 lines: `LSTMBuffer`, `DirectCCoT`,
  `_extract_summary_vectors`, `AccumCCoT`, `GatedAccumBuffer`), `buffers.py` trimmed to the live
  prefix surface (748 -> 233 lines) with a re-export block so every existing import still resolves.
  All six buffer classes verified to construct; the graft still imports.
* `cortex_memory/__init__.py` — **it exported the four legacy classes and neither live one.** Fixed:
  `PrefixAccumBuffer` / `PrefixGatedBuffer` added to the imports and `__all__`, components section
  split into LIVE vs LOAD-PATH-ONLY, stale `_to_delete/` path corrected to `archive/planning/`.

Two corrections to what this audit originally claimed, both found by reading
`wandb_exports/cortex-retro-ft/summary.csv`:
1. `freeze_loop_until_step` **was** used — `rung2-k4-unfreeze500` is a real exported run. Removal is
   still correct (its sbatch was already gone, and the flag never reaches a checkpoint config), but
   `tools/pull_wandb_metrics.py`'s unfreeze-shock analysis reads **historical wandb configs**, not
   our dataclass, so it stays and now carries a comment saying why. *The producer can go while the
   reader of history stays.*
2. **§5's "`lora_rank: 0` in every run on record" was wrong** — `rung1b-k4-lora16-a32` ran at
   rank 16. The conclusion (keep LoRA) is unchanged and now better supported: the path has executed.
   Strike the "never exercised at 1B" caveat.

Still open from this audit: §0 (commit the working tree), §4 (the Tier-3 cluster grep), §6 (eval
debt).

---

## 0. URGENT — uncommitted work that should not be lost

* **`tools/analyze_longcontext_pairs.py` carries the mode-selection fix and is STILL UNCOMMITTED**
  (7 insertions in the working tree). This is the fix for the bug where `scoring_report` chose its
  modes from `scored_correct(on[shared[0]], m)` — one record — so every mixed babilong+lme run
  silently collapsed to the `gen` column and discarded rank/norm/PMI. **It is the reason the powered
  run produced `PMI-rank +1.08 pt, p=0.0075` and the `+3.37 / -2.08` dissociation at all.** Losing
  this working tree loses that analysis.
* **`tools/reconstruct_babilong_outputs.py` is untracked** (`??`).
* A large, sensible deletion of upstream McLeish cruft is already **staged but uncommitted**:
  `paper_plots/`, `shells/{eval,llama,tinyllama}.sh`, `mix_datasets.py`, `multi_recurence_eval.py`,
  `param_counter.py`, `plot_evals.py`, `convert_pretrained_model/raven_modeling_minimal_compare_*.py`,
  plus 86 `logs/Report-*.out`.

**Before deleting that staged set, check one file: `paper_plots/data/olmo_50k_steps.jsonl`.** If it
is McLeish's 50,000-step OLMo retrofit loss curve, it is the natural overlay for the control's own
38,147-step curve and is worth keeping out of the deletion. `shells/olmo.sh` is correctly retained —
it is the source of the batch-semantics finding in the framework doc.

---

## 1. The safety rule that decides everything else

`train.py:799-805` pushes **exactly 16** cortex flags onto the model config:

```
use_memory  memory_slots  memory_slots_iter  memory_heads  ccot_direct  h_T_proj
lora_rank  lora_alpha  accum_ccot  accum_vecs  accum_max  gated_accum
prefix_memory  summary_init_token  prefix_pos  prefix_eos_reset
```

Those 16 therefore appear in **every memory checkpoint's `config.json`**, and the graft is the load
path for all of them. Removing one reproduces the 2026-08-04 incident exactly: for two days
`CortexMemory.__init__` raised on the retired flags and **the entire historical results table became
unopenable** — the write-capacity diagnostic could not load the very checkpoint whose -0.01282 it
existed to explain.

The other **8 are training-time only** and no checkpoint carries them:

```
cross_chunks  freeze_loop  freeze_loop_until_step  eos_from_tokens
l2sp_coeff  memory_lr  carry_grad_chunks  random_segments
```

> **Rule: a flag in the persist list may be quarantined but never removed. A flag outside it can be
> deleted outright once shown unused.** This is mechanical and checkable, which is what makes it
> safe.

---

## 2. Tier 1 — safe deletions (training-only, verified unused)

### `l2sp_coeff` — DELETE

L2-SP anchor, "experiment-ladder rung 3."

* Training-only, so no checkpoint's `config.json` carries it and nothing becomes unloadable.
* **Its launcher is already gone.** `pace/rung3_l2sp.sbatch` does not exist, yet
  `pace/setup_login_node.sh:61` still prints `sbatch pace/rung3_l2sp.sbatch` — a dangling reference
  to a deleted file, which is a bug in its own right. Fix or delete that line either way.
* It *was* used once: `rung3-k4-l2sp1e-3` in `wandb_exports/cortex-retro-ft/summary.csv`, whose own
  summary column reads `final loss >> min loss`. The run is recorded in the ledger; the code is not
  what preserves it.

Removing it takes out: the config key, the `__post_init__` assert (268-271), the `l2sp_pairs`
construction and log (948-957), a `state` dict entry (1263), and — the reason this is worth
doing — **a branch inside the hot `cortex_fwd_bwd` loop** (1605-1609) that is evaluated on every
micro-step of every run and can never be true.

### `freeze_loop_until_step` — DELETE

Staged unfreeze: keep the loop frozen until step N, then unfreeze.

* Training-only. **Never set by any sbatch, never present in any wandb export.** Fully dead.
* Drags along: the config key, a DDP guard that raises (924-932), a log line (941), a per-step
  equality check in the training loop (1683-1684), and an entire before/after analysis mode in
  `tools/pull_wandb_metrics.py` (26, 60-62) that can never trigger.

### `random_segments` — KEEP, despite looking similar

Training-only and only referenced from `train.py`, so it scans as dead. It is not:

* it was used in a Track A arm (`rung1_corpus_mix.sbatch:58` documents `random_segments=true`), and
* it is **AutoCompressor's randomized segmenting**, which is the published precedent for the
  dynamic-`n_vec` / matryoshka idea the next model wants (AC randomizes the compression *ratio*;
  randomizing buffer *depth* goes beyond it).

It is prior art for live design work, not residue.

---

## 3. Tier 2 — quarantine, do not delete

The retired mechanisms and their buffers: `memory_slots`, `memory_slots_iter`, `memory_heads`,
`ccot_direct`, `h_T_proj`, `accum_ccot`, `gated_accum`, and the classes `LSTMBuffer`, `DirectCCoT`,
`AccumCCoT`, `GatedAccumBuffer` in `cortex_memory/buffers.py`.

All seven flags are in the persist list. Every Track-A and B1 `config.json` names one of them.
**Deletion orphans every number in the results table.** The current arrangement is already correct —
the graft *builds* them (load path), `train.py.__post_init__` *refuses to start* a run on them — it
is just not legible, which is why it keeps looking deletable.

Proposal, zero behaviour change:

1. Move the four legacy classes to **`cortex_memory/legacy.py`** with a module docstring stating:
   load-path only; required to open pre-2026-08-02 checkpoints; `train.py` refuses to start a new
   run on them; deleting them makes the historical results table unloadable (cite 2026-08-04).
   Re-export from `buffers.py` so no import breaks.
2. Leave `buffers.py` holding only `_PrefixBufferBase`, `PrefixAccumBuffer`, `PrefixGatedBuffer` —
   the live surface, and the file the next model edits.
3. Group the seven flags in the `cortex` dict under one `# --- legacy: load path only ---` comment
   instead of leaving them interleaved with live ones.

This is the highest-value item in the audit: it does not remove a line of behaviour, and it stops
the next person (or the next session) from re-proposing the deletion that already cost two days.

## 4. Tier 3 — compat branches — GATE RAN 2026-09-15, CLEAR, BRANCHES REMOVED

**Result: CLEAR across 148 surviving `config.json` files** (`cortex-retrofit/` — b0 both arms,
b1 both arms, the b2 arm ladder and the b2 heal ladder). Nothing on scratch carries either old
value. The branches were removed the same day; both keys still load and are now **asserted**:

| site | was | now |
|---|---|---|
| `cortex_graft.py` `prefix_pack` | `if n_sum and self.prefix_pos == "tail" and ...` | conjunct dropped, tail layout unconditional |
| `cortex_graft.py` `prefix_unpack` | `_valid_write` zeroing gated on `prefix_eos_reset` | removed; the prefix write is unmasked, and the comment says why |
| `cortex_graft.py` `_carried_state` | `_write_reset` zeroing gated on `prefix_eos_reset` | `return self._cross_buf` |
| `cortex_graft.py` build | `prefix_pos not in ("tail","zero")` raises | `prefix_pos != "tail"` raises; `prefix_eos_reset` true raises |

`_valid_write` and `_write_reset` are **not** dead — `begin()` still computes them and the bolt-on
buffer path (graft ~575-596) still consumes them. Only the prefix-mode uses went.

Tests rewritten from exercise-the-old-branch to assert-the-modern-value, plus a new
`test_the_key_still_loads` pinning the persist-list rule directly. `tests/test_smoke_prefix_real.py`
kept its negative control by zeroing `_carried_state` through a monkeypatch instead of flipping the
retired flag — it tests the *checker*, which was always the point. 315 -> 316 passing.

**Two roots were NOT in the 148** and the script now says so out loud rather than swallowing them:
`$SCRATCH/ckpts` (does not exist at that path — `--model_name ckpts/olmo-retrofit-cortex` is
relative to the submit dir) and `$SCRATCH/cortex-retro-ft` (the rung1 arms; apparently pruned).
Neither weakens the verdict — the flags were introduced 2026-08-04 with defaults `tail`/`false`, so
anything older simply lacks the keys and takes the default, and the one run that ever set them was
`rm -rf`'d — but confirm the base-checkpoint dir if it ever matters:
`find -L "$(readlink -f ckpts)" -maxdepth 2 -name config.json -exec grep -l '"prefix_pos": *"zero"\|"prefix_eos_reset": *true' {} +`

---

### The original gate, kept for the reasoning

`prefix_pos="zero"` and `prefix_eos_reset=true` exist **only** to reproduce the cancelled
4vop2ym8 run, which was `rm -rf`'d. Both defaults are the correct 2026-08-04 values, and b2-final
carries `"prefix_pos": "tail", "prefix_eos_reset": false`.

A subtlety worth stating because it is the whole point of the persist-list rule: **you can delete a
code path without deleting the config key.** The key must keep loading; the branch need not keep
working. So the option is to retain both keys, assert the modern value, and remove the old branches
from `prefix_pack` / `_carried_state`.

**Gate it cluster-side, across every surviving checkpoint dir:**

```bash
bash pace/check_tier3_compat.sh
```

Empty result: remove the branches, keep the keys. Any hit: leave both alone and record which
checkpoint depends on them.

**The one-liner this section used to carry was WRONG, and wrong in the direction that deletes a
live code path** (found 2026-09-15, before it was acted on):

```bash
grep -l '"prefix_pos": *"zero"\|"prefix_eos_reset": *true' $SCRATCH/cortex-*/*/config.json   # BROKEN
```

That glob is **one level too shallow**. `train.py` writes `config.json` only through
`save_model_only` -> `save_pretrained(f"{out_path}/{run_name}/{chkpt_name}")` (train.py:531-534),
so every one lives at `$SCRATCH/<out_path>/<run_name>/<chkpt_name>/config.json` — **three** levels
under `$SCRATCH`, not two. `save_checkpoint` (train.py:536) writes only `chkpt.pt` into
`checkpoint_<step>/` and no `config.json` at all. So the old glob expands to
`$SCRATCH/cortex-control/c-chunked/config.json` and its siblings, none of which exist, and it
returns empty on **every** cluster regardless of what is stored there. Verified against a tree
holding a genuine `prefix_pos="zero"` checkpoint: the old glob reported it clear.

This is bug class 2 again - *find where a value is CONSUMED, not where it is set* - now on the
verification side rather than the training side. The script counts its scan set first and exits 2
rather than reporting a verdict when nothing was scanned.

Scope note: `config.json` is the branches' load path (`cortex_graft.py:266-270` reads both flags
off the HF config), so `config.json` coverage is the right coverage - but it has to include the
base checkpoints under `$SCRATCH/ckpts`, since `--model_name` is what a resume link constructs its
model from before `chkpt.pt` restores weights into it. The script covers both.

## 5. Tier 4 — LoRA: keep, but it is untested

`lora_rank` / `lora_alpha` are persisted and inert (`lora_rank: 0` in every run on record, including
the rung1 frozen-loop arms). It scans as dead weight.

Keep it. AutoCompressor's Llama recipe spends its **entire** LoRA budget on the four attention
projections that read the soft prompt, and this project's largest measured mechanism effect is that
unfreezing the read path turned x0.90 into x1.33. LoRA on the read path is a live future option.

But treat it as **unverified**: if `lora_rank` has been 0 in every run, the LoRA code path has never
executed at 1B scale. If it is ever switched on, gate it with a parameter-count and gradient-flow
check first — this codebase's failure mode is silently-inert memory paths behind healthy loss curves.

## 6. Tier 5 — eval flags: review, do not bulk-delete

`evals/eval_babilong.py` has 23 CLI flags, `eval_longmemeval.py` 21. Six babilong flags are set by
no `pace/` script: `--checkpoint`, `--length_buckets`, `--max_new_tokens`, `--memory_slots`,
`--no_seed_per_example`, `--records`.

"Never set by a pace script" is a **much weaker signal** than the persist-list rule — several are
either load-bearing defaults or deliberate escape hatches (`--memory_slots` is needed to open legacy
checkpoints; `--records` produces the `records.jsonl` the entire paired analysis reads). Review
individually or leave alone.

**The real debt on the eval side is not flags.** In rough priority:

1. `greedy_generate` (`evals/model_utils.py:366`) is a hand-rolled greedy loop with **no
   `min_new_tokens`**, so EOS at position 0 yields an empty string. ~4 lines. Known to recover only
   ~14% of the early-EOS deficit and to break comparability with every existing LongMemEval table —
   so it is a deliberate decision, not an oversight, and should be *recorded as one* rather than
   left looking unfinished.
2. LongMemEval still has no answer-likelihood path, so it cannot be scored the way BABILong is —
   which is why its headline number is a generation-length artifact (r=0.949).
3. `eval_context_ceiling.py:266` silently clamps context to whatever precedes the chunk
   (`ctx = x[max(0, off - k):off]`), so any pooled `ceiling_k*` for k larger than the chunk offset
   mixes chunks that never received the requested context. This has already produced one
   unnoticed bad number (`ceiling_k1024: -0.456`, unread from 2026-08-07). Make it warn.

---

## 7. Suggested order

1. **Commit the working tree** (§0) — the analyzer fix especially. Nothing else should happen first.
2. Check `paper_plots/data/olmo_50k_steps.jsonl`, then commit the staged upstream deletions.
3. Tier 1: delete `l2sp_coeff` and `freeze_loop_until_step`; fix `setup_login_node.sh:61`.
4. Tier 2: the `legacy.py` quarantine.
5. Tier 3: run the one grep cluster-side, then decide.
6. Tiers 4-5: leave for when they become load-bearing.

Steps 3 and 4 both touch `train.py`'s config block, which the control run is about to depend on —
so do them **before** the launch, not during, and re-run the test suite
(`/c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q`) after each.
