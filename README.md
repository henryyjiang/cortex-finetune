# CortexGPT — episodic memory for retrofitted-recurrence LMs

A fork of [mcleish7/retrofitting-recurrence](https://github.com/mcleish7/retrofitting-recurrence)
("Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence") that grafts an
**episodic latent memory** onto the Pre/Loop/Coda (raven) model, plus the PACE cluster harness and
the diagnostic eval suite used to measure whether that memory actually carries anything.

**The thesis.** A depth-recurrent LM re-reads its own latent state many times per token, but it
still forgets everything at a context-window boundary. Cortex adds a small carried state across
those boundaries and asks a single measurable question: *is the carried state worth anything, in
nats/token, against the same model with the state zeroed?*

Upstream's own README (dataset download/tokenization/packing, the model-conversion procedure,
lm_eval usage) is preserved verbatim in [README_upstream.md](README_upstream.md) and is still the
procedure for everything this fork does not change.

## The base model

OLMo-2-1B, rearranged by the retrofit conversion into **prelude (4 layers) / recurrent core block
(6 layers, applied T times) / coda (4 layers)** — effective depth `4 + 6T + 4` (= 56 at T=8). Same
pretrained weights, untrained arrangement: an honest retrofit starts around loss 10.3 and the
conversion trains it back down.

Two base checkpoints exist and **confusing them invalidates a study**:

| checkpoint | what it is | using it means |
|---|---|---|
| `ckpts/olmo-retrofit-cortex` | `smcleish/Recurrent-OLMo-2-0425-untrained` — pretrained weights, untrained arrangement | a **retrofit** (memory present from step 0) |
| `ckpts/olmo8-cortex` | `train-recurrence-8` — already converted (~50B tokens) | a **finetune** (bolt-on memory, converged loop) |

## What this fork adds

| path | what it is |
|---|---|
| `cortex_memory/` | base-model-agnostic memory components (pure torch, no host-model imports) |
| `cortex_graft.py` | wires `cortex_memory` into `RavenForCausalLM`; flag-gated, **default off** |
| `train.py` | upstream's finetuning script + the `cortex.*` config block, chunk chains, branch/resume chaining, Muon/ELLISAdam |
| `pace/` | Slurm drivers and sbatch files for PACE Phoenix (training + every eval) |
| `evals/` | the diagnostic instruments (carry ablation, context ceiling, buffer diagnostic) and the benchmark runners |
| `tools/` | dataset packing/mixing, checkpoint prep, pack verification, wandb export |
| `tests/` | unit + smoke tests; the real raven modeling files import and run off-cluster |
| `eval_results/` | every measurement, one directory per `<instrument>_<EVAL_TAG>` |

With `use_memory=False` (the default) the host model is byte-for-byte unchanged — every hook is a
guarded no-op, and all settings are read from the config via `getattr` with safe defaults, so an
unmodified checkpoint `config.json` still loads.

### Memory mechanisms

All live in `cortex_memory/buffers.py`:

- **`PrefixAccumBuffer`** (`--cortex.prefix_memory accum`) — AutoCompressor-faithful: `accum_vecs`
  summary vectors are extracted per chunk and **prepended as a prefix** to the next chunk, seeded
  from the EOS embedding, unbounded (no tanh/LN), FIFO-capped at `accum_max`. This is the current
  arm.
- **`PrefixGatedBuffer`** (`--cortex.prefix_memory gated`) — same extraction write, LM2 gated merge
  onto fixed slots. Append-vs-overwrite is the whole comparison; it forks the same healing
  checkpoint, so it stays controlled however long you wait.
- **`AccumCCoT` / `DirectCCoT` / `LSTMBuffer` / `GatedAccumBuffer`** — the earlier bolt-on
  mechanisms (Track A). Retained for reproducing published numbers; the legacy flags now **raise**
  rather than being silently ignored.

Geometry flags that must move together: `cross_chunks` (sub-windows per sequence), `accum_vecs`
(vectors written per chunk), `accum_max` (FIFO cap), `carry_grad_chunks` (stop-grad horizon).
`train.py` asserts `cross_chunks * accum_vecs <= accum_max` — deliberately a hard failure, because
silent FIFO trimming would break the stop-grad slice bookkeeping. Never satisfy that assert by
lowering `accum_vecs`: `summary_emb` is an `nn.Parameter(n_vec, hidden)` and changing it is not
resume-compatible. `accum_max` is a plain int and is safe to raise.

## Training on PACE

Everything runs through one driver per study; nothing past the current stage is ever committed, and
every stage is one command. The current study is **B2**, a two-phase retrofit:

```bash
bash pace/submit_retrofit_b2.sh heal          # shared healing phase, 0 -> 91,552 steps (1.5B tok)
bash pace/submit_retrofit_b2.sh status        # queue + resumable/evaluable step per run
bash pace/submit_retrofit_b2.sh branch accum  # branch an arm off the healing checkpoint
bash pace/submit_retrofit_b2.sh resume accum  # continue an arm (one 48h H200 link at a time)
bash pace/submit_retrofit_b2.sh gate          # carry ablation + slices + buffer diagnostic
```

Why a *shared* healing phase is legitimate: accum's parameters are a strict subset of gated's, so
healing as accum trains everything both arms need except the gate. The branch loads weights
non-strictly, drops the optimizer (its parameter set changed) and the dataloader position (the
corpus changed), and **continues** the LR schedule, the recurrence ramp and the step counter. Both
arms branch identically, so nothing about the branch can favour one of them.

`MAX_STEPS = 305176` (5B / 16,384 tok-step) is the horizon for **both** the LR cosine and the
mean-recurrence ramp, and it is fixed for every link of every chain. Varying it across links re-runs
cooldown per segment and bakes a sawtooth into the weights. To end a phase early use
`STOP_AT_STEP`, which does not move the horizon. Recurrence warmup is 0.25, so full recurrence 8
lands at step 76,294 — inside the healing phase, which is what makes every arm checkpoint
comparable to the fixed-recurrence Track A arms.

`pace/b1_retrofit.sbatch` + `submit_retrofit_b1.sh` are the earlier single-phase retrofit;
`pace/rung1_*.sbatch` are the short frozen-loop side-runs.

Hardware: H200 (141GB) is the standard for anything at full recurrence — the 4x1024-chunk backward
graph OOMs an A100-80GB even with no memory carried, and there is no gradient-checkpointing flag
here. A100 requests need `--gres=gpu:A100:1` plus `--constraint=A100-80GB` (the memory size is a
node feature, not a gres variant); H200 is a gres type and needs no constraint.

## Evals

The instrument is the **carry ablation**: per-chunk PG-19 validation NLL, paired carried-vs-zeroed
on the same sample with the recurrent init seeded identically across conditions. Chunk 1 is
identical by construction and acts as the control; the delta on chunks 2+ is what the memory is
worth, in nats/token. It resolves ~0.001 nats at n=150-200.

```bash
bash pace/submit_evals_all.sh                 # basic + long-context for every final_checkpoint
```

```bash
bash pace/submit_ceiling_probes.sh            # oracle-ceiling probes (forward-only, cheap)
```

Individual sbatch files take their configuration from the environment (`OUT_ROOT`, `RUN`,
`CKPT_NAME`, `BASE`, `EVAL_TAG`, `N_CHUNKS`, `T_EVAL`, ...). **`EVAL_TAG` is the results
namespace** — two reads of the same checkpoint at different geometries need different tags, or the
second silently overwrites the first.

Read a carry delta as a **fraction of the oracle ceiling**, never against an absolute bar. The old
~0.1-nat bar was retracted: real full attention over the previous chunk is worth at most ~0.10 nats
on this instrument, so no mechanism could ever have cleared it. Run `eval_context_ceiling` at the
*same* `N_CHUNKS` as the arm to get that arm's own denominator, and compare only at matched
geometry — chunk length is the single biggest lever in the project.

Reference points, all with raw JSON under `eval_results/`:

| arm | carry delta (nats, chunks 2+) |
|---|---|
| Track A `AccumCCoT` 4-vec (bolt-on, frozen loop) | +0.0237 +/- 0.0012 |
| Track A, same checkpoint read at 8 chunks instead of 4 | +0.0433 +/- 0.0012 |
| Track A `GatedAccumBuffer` k=32 | +0.0283 +/- 0.0010 (best delta, worst benchmarks) |
| B1 co-trained retrofit, 22.5k gate | -0.0128 +/- 0.0014 (carrying was worse than blanking) |
| oracle ceiling at `cross_chunks=8` | +0.1485 |

Two things that ablation established and that still bind: **write diversity is not the bottleneck**
(B1 wrote near-orthogonal vectors and produced the worst delta ever measured), and **carry delta
and downstream behaviour are anti-correlated across arms** — do not read one off the other, and do
not read either off the training loss, which separated none of these arms.

## Tests

The real raven modeling files import and run off-cluster, so the smoke tests are not mocks:

```bash
python -m pytest tests/ -q
```

```bash
python tools/smoke_prefix_real.py --cross_chunks 8 --accum_max 256
```

`transformers==4.51.0` is pinned upstream because of a KV-cache breaking change. `train.py` cannot
be imported off-cluster (it runs a device health check at import), so `tests/test_cortex_train.py`
and `tests/test_timeleft.py` **mirror** its logic — keep the mirrors in sync when you edit it.

## Traps that have each cost real GPU-hours

1. **A silent memory-off run looks exactly like a real one.** It trains a perfectly healthy loss
   curve. The log line `[cortex] memory ON: ... prefix_memory=accum` must appear, and `train.py`
   has a step-0 guard that fails the job when a model dir's modeling snapshot predates the prefix
   rewrite. Three arms were trained to completion without memory before that guard existed.
2. **A driver script's variables do not reach its sbatch.** A plain assignment is a shell variable,
   not an environment variable, so the sbatch's own default wins silently. Pass values as a
   var-assignment prefix on the `sbatch` command itself. **The banner is the only real check:**
   `grep -m1 '^=== B2' logs/Report-<job>.out`. `require <path>` succeeding is not evidence that the
   run uses that path — the check and the consumer are different variables in different processes.
3. **`out_path` must be a symlink to `$SCRATCH`.** `wandb.init` ignores `WANDB_DIR` here and a
   relative `out_path` resolves into `$HOME`; a full home quota starves *unrelated* running jobs
   into dying with no traceback. `HF_HOME` belongs in `~/.bashrc`, not just in the job script.
4. **The parquet loader does not wrap.** Exhaustion is a clean exit, so wandb reports "finished" at
   half the intended run. Always size packs past the budget and verify with `tools/check_pack.py`.
5. **Checkpoint semantics.** `save_interval` writes a weights-only `model_only_chkpt_<step>`
   (evaluable, not resumable); the full `checkpoint_<step>` (optimizer + scheduler + dataloader)
   fires every *2x* `save_interval`, so worst-case crash cost is 2x, not 1x. Nothing rotates —
   prune as a chain advances. Discover checkpoints numerically, never lexically.
6. **Changing corpus on a resume needs `EXTRA_ARGS="--ignore_past_parquet_dataset true"`, on the
   switch link only.** A same-corpus link that gets it replays its pack from row 0.
7. **`save_to_disk` without `num_proc` costs ~14h on Lustre** for a shuffled pack (a million random
   seeks over a 20GB arrow file). Grep any new pack/mix script for it before launching.
8. **Fresh modules grafted onto a `from_pretrained` checkpoint are not initialized by `__init__`.**
   `post_init` re-initializes them as "missing keys", and the transformers meta-device path leaves
   them as uninitialized VRAM — finite garbage, so a non-finite sweep will not catch it.
   `train.py:reset_cortex_graft_init` handles this after load and is skipped on resume.

## Status

- The from-scratch 155M line lives in the sibling repo
  [cortex](https://github.com/henryyjiang/cortex) and is closed.
- Track A (bolt-on, frozen loop) and B1 (single-phase co-trained retrofit) are closed as
  characterized negatives; the numbers above are what survives from them.
- **B2 is live**: the shared healing phase to 91,552, then the `accum` arm to 305,176 — math-heavy
  corpus first, then a 50/50 FineWeb-Edu + Nemotron mix for the anneal (report it as an ordered
  schedule, not as "trained on the mix"). The `gated` arm can still be branched from the same
  healing checkpoint at any time; if it is, it must replicate the accum arm's corpus switch point
  exactly or the two arms are not controlled.

## Upstream and licence

This work builds directly on retrofitting-recurrence; the conversion scripts, the raven modeling
files, the packing pipeline and the training loop are theirs. See [LICENCE](LICENCE) and cite:

```
@article{mcleish2025teaching,
    title={Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence},
    author={Sean McLeish and Ang Li and John Kirchenbauer and Dayal Singh Kalra and Brian R. Bartoldson and Bhavya Kailkhura and Avi Schwarzschild and Jonas Geiping and Tom Goldstein and Micah Goldblum},
    journal={arXiv preprint arXiv:2511.07384},
    year={2025}
}
```

Mechanism references: AutoCompressors for the prefix summary-vector carry, LM2 for the gated slot
buffer, Parcae for LTI injection and the recurrence sampler, Coconut for the single-vector latent
carry.
