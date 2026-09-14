# Slurm job logs

One `Report-<jobid>.out` per PACE job, force-added past `.gitignore`'s `*.out` rule because the
sbatch **banner** is the only real evidence of what a run actually did (README.md, traps 1-2):
`require <path>` succeeding does not prove the run used that path, and a memory-off run looks
exactly like a real one on the loss curve.

```bash
grep -m1 '^=== ' logs/Report-<jobid>.out      # what this job actually ran
```

## Pruned 2026-09-10

58 logs were removed. The rule was **no surviving output**, applied two ways:

- **43 eval logs** whose banner named an `eval_results/<tag>/<run>` directory that no longer
  exists — the results they recorded were superseded and deleted (the `20260712`, `20260713`,
  `20260714`, `20260725` and `teacher_advantage` tags), plus 2 prolog-only stubs from jobs that
  died before producing output.
- **15 training logs** for arms with no surviving `eval_results/` entry anywhere: the `-v2` /
  `-rcl` / `-ri1e-3` rung1 probes, `rung2-k4-unfreeze500`, the `retro-b0-*` parity stage, the
  `retro-b1-base-mr8` control, and the un-suffixed `rung1-pfxaccum32-*-pg19` runs superseded by
  their `-ep1` / `-ep3` links. Every one of these keeps its loss curve in `wandb_exports/`.

`Report-11848359.out` (`rung1-pfxgated32-cc4-tb2-rs-mix`) was **kept** despite having no eval
results: it has no `wandb_exports/` CSV either, so the log is that arm's only record.

Nothing backing a number cited in README.md, the deck notes or `paper/` was touched.
