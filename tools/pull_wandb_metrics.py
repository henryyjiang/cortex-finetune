"""
Pull training curves for the cortex-retro-ft runs from wandb and flag anomalies.

Runs LOCALLY (or anywhere with a wandb login — credentials come from
`wandb login` / ~/.netrc).  On Henry's machine use the `cortex` env (has
wandb + pandas); the cluster's cortex-retro env also has wandb.

    python tools/pull_wandb_metrics.py                       # latest attempt per run name
    python tools/pull_wandb_metrics.py --all_attempts        # every attempt (crashed probes too)
    python tools/pull_wandb_metrics.py --runs rung1-k4 rung2-k4-unfreeze500

Outputs
-------
  wandb_exports/<project>/<run_name>__<run_id>.csv   full per-step history (train/*)
  wandb_exports/<project>/summary.csv                one row per run
  stdout: per-run anomaly report + cross-run summary table

What "anomaly" means here
-------------------------
  * NaN loss rows / non-finite grad-norm rows (non-finite total_norm = the
    train.py guard SKIPPED that update; a long run of them = the run wasn't
    actually training)
  * largest step-over-step loss jump (spikes)
  * grad-clip engagement rate (grad_clip_coef < 1)
  * unfreeze shock for staged-unfreeze runs: mean loss / grad-norm in the 50
    steps before vs after cortex.freeze_loop_until_step
  * end-of-run trend: mean loss over the last 100 steps vs the 100 before
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

KEYS = [
    "train/step", "train/loss", "train/log_ppl", "train/lr",
    "train/lr_recur", "train/lr_nonrecur", "train/total_norm",
    "train/grad_clip_coef", "train/mean_recurrence", "train/mean_backprop_depth",
    # total_tokens counts every position INCLUDING padding; total_tokens_with_loss
    # is the unmasked-label count.  Pulling only the first made the padding
    # fraction invisible in every local export — it matters on any doc-per-row
    # pack (PG-19) and is ~1.0 on a wrapped one.
    "train/total_tokens", "train/total_tokens_with_loss", "train/epoch",
    # per-group LRs (2026-09-14): lr_recur/lr_nonrecur both read group 0 when
    # throttle is off, so neither ever showed the AdamW group's LR or memory_lr.
    "train/lr_group0", "train/lr_group1", "train/lr_group2", "train/lr_group3",
]


def fetch_history(run) -> pd.DataFrame:
    rows = list(run.scan_history())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = [k for k in KEYS if k in df.columns]
    df = df[cols].apply(pd.to_numeric, errors="coerce")
    if "train/step" in df.columns:
        df = df.sort_values("train/step").reset_index(drop=True)
    return df


def freeze_until(config: dict):
    # Reads HISTORICAL wandb run configs.  cortex.freeze_loop_until_step was
    # removed from train.py on 2026-09-14, so no new run can set it — but
    # rung2-k4-unfreeze500 is still in wandb_exports/cortex-retro-ft/ and its
    # stored config still carries the key, so this analysis stays usable for
    # reading that history.  Do not delete it along with the producing flag.
    cortex = config.get("cortex") or {}
    if isinstance(cortex, dict) and cortex.get("freeze_loop_until_step"):
        return int(cortex["freeze_loop_until_step"])
    v = config.get("cortex.freeze_loop_until_step")
    return int(v) if v else None


def analyze(name: str, run, df: pd.DataFrame) -> dict:
    info = {"run": name, "id": run.id, "state": run.state}
    if df.empty or "train/loss" not in df.columns:
        info["note"] = "NO HISTORY"
        return info

    step = df.get("train/step")
    loss = df["train/loss"]
    norm = df.get("train/total_norm")
    clip = df.get("train/grad_clip_coef")

    info["steps"] = int(step.max()) if step is not None else len(df)
    info["tokens"] = int(df["train/total_tokens"].max()) if "train/total_tokens" in df else None
    finite_loss = loss.dropna()
    info["loss_start"] = round(float(finite_loss.iloc[:10].mean()), 4) if len(finite_loss) else None
    info["loss_final"] = round(float(finite_loss.iloc[-20:].mean()), 4) if len(finite_loss) else None
    info["loss_min"] = round(float(finite_loss.min()), 4) if len(finite_loss) else None

    # non-finite bookkeeping: coerce turned nan/inf-serialized values into NaN;
    # keep inf detection for numerically-delivered infs too.
    info["nan_loss_rows"] = int(loss.isna().sum() + np.isinf(loss.fillna(0)).sum())
    if norm is not None:
        info["skipped_updates"] = int(norm.isna().sum() + np.isinf(norm.fillna(0)).sum())
        fn = norm.replace([np.inf, -np.inf], np.nan).dropna()
        if len(fn):
            info["gnorm_p50"] = round(float(fn.quantile(0.5)), 3)
            info["gnorm_p99"] = round(float(fn.quantile(0.99)), 3)
            info["gnorm_max"] = round(float(fn.max()), 3)
    if clip is not None and clip.notna().any():
        info["clip_active_frac"] = round(float((clip.dropna() < 0.999).mean()), 3)

    # biggest step-over-step loss spike
    if len(finite_loss) > 1:
        jumps = finite_loss.diff()
        j = jumps.idxmax()
        if not math.isnan(jumps.max()):
            at = int(step.loc[j]) if step is not None else int(j)
            info["max_loss_jump"] = f"+{jumps.max():.3f}@{at}"

    # unfreeze shock (staged-unfreeze runs)
    fu = freeze_until(run.config)
    if fu and step is not None:
        pre = df[(step >= fu - 50) & (step < fu)]
        post = df[(step >= fu) & (step < fu + 50)]
        if len(pre) and len(post):
            info["unfreeze"] = fu
            info["loss_pre/post_unfreeze"] = (
                f"{pre['train/loss'].mean():.3f}/{post['train/loss'].mean():.3f}")
            if norm is not None:
                info["gnorm_pre/post_unfreeze"] = (
                    f"{pre['train/total_norm'].mean():.3f}/{post['train/total_norm'].mean():.3f}")

    # end-of-run trend: still improving?
    if len(finite_loss) >= 200:
        last, prev = finite_loss.iloc[-100:].mean(), finite_loss.iloc[-200:-100].mean()
        info["end_trend"] = f"{prev:.4f}->{last:.4f}" + ("  [NOT IMPROVING]" if last > prev else "")

    flags = []
    if info.get("nan_loss_rows"):
        flags.append(f"{info['nan_loss_rows']} NaN-loss rows")
    if info.get("skipped_updates"):
        flags.append(f"{info['skipped_updates']} guard-skipped updates")
    if info.get("loss_final") is not None and info.get("loss_min") is not None \
            and info["loss_final"] > info["loss_min"] + 0.1:
        flags.append("final loss >> min loss")
    info["FLAGS"] = "; ".join(flags) if flags else "-"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="cortex-retro-ft")
    ap.add_argument("--entity", default=None, help="default: your wandb default entity")
    ap.add_argument("--runs", nargs="+", default=None, help="run names to include (default: all)")
    ap.add_argument("--all_attempts", action="store_true",
                    help="export every attempt; default keeps only the latest per run name")
    ap.add_argument("--out", default="wandb_exports")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    entity = args.entity or api.default_entity
    path = f"{entity}/{args.project}"
    runs = sorted(api.runs(path), key=lambda r: r.created_at, reverse=True)
    if args.runs:
        runs = [r for r in runs if r.name in set(args.runs)]
    if not args.all_attempts:
        seen, latest = set(), []
        for r in runs:  # newest first
            if r.name not in seen:
                seen.add(r.name)
                latest.append(r)
        runs = latest
    if not runs:
        print(f"No runs found in {path}")
        return 1
    print(f"{path}: {len(runs)} run(s)\n")

    out_dir = os.path.join(args.out, args.project)
    os.makedirs(out_dir, exist_ok=True)

    reports = []
    summary_csv = os.path.join(out_dir, "summary.csv")
    failed = []
    for r in sorted(runs, key=lambda r: r.name):
        # One flaky run must not cost the whole table.  scan_history() streams,
        # and a dropped wandb service connection raises CommError mid-run; before
        # 2026-09-14 that propagated out of main() and summary.csv was never
        # written at all, losing every run already fetched.
        try:
            df = fetch_history(r)
        except Exception as exc:
            print(f"-- {r.name}  [{r.state}] SKIPPED: {type(exc).__name__}: {exc}")
            failed.append(f"{r.name}__{r.id}")
            continue
        csv = os.path.join(out_dir, f"{r.name}__{r.id}.csv")
        df.to_csv(csv, index=False)
        rep = analyze(r.name, r, df)
        reports.append(rep)
        print(f"-- {r.name}  [{r.state}, {len(df)} logged steps] -> {csv}")
        for k, v in rep.items():
            if k not in ("run", "id", "state"):
                print(f"     {k:26} {v}")
        print()
        # Rewrite the summary after every run, so an interrupted pull still
        # leaves a usable (partial) table rather than a stale one.
        pd.DataFrame(reports).to_csv(summary_csv, index=False)

    if not reports:
        print("No run history could be fetched.")
        return 1
    if failed:
        print(f"WARNING: {len(failed)} run(s) could not be fetched and are NOT "
              f"in summary.csv: {', '.join(failed)}\n  Re-run to fill them in.")

    summary = pd.DataFrame(reports)
    summary.to_csv(summary_csv, index=False)
    core = [c for c in ("run", "state", "steps", "loss_start", "loss_final", "loss_min",
                        "skipped_updates", "clip_active_frac", "FLAGS") if c in summary.columns]
    print("=== summary ===")
    print(summary[core].to_string(index=False))
    print(f"\nSaved -> {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
