"""
Narrow a trained `summary_emb` from [W_old, D] to [W_new, D] so a P1 arm can
branch at a write width the parent never trained at.

WHY THIS EXISTS
---------------
`retro-b2-heal/checkpoint_91552` is a cc4 / accum_max 128 run at W=32.  A1 and
A3' both run W=16 (P1.0: cheaper writes buy horizon), and W is `summary_emb`'s
SHAPE, so it cannot change on a load:

    unwrap.load_state_dict(ckpt["model"], strict=False)

`strict=False` forgives missing and unexpected keys.  It does NOT forgive a size
mismatch — that is collected into error_msgs and raised regardless — so the
branch would die at startup.  The tempting fix is to drop the key and let the
arm start from a fresh `summary_emb`, and that is the failure this whole file is
written to prevent: the graft would re-seed all 16 rows from `wte[eos]`, the arm
would silently throw away 91,552 steps of write-path training, and the loss curve
would look completely healthy while doing it.  Recurring bug class 2, one more
time.  So: slice explicitly, and LOG WHAT WAS SLICED.

WHICH ROWS, AND WHY THE FIRST ONES
----------------------------------
The FIRST W_new rows, and this is not arbitrary.  Under `prefix_pos="tail"` the
summary slots sit at the end of the packed chunk and see each other CAUSALLY —
slot j attends slots < j, which is the only thing that stops W identically
seeded slots from collapsing into copies of one vector (see prefix_pack).  So
row j's trained function is conditioned on rows 0..j-1 and on nothing after it.
Keeping a prefix therefore keeps every row's own context intact; keeping a
suffix, or a strided subset, would hand each survivor a context that no longer
exists.  Same argument as AutoCompressor's ordering, and it is the reason a
"take every other row" variant is not offered here.

WHAT IS CHECKED BEFORE ANYTHING IS WRITTEN
------------------------------------------
  * `summary_seeded` is True.  False means the parent never ran a forward and
    `summary_emb` is post_init noise — slicing noise is not an error the loss
    curve will ever show you.
  * the rows are not all identical.  Identical rows are the `wte[eos]` seed
    unchanged, i.e. a write path that did not train.  Then "the first 16" is
    vacuous and the real problem is upstream.
  * exactly ONE tensor under `cortex.prefix.` carries the old width.  If a
    second one appears (a future per-write parameter), this tool would leave it
    at the wrong shape and the branch would fail confusingly, or worse, not.

WHAT IT REPORTS, because "and log it" is half the task: per-row norms for kept
and dropped rows, the mean centred cosine within each group and across them, and
the effective rank of both — so "the 16 we kept carry what the 32 carried" is a
measurement in the run log and not an assumption.  Effective rank is the one to
read: P0.2 measured W=32 at ~1.87 participation ratio, so a drop from 32 to 16
that costs little rank is the P1.0 bet being confirmed on the TRAINED weights
rather than at init.

USAGE
-----
    python tools/slice_summary_emb.py \
        --src cortex-retrofit/retro-b2-heal/checkpoint_91552 \
        --dst cortex-retrofit/retro-b2-heal/checkpoint_91552_w16 \
        --n_vec 16

Then branch the arms off --dst.  The written checkpoint carries a
`cortex_slice` provenance record; train.py REFUSES to --resume_path it (the
optimizer state still has the old shapes and restoring it would be wrong), and
accepts it only as a --branch_path, which is the only intended use.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime

import torch

KEY = "cortex.prefix.summary_emb"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("slice a trained summary_emb to a narrower W")
    p.add_argument("--src", required=True,
                   help="checkpoint dir holding chkpt.pt (the PARENT run)")
    p.add_argument("--dst", required=True,
                   help="new checkpoint dir to write (must not be --src)")
    p.add_argument("--n_vec", type=int, required=True,
                   help="W_new, the arm's accum_vecs")
    p.add_argument("--key", default=KEY,
                   help="state-dict key to slice (override only if the graft "
                        "moves; the default is where PrefixAccumBuffer and "
                        "PrefixGatedBuffer both keep it)")
    p.add_argument("--force", action="store_true",
                   help="write anyway when a diagnostic says the parent's write "
                        "path looks untrained.  There is no good reason to pass "
                        "this on a real branch; it exists for tests.")
    return p.parse_args()


def _row_stats(w: torch.Tensor) -> dict:
    """Norms, centred cosine and participation-ratio rank for a [n, D] block.

    Centred, because the write is ~98% a constant direction (measured: cos 0.981
    between writes from DIFFERENT documents).  An uncentred cosine on these rows
    reports that constant and says nothing about whether the rows differ.
    """
    w = w.detach().float()
    n = w.shape[0]
    norms = w.norm(dim=-1)
    c = w - w.mean(dim=0, keepdim=True)
    cn = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    g = cn @ cn.T
    off = g[~torch.eye(n, dtype=torch.bool)] if n > 1 else torch.zeros(1)
    sv = torch.linalg.svdvals(c)
    pr = float((sv.pow(2).sum() ** 2) / sv.pow(4).sum().clamp_min(1e-30)) if n > 1 else 1.0
    return {
        "rows": n,
        "norm_mean": float(norms.mean()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "centred_cos_mean": float(off.mean()),
        "participation_rank": pr,
    }


def _cross_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean centred cosine between two row blocks, centred on their UNION.

    Centring each block on its own mean would remove exactly the between-group
    difference this number is supposed to measure.
    """
    a, b = a.detach().float(), b.detach().float()
    mu = torch.cat([a, b], dim=0).mean(dim=0, keepdim=True)
    an = (a - mu); bn = (b - mu)
    an = an / an.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    bn = bn / bn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return float((an @ bn.T).mean())


def main() -> int:
    args = parse_args()
    src_pt = os.path.join(args.src, "chkpt.pt")
    if not os.path.isfile(src_pt):
        print(f"FAILED: no chkpt.pt under --src {args.src}")
        return 2
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        print("FAILED: --dst must differ from --src.  Slicing in place would "
              "destroy the parent checkpoint every other arm branches from.")
        return 2

    print(f"loading {src_pt} ...", flush=True)
    ckpt = torch.load(src_pt, map_location="cpu", weights_only=False)
    model = ckpt.get("model")
    if not isinstance(model, dict):
        print("FAILED: chkpt.pt has no 'model' state dict.")
        return 2
    if args.key not in model:
        near = [k for k in model if "prefix" in k]
        print(f"FAILED: {args.key} is not in the checkpoint.  Keys under the "
              f"prefix buffer: {near}")
        return 2

    w = model[args.key]
    W_old, D = int(w.shape[0]), int(w.shape[1])
    W_new = args.n_vec
    print(f"  {args.key}: [{W_old}, {D}] -> [{W_new}, {D}]")
    if not 0 < W_new <= W_old:
        print(f"FAILED: --n_vec {W_new} must be in 1..{W_old}.  Widening is not "
              "a slice: new rows would be untrained, and a summary slot that "
              "enters at the wrong scale is a dead key, not a fresh start.")
        return 2

    ok = True

    # -- the two pre-flight diagnostics -----------------------------------
    seeded = model.get("cortex.prefix.summary_seeded")
    seeded_ok = seeded is not None and bool(seeded)
    print(f"  summary_seeded = {None if seeded is None else bool(seeded)}")
    if not seeded_ok:
        ok = False
        print("  !! the parent never seeded summary_emb from wte[eos], so this "
              "is post_init NOISE, not a trained write path.")

    spread = float((w.float() - w.float().mean(dim=0, keepdim=True))
                   .norm(dim=-1).max())
    print(f"  max row deviation from the row mean: {spread:.6g}")
    if spread < 1e-6:
        ok = False
        print("  !! every row is identical — this is the wte[eos] seed "
              "unchanged, i.e. a write path that never trained.  Slicing it is "
              "meaningless and the problem is upstream.")

    # Any OTHER tensor carrying the old width would be left at the wrong shape.
    others = [k for k, v in model.items()
              if k.startswith("cortex.prefix.") and k != args.key
              and torch.is_tensor(v) and v.ndim >= 1 and int(v.shape[0]) == W_old]
    if others:
        ok = False
        print(f"  !! other prefix tensors carry width {W_old} and this tool "
              f"does not know what they mean: {others}.  Slicing only "
              f"{args.key} would leave the buffer internally inconsistent.")

    if not ok and not args.force:
        print("\nREFUSING to write.  Re-read the lines marked !! above; "
              "--force exists for tests, not for a real branch.")
        return 3

    # -- the slice ---------------------------------------------------------
    kept, dropped = w[:W_new], w[W_new:]
    model[args.key] = kept.clone()

    report = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "src": args.src,
        "dst": args.dst,
        "key": args.key,
        "W_old": W_old,
        "W_new": W_new,
        "rule": "first W_new rows (tail layout: slot j attends slots < j)",
        "summary_seeded": None if seeded is None else bool(seeded),
        "kept": _row_stats(kept),
        "dropped": _row_stats(dropped) if dropped.shape[0] > 1 else None,
        "all": _row_stats(w),
        "cross_centred_cos": (_cross_cos(kept, dropped)
                              if dropped.shape[0] > 0 else None),
        "forced": bool(args.force and not ok),
    }

    print("\n-- what was kept, and what it cost ------------------------------")
    for tag in ("all", "kept", "dropped"):
        r = report[tag]
        if r is None:
            continue
        print(f"  {tag:<8} rows={r['rows']:<3d} |row| {r['norm_mean']:8.3f} "
              f"[{r['norm_min']:.3f}, {r['norm_max']:.3f}]  "
              f"centred cos {r['centred_cos_mean']:+.4f}  "
              f"eff. rank {r['participation_rank']:.2f}")
    if report["cross_centred_cos"] is not None:
        print(f"  kept vs dropped, centred cosine: "
              f"{report['cross_centred_cos']:+.4f}")
    if report["all"]["participation_rank"] > 0:
        keep_frac = (report["kept"]["participation_rank"]
                     / report["all"]["participation_rank"])
        print(f"  effective rank retained: {100 * keep_frac:.0f}% of the "
              f"parent's, at {100 * W_new / W_old:.0f}% of the write columns")
        print("  (P1.0's bet, on trained weights: halving W costs far less rank "
              "than half.\n   If this reads near 50%, the rows WERE carrying "
              "W independent things and the\n   horizon argument has to be "
              "re-made against a real cost.)")

    # -- write --------------------------------------------------------------
    os.makedirs(args.dst, exist_ok=True)
    # Carry any sidecar files (config.json, tokenizer, the grafted modeling
    # file) so --dst is a drop-in --branch_path.
    for name in os.listdir(args.src):
        if name == "chkpt.pt":
            continue
        s = os.path.join(args.src, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.dst, name))

    # Provenance, and the reason train.py can refuse a resume off this file:
    # the optimizer state still holds a [W_old, D] moment for summary_emb and
    # restoring it would either crash or, worse, be silently reshaped.  A
    # BRANCH never restores the optimizer, which is the only intended use.
    ckpt["cortex_slice"] = report
    torch.save(ckpt, os.path.join(args.dst, "chkpt.pt"))
    with open(os.path.join(args.dst, "cortex_slice.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  wrote {os.path.join(args.dst, 'chkpt.pt')}")
    print(f"  wrote {os.path.join(args.dst, 'cortex_slice.json')}")
    print("\n  BRANCH off this dir (--branch_path).  A --resume_path is refused: "
          "the\n  optimizer state in it still has the parent's width.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
