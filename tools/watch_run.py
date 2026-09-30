#!/usr/bin/env python
"""Is this 5B link healthy, and will the chain finish?  Login node, no GPU.

Run it after every link of pace/control_5b.sbatch or pace/cortex_final_5b.sbatch
(the shared pace/final_5b_common.sh calls it itself on
exit) and any time you want to know where a run stands.  It reads two files and
nothing else -- the Slurm log and the run's cortex_diag.jsonl -- so it costs
seconds, needs no allocation, no wandb login and no checkpoint.

    python tools/watch_run.py --run cortex-5b/cortex-final \
        --log logs/Report-14000001.out --arm cortex --znorm_target 136.0
    python tools/watch_run.py --run cortex-5b/c-chunked --arm control   # log optional

WHAT IT IS FOR, AND WHAT IT IS NOT FOR
--------------------------------------
It answers the questions a loss curve cannot, and every one of them has cost
this project a run:

  * did the pack this link CLAIMS to have read actually arrive (the ARM_DATA
    trap: 1.7B tokens on the wrong corpus, healthy curve throughout);
  * did the corpus-switch link fast-forward when it should not have, or fail
    to when it should have (the B1 incident: exits clean, wandb says
    "finished", 1.15B tokens short);
  * is the mechanism ATTACHED TO THE LOSS -- did either gate leave its
    exactly-zero init, is Z's read receiving gradient, did the stagger fire,
    is the projection still frozen.  e_carry_read parsed, printed and
    persisted while doing nothing for two whole runs; the only thing that
    catches that class is a number the forward pass had to produce;
  * is Z entering at E's scale, or has ||E|| drifted out from under a
    znorm_target that was fixed before the run (P2.5's question, asked live);
  * how many 48h links are left, from THIS link's measured throughput.

It CANNOT say whether memory helps.  That needs the 2x2 and the held-out NLL
on a real checkpoint -- pace/midrun_check.sbatch.  This tool prints that
sentence in its own output rather than trusting a reader to remember it,
because a green health check is exactly the kind of thing that gets quoted
later as "the run worked".

EXIT CODES
    0  nothing failed (warnings may still be printed)
    1  at least one FAIL -- something about this link is not what it claims
    2  could not read what it needed to judge
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

# --- the bars, pre-registered here so they are not invented while reading ---

#: Z is rescaled to znorm_target before the splice, so E's own carried row norm
#: drifting away from it puts Z in at the wrong scale.  15% is the band the J6
#: smoke gate used against a MEASURED 136.0 and it is kept for continuity.
ZNORM_BAND = 0.15

#: latent_embed frozen at its identity init is an exact no-op, measured at gain
#: 0.999991x over 80 diag rows.  z_embed_ratio is Z's spliced norm against E's;
#: under the rms rescale it should sit at ~1.0 and STAY there.  A trainable
#: projection went 2.17x within 23 steps and 4.13x by step 93,550, so anything
#: outside this band means the freeze is not holding.
EMBED_RATIO_BAND = (0.80, 1.25)

#: gate_init="zero" sets both projections to exactly zero, so any movement at
#: all proves a gradient reached them.  Same constant as health.GATE_MOVED_MIN.
GATE_MOVED_MIN = 1e-6

#: A staggered band at mr8 draws depths that land in the no-grad PREFIX on a
#: real share of batches -- J6 measured 0.762 of slots on average and ENTIRELY
#: frozen on 13.8% of records.  So write_grad_frac < 1.0 is the design, not a
#: defect; only a write that NEVER trains is a failure.  Reported, not gated.
WRITE_GRAD_EXPECTED = 0.76

#: The seconds a 48h link actually spends training: 48h minus the 20-minute
#: pre-timeout save, minus a few minutes of module load and checkpoint restore.
LINK_SECONDS = 47.5 * 3600

_STEP = re.compile(
    r"Step:\s*(\d+)\s*\|\s*Updates:\s*(\d+)\s*\|\s*Time/step:\s*([\d.]+)\s*\|\s*"
    r"Tok/sec=\s*([\d.]+)\s*\|\s*Loss:\s*([\d.naif-]+)\s*/\s*log-ppl:\s*[\d.naif-]+\s*\|\s*"
    r"Grad-Norm\s*([\d.naif-]+)\s*\|\s*ClipCoef\s*([\d.naif-]+)"
    r"(?:\s*\|\s*Peak-Mem\s*([\d.]+)GiB)?")


def _f(x, spec=".4g", dash="-"):
    return dash if x is None else format(x, spec)


# ---------------------------------------------------------------------------
# the Slurm log
# ---------------------------------------------------------------------------

def read_log(path: str) -> dict:
    """Everything the link's stdout can prove about itself."""
    out = {"banner": None, "data_line": None, "fast_forward": None,
           "tok_per_step": None,
           "steps": [], "nonfinite": 0, "exhausted": False, "oom": False,
           "peak_mem": None, "commit": None, "znorm_echo": None,
           "window_backward": None, "saves": 0}
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        out["error"] = str(exc)
        return out
    for ln in lines:
        if out["banner"] is None and ln.startswith("=== 5B "):
            out["banner"] = ln.rstrip()
        if "    data:" in ln and out["data_line"] is None:
            out["data_line"] = ln.split("data:", 1)[1].strip()
        if "[data] fast-forward:" in ln:
            out["fast_forward"] = ln.strip()
        if "[data] reset_dataset_position:" in ln:
            out["reset_position"] = ln.strip()
        if out.get("tok_per_step") is None:
            m2 = re.search(r"-> (\d+) tok/step", ln)
            if m2:
                out["tok_per_step"] = int(m2.group(1))
        if "    commit:" in ln:
            out["commit"] = ln.split("commit:", 1)[1].strip()
        if "    Z scale:" in ln:
            out["znorm_echo"] = ln.strip()
        if "window_backward: one backward per" in ln:
            out["window_backward"] = ln.strip()
        if "non-finite grad-norm" in ln:
            out["nonfinite"] += 1
        if "Dataloader exhausted" in ln:
            out["exhausted"] = True
        if "CUDA out of memory" in ln or "OutOfMemoryError" in ln:
            out["oom"] = True
        if "Saving checkpoint" in ln or "save_checkpoint" in ln:
            out["saves"] += 1
        m = _STEP.search(ln)
        if m:
            def _num(s):
                try:
                    return float(s)
                except ValueError:
                    return float("nan")
            out["steps"].append({
                "data_step": int(m.group(1)), "update": int(m.group(2)),
                "time_step": _num(m.group(3)), "tok_sec": _num(m.group(4)),
                "loss": _num(m.group(5)), "gnorm": _num(m.group(6)),
                "clip": _num(m.group(7)),
                "peak_mem": _num(m.group(8)) if m.group(8) else None})
    mems = [s["peak_mem"] for s in out["steps"] if s["peak_mem"] is not None]
    if mems:
        out["peak_mem"] = max(mems)
    return out


def report_log(lg: dict, arm: str, max_steps: int, heal_end: int,
               fails: list, warns: list, notes: list) -> None:
    print("--- the link, from its own stdout " + "-" * 40)
    if lg.get("error"):
        warns.append(f"could not read the log: {lg['error']}")
        print(f"  (no log: {lg['error']})")
        return
    print(f"  banner   {lg['banner'] or '(ABSENT)'}")
    if lg["banner"] is None:
        fails.append("no '=== 5B ' banner in the log.  The banner is the only "
                     "proof of which arm, phase and pack this link ran -- "
                     "without it nothing below is attributable.")
    print(f"  pack     {lg['data_line'] or '(ABSENT)'}")
    print(f"  commit   {lg['commit'] or '(not recorded)'}")
    if lg.get("reset_position"):
        print(f"  {lg['reset_position']}")
    print(f"  {'fast-fwd ' + lg['fast_forward'] if lg['fast_forward'] else 'fast-fwd (none -- this link read its pack from the top)'}")

    if lg["oom"]:
        fails.append("CUDA OOM in the log.  Lower MICRO_BS *and start a new "
                     "run_name* -- mid-chain mbs changes skip the wrong rows.")
    if lg["exhausted"]:
        fails.append("Dataloader exhausted.  train.py now RAISES on this "
                     "instead of exiting clean, so the run stopped short of "
                     "the horizon.  Check the pack against "
                     "tools/check_pack.py --resume_rows <consumed>.")
    if lg["nonfinite"]:
        (warns if lg["nonfinite"] < 5 else fails).append(
            f"{lg['nonfinite']} non-finite grad-norm guard hits -- those "
            f"updates were SKIPPED.  20 consecutive aborts the run.")

    st = lg["steps"]
    if not st:
        warns.append("no per-step lines parsed from the log -- nothing to say "
                     "about throughput or the loss trend.")
        return

    first, last = st[0], st[-1]
    span = last["update"] - first["update"]
    tps = [s["tok_sec"] for s in st if s["tok_sec"] == s["tok_sec"]]
    tps_med = sorted(tps)[len(tps) // 2] if tps else float("nan")
    print(f"  updates  {first['update']:,} -> {last['update']:,} "
          f"({span:,} this link, {last['update']/max_steps:.1%} of the horizon)")
    if last["update"] > max_steps:
        fails.append(
            f"the update counter is at {last['update']:,}, PAST the "
            f"{max_steps:,} horizon.  Either --max_steps here is wrong for "
            "this run, or the link resumed a checkpoint from a different "
            "chain -- and the LR cosine and recurrence ramp are both keyed to "
            "the horizon, so a mismatched one re-runs cooldown per segment "
            "(the 2026-06-24 sawtooth, baked into the weights).")
    print(f"  tok/sec  median {tps_med:,.0f}   peak-mem {_f(lg['peak_mem'], '.1f')} GiB")

    if lg["peak_mem"] and lg["peak_mem"] > 130.0:
        warns.append(f"peak memory {lg['peak_mem']:.1f} GiB of an H200's 138.7 "
                     "-- little headroom; a longer row or a ragged batch OOMs.")

    # --- the projection.  THIS is what decides whether to keep queueing. ----
    if tps_med == tps_med and tps_med > 0:
        tok_per_step = lg.get("tok_per_step")
        if tok_per_step:
            left = (max_steps - last["update"]) * tok_per_step
            hours = left / tps_med / 3600.0
            links = math.ceil(hours * 3600.0 / LINK_SECONDS)
            print(f"  REMAINING {left/1e9:.2f}B tokens = {hours:.0f} GPU-h "
                  f"= ~{links} more 48h link(s) at this rate")
            if last["update"] < heal_end:
                to_heal = (heal_end - last["update"]) * tok_per_step / tps_med / 3600.0
                print(f"            heal boundary {heal_end:,} is "
                      f"{to_heal:.1f} h away")

    # --- the loss trend, on the log's own sampling --------------------------
    def _mean(rows, key):
        v = [r[key] for r in rows if r[key] == r[key]]
        return sum(v) / len(v) if v else float("nan")
    half = max(1, len(st) // 2)
    l_early, l_late = _mean(st[:half], "loss"), _mean(st[half:], "loss")
    print(f"  loss     {l_early:.4f} -> {l_late:.4f} over this link "
          f"(first half vs second)")
    if l_late > l_early + 0.05:
        warns.append(f"loss ROSE {l_late - l_early:+.4f} across this link. "
                     "Expected across the heal->mix switch (a new corpus is a "
                     "real distribution change); not expected within a phase.")
    if any(r["loss"] != r["loss"] for r in st):
        fails.append("NaN in the logged loss.")

    jumps = [(b["update"], b["loss"] - a["loss"])
             for a, b in zip(st, st[1:])
             if b["loss"] == b["loss"] and a["loss"] == a["loss"]]
    if jumps:
        worst = max(jumps, key=lambda t: t[1])
        print(f"  biggest loss jump  {worst[1]:+.4f} at update {worst[0]:,}")
        if worst[1] > 0.5:
            warns.append(f"loss jumped {worst[1]:+.4f} in one logged interval "
                         f"at update {worst[0]:,}.")

    gn = [r["gnorm"] for r in st if r["gnorm"] == r["gnorm"]]
    if gn:
        gmax = max(gn)
        clipped = sum(1 for r in st if r["clip"] == r["clip"] and r["clip"] < 1.0)
        print(f"  grad     mean {sum(gn)/len(gn):.3f}  max {gmax:.3f}  "
              f"clipped on {clipped}/{len(st)} logged steps")
        # B1's accum arm had heavy-tail spikes its control did not: gnorm max
        # 123.7 vs 8.7.  This run sits at HIGHER recurrence than B1 reached.
        if gmax > 50.0:
            warns.append(f"grad-norm max {gmax:.1f}.  B1's accum arm spiked to "
                         "123.7 where its control held 8.7; this run is at "
                         "higher recurrence than B1 ever reached.  Watch it.")
    notes.append("the log side says the link RAN as configured; it says "
                 "nothing about whether memory helps.")


# ---------------------------------------------------------------------------
# the architecture diagnostic
# ---------------------------------------------------------------------------

def read_diag(path: str) -> list:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return rows


def _series(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def report_diag(rows: list, arm: str, znorm_target, fails: list, warns: list,
                notes: list) -> None:
    print()
    print("--- the mechanism, from cortex_diag.jsonl " + "-" * 32)
    if arm == "control":
        if rows:
            fails.append(f"the CONTROL wrote {len(rows)} architecture diag "
                         "rows.  training_diag returns {} when there is no "
                         "cortex, so a non-empty file means a graft was built "
                         "-- this is not a no-memory arm.")
            print(f"  {len(rows)} rows -- and there should be NONE.")
        else:
            print("  empty, as it must be: no cortex, nothing to report.")
            notes.append("control: no mechanism to check.  Its health lives in "
                         "the loss curve and in held-out NLL "
                         "(pace/midrun_check.sbatch).")
        return

    if not rows:
        fails.append("no cortex_diag.jsonl rows on a MEMORY arm.  Either "
                     "--cortex.diag_interval is 0 or the graft was never "
                     "built, and either way nothing here is verifiable.")
        print("  (no rows)")
        return

    last = rows[-1]
    print(f"  {len(rows)} rows, last at step {last.get('step')}, "
          f"loss {_f(last.get('loss'))}")

    # ---- 1. the channel exists and is spliced ------------------------------
    if last.get("z_n_pre") != 64:
        fails.append(f"z_n_pre {last.get('z_n_pre')} != 64: E's carried "
                     "columns were not spliced.")
    if last.get("z_n_zpre") != 64:
        fails.append(f"z_n_zpre {last.get('z_n_zpre')} != 64: Z's columns "
                     "were not spliced, so there is no Z read.")
    if not last.get("z_e_spliced_norm"):
        fails.append("z_e_spliced_norm is 0/absent: E's rows never reached "
                     "the token stream.")
    print(f"  columns  E {last.get('z_n_pre')}  Z {last.get('z_n_zpre')}  "
          f"writes/chunk {last.get('z_n_sum')}  "
          f"E spliced norm {_f(last.get('z_e_spliced_norm'), '.1f')}")

    # ---- 2. gradient actually reaches both paths ---------------------------
    rg = _series(rows, "z_read_grad_frac")
    wg = _series(rows, "z_write_grad_frac")
    if not rg or min(rg) < 1.0:
        fails.append(f"z_read_grad_frac min {_f(min(rg) if rg else None)}, "
                     "expected 1.0 -- the embeds read is live by construction, "
                     "so below 1.0 means the read site moved.")
    if not wg or max(wg) == 0:
        fails.append("z_write_grad_frac 0 on every record: the staggered "
                     "write never entered the gradient window, so Z is "
                     "carrying an untrained encoding.")
    elif wg:
        mean_w = sum(wg) / len(wg)
        frozen_share = sum(1 for w in wg if w == 0) / len(wg)
        print(f"  gradient read {min(rg) if rg else float('nan'):.3f} (min)   "
              f"write {mean_w:.3f} mean, {min(wg):.3f} min, "
              f"{frozen_share:.1%} of records fully frozen")
        print(f"           <1.0 on the write is the DESIGN here -- a staggered "
              f"band draws depths inside the no-grad prefix (J6 measured "
              f"{WRITE_GRAD_EXPECTED:.2f} mean, 13.8% frozen)")
        if mean_w < 0.5 * WRITE_GRAD_EXPECTED:
            warns.append(f"z_write_grad_frac mean {mean_w:.3f} is far under "
                         f"J6's {WRITE_GRAD_EXPECTED:.2f}.  The band is being "
                         "clamped by short sampled loops more than expected.")

    # ---- 3. the stagger fired, and the projection is still frozen ----------
    nd = last.get("z_n_depths") or 0
    tl = last.get("z_tape_len") or 0
    if tl < 2:
        fails.append(f"z_tape_len {tl}: the loop tape is empty, so the "
                     "staggered write had no depths to draw from.")
    elif nd < 2:
        fails.append(f"z_n_depths {nd}: the write drew ONE depth.  The stagger "
                     "collapsed and this is a single-depth write under a "
                     "staggered name -- the inert-config failure J4/J5 shipped.")
    nds = _series(rows, "z_n_depths")
    if nds:
        print(f"  stagger  n_depths last {nd}, median "
              f"{sorted(nds)[len(nds)//2]}, "
              f"collapsed to 1 on {sum(1 for d in nds if d < 2)}/{len(nds)} "
              f"records   tape_len {tl}")

    ef = last.get("z_embed_frozen")
    if ef is None:
        fails.append("no z_embed_frozen in the diag: the flag never reached "
                     "the graft, so the condition is unverifiable.")
    elif int(ef) != 1:
        # NOT `is not True`: training_diag casts bools to int for wandb, and
        # `1 is not True` is True in Python -- the check that failed a run
        # whose projection WAS frozen (13662406_2).  Compare the VALUE.
        fails.append("z_embed_frozen is 0: the projection is TRAINING.  That "
                     "is J4's read, whose measured content share was 0%.")
    er = _series(rows, "z_embed_ratio")
    if er:
        lo, hi = EMBED_RATIO_BAND
        print(f"  embed    frozen={ef}  z_embed_ratio first {er[0]:.3f} "
              f"-> last {er[-1]:.3f} (identity holds it at ~1.0; a trainable "
              f"map went 2.17x in 23 steps)")
        if not (lo <= er[-1] <= hi):
            fails.append(f"z_embed_ratio {er[-1]:.3f} is outside "
                         f"[{lo}, {hi}] with the projection nominally frozen. "
                         "Either the freeze is not holding or ||E|| moved "
                         "under the rescale.")

    # ---- 4. Z's injection scale against E's actual norm (P2.5, live) -------
    en = _series(rows, "z_e_carried_norm")
    if en:
        drift = (en[-1] - en[0]) / max(en[0], 1e-9)
        print(f"  ||E||    {en[0]:.1f} -> {en[-1]:.1f} over the trace "
              f"({drift:+.1%})")
        if znorm_target:
            off = (en[-1] - znorm_target) / znorm_target
            print(f"           znorm_target {znorm_target:.1f}, "
                  f"now {off:+.1%} away")
            if abs(off) > ZNORM_BAND:
                fails.append(
                    f"z_e_carried_norm {en[-1]:.1f} is {off:+.1%} from "
                    f"znorm_target {znorm_target:.1f} (band "
                    f"+-{ZNORM_BAND:.0%}).  Z is being rescaled to a norm E no "
                    "longer has, so it enters the splice at the wrong strength "
                    "-- the measured s0 failure mode.  This is P2.5's question "
                    "and it is now answered against you: decide whether to "
                    "re-target on the next link and record that you did.")
        if abs(drift) > 0.10:
            warns.append(f"||E|| has moved {drift:+.1%} across the trace. "
                         "Settled high is a calibration constant; still "
                         "climbing is an instability with a schedule -- "
                         "compare the first and last thirds before deciding.")
    else:
        warns.append("no z_e_carried_norm in the diag, so Z's injection scale "
                     "cannot be checked against E's actual norm.")

    # ---- 5. both gates left their exactly-zero init ------------------------
    gl = last.get("gate_left_init")
    gzl = last.get("gate_z_left_init")
    print(f"  gates    E left-init={gl} (w {_f(last.get('gate_in_w_norm'), '.3g')})"
          f"   Z left-init={gzl} (w {_f(last.get('gate_in_z_w_norm'), '.3g')})")
    print(f"           fg {_f(last.get('fg_at_bias'), '.3f')} / "
          f"ig {_f(last.get('ig_at_bias'), '.3f')}   "
          f"fg_z {_f(last.get('fg_z_at_bias'), '.3f')} / "
          f"ig_z {_f(last.get('ig_z_at_bias'), '.3f')}   "
          f"summary_emb {_f(last.get('summary_emb_norm'), '.1f')}")
    step = last.get("step") or 0
    if step > 600:
        if not gl:
            fails.append("the E gate has NOT left its exactly-zero init after "
                         f"{step} steps: no gradient is reaching it, and the "
                         "buffer is running as a plain append.")
        if gzl is not None and not gzl:
            fails.append("the Z gate has NOT left its exactly-zero init after "
                         f"{step} steps.")

    # ---- 6. carry geometry: collapse, and the ring's fill -------------------
    print(f"  carry    rows {last.get('rows')}  "
          f"E norm {_f(last.get('e_row_norm'), '.1f')} "
          f"rank_pr {_f(last.get('e_eff_rank_pr'), '.2f')} "
          f"cos {_f(last.get('e_centred_cosine'), '.3f')}")
    print(f"           Z norm {_f(last.get('z_row_norm'), '.3g')} "
          f"rank_pr {_f(last.get('z_eff_rank_pr'), '.2f')} "
          f"cos {_f(last.get('z_centred_cosine'), '.3f')}  "
          f"zero rows {last.get('z_zero_rows')}")
    pr = _series(rows, "e_eff_rank_pr")
    if pr and len(pr) > 4 and pr[-1] < 0.5 * (sum(pr[:len(pr)//4]) / max(1, len(pr)//4)):
        warns.append(f"E's carry effective rank has halved over the trace "
                     f"({pr[0]:.2f} -> {pr[-1]:.2f}): the carry is collapsing "
                     "toward a few directions.")
    zz = last.get("z_zero_rows")
    if zz and step > 600:
        warns.append(f"z_zero_rows {zz}: the ring has slots it has never "
                     "written.  Early in a chain this is fill, not death -- "
                     "it should reach 0 once chunk 4 of every row has run.")

    # ---- 7. no treatment flag has leaked in --------------------------------
    leaks = []
    if last.get("z_mix_frac"):
        leaks.append(f"z_mix_frac {last['z_mix_frac']:.4f} (J6's `mix` limb)")
    if (last.get("z_stride") or 1) != 1:
        leaks.append(f"z_stride {last['z_stride']} (J6's `slow` limb)")
    if last.get("z_e_carry_read") is not None and int(last["z_e_carry_read"]) != 1:
        leaks.append("e_carry_read off (the Z-only arm)")
    if last.get("z_e_dropout"):
        leaks.append(f"e_dropout {last['z_e_dropout']}")
    if leaks:
        fails.append("cortex-final is carrying a J-arm treatment flag: "
                     + "; ".join(leaks))
    else:
        print("  clean    no mix, stride 1, E read on, no e_dropout")

    notes.append("every line above is a MEASURED number the forward pass had "
                 "to produce, not a flag echo.  None of them says whether Z "
                 "helps: that is pace/midrun_check.sbatch.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="the run dir, e.g. cortex-5b/cortex-final")
    p.add_argument("--log", default=None,
                   help="this link's Slurm log (logs/Report-<jobid>.out). "
                        "Omit to check the diag alone.")
    p.add_argument("--arm", default=None, choices=["control", "cortex"],
                   help="default: inferred from the run dir's name")
    p.add_argument("--max_steps", type=int, default=38147)
    p.add_argument("--heal_end", type=int, default=11444)
    p.add_argument("--znorm_target", type=float, default=None,
                   help="the value the run was launched with; without it the "
                        "||E|| drift is printed but not judged")
    args = p.parse_args()

    arm = args.arm
    if arm is None:
        arm = "cortex" if "final" in os.path.basename(args.run.rstrip("/")) \
              else "control"

    if not os.path.isdir(args.run):
        print(f"ERROR: no such run dir: {args.run}")
        return 2

    print(f"=== watch_run | {args.run} | arm={arm} ===")
    fails: list = []
    warns: list = []
    notes: list = []

    if args.log:
        report_log(read_log(args.log), arm, args.max_steps, args.heal_end,
                   fails, warns, notes)
    else:
        print("--- (no --log passed; skipping the link's stdout) ---")

    report_diag(read_diag(os.path.join(args.run, "cortex_diag.jsonl")),
                arm, args.znorm_target, fails, warns, notes)

    ckpts = sorted(
        (int(d.rsplit("_", 1)[1]) for d in os.listdir(args.run)
         if d.startswith("checkpoint_") and d.rsplit("_", 1)[1].isdigit()))
    print()
    print("--- state on disk " + "-" * 55)
    if ckpts:
        print(f"  {len(ckpts)} resumable checkpoints, newest "
              f"checkpoint_{ckpts[-1]} "
              f"({ckpts[-1]/args.max_steps:.1%} of the horizon)")
        if ckpts[-1] == args.heal_end:
            print(f"  >> the next link is THE CORPUS SWITCH.  It, and only it, "
                  f"passes EXTRA_ARGS=\"--reset_dataset_position true\".")
    else:
        print("  no resumable checkpoints (they land every 2 x save_interval)")

    print()
    for n in notes:
        print(f"  note {n}")
    for w in warns:
        print(f"  WARN {w}")
    for f in fails:
        print(f"  FAIL {f}")
    print()
    if fails:
        print(f"VERDICT: {len(fails)} FAIL, {len(warns)} warn.  Do not queue "
              f"the next link until these are understood.")
        return 1
    print(f"VERDICT: no failures, {len(warns)} warn.  This says the link ran "
          f"as configured.  It does NOT say the mechanism helps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
