"""
Stitching a chain of wandb links back into one series.

The tool's whole job is to decide WHICH link owns an overlapping step, and the
ways it can be wrong all produce a CSV that looks perfectly healthy: a series
that re-traverses 200 steps, a branch that was rolled back kept in preference
to the one that survived, a gap where a link went unexported.  So the tests
here are about ownership, about the chain shapes the rule cannot describe (it
must refuse, not guess), and about the paired contrast being split at the phase
boundary rather than pooled across it.

Run: python -m pytest tests/test_stitch_wandb_links.py -q
"""
from __future__ import annotations

import csv
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.stitch_wandb_links import (  # noqa: E402
    dedup_diag, detect_containment, order_links, pair_runs, paired_stats,
    read_link, stitch,
)


def _ordered(links, created=None):
    return order_links(links, created or {})[0]


def _link(tmp_path, name, run_id, first, last, loss=2.5, project="proj"):
    """One export CSV, exactly the shape pull_wandb_metrics.py writes."""
    d = tmp_path / "wandb_exports" / project
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}__{run_id}.csv"
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["train/step", "train/loss", "train/total_norm"])
        for s in range(first, last + 1):
            w.writerow([s, f"{loss + s * 1e-6:.8f}", "0.5"])
    return read_link(str(p))


def test_later_link_owns_the_overlap(tmp_path):
    # A link that died at 1,200 was resumed from checkpoint_1,000: steps
    # 1,001..1,200 were trained twice and only the second traversal is on the
    # lineage that continued.
    a = _link(tmp_path, "arm", "aaa", 1, 1200)
    b = _link(tmp_path, "arm", "bbb", 1001, 1500)
    res = stitch(_ordered([b, a]))

    steps = [r["_step"] for r in res["rows"]]
    assert steps == list(range(1, 1501))
    owners = {r["_step"]: r["link_id"] for r in res["rows"]}
    assert owners[1000] == "aaa"
    assert owners[1001] == "bbb", "the resumed link must own the overlap"
    assert owners[1200] == "bbb"
    assert res["manifest"][0]["rows_rolled_back"] == 200
    assert res["gaps"] == []


def test_fully_superseded_link_is_dropped_and_named(tmp_path):
    # A false start: resumed from the same checkpoint as the link that went on
    # to run, died before logging past it.  It contributes nothing.
    dead = _link(tmp_path, "arm", "dead", 1001, 1002)
    live = _link(tmp_path, "arm", "live", 1001, 1400)
    res = stitch(_ordered([live, dead]))

    assert res["superseded"] == ["dead"]
    assert all(r["link_id"] == "live" for r in res["rows"])
    assert [r["_step"] for r in res["rows"]] == list(range(1001, 1401))


def test_a_rollback_is_caught_rather_than_mis_stitched(tmp_path):
    # A link relaunched from checkpoint_399 while an earlier-logged link had
    # already reached 900.  First-step ordering puts the rollback first and
    # would truncate it at 500, silently losing 901..1200.  It must be caught.
    a = _link(tmp_path, "arm", "aaa", 500, 900)
    b = _link(tmp_path, "arm", "bbb", 400, 1200)
    ordered, by = order_links([a, b], {})
    assert by == "first_step"
    assert detect_containment(ordered) == [("bbb", "aaa")]

    # wandb's clock resolves it: the rollback ran last, so it owns everything
    # from its own resume point and the link it rolled back over is dropped.
    created = {"aaa": "2026-10-01T00:00:00", "bbb": "2026-10-02T00:00:00"}
    ordered, by = order_links([a, b], created)
    assert by == "created_at"
    assert detect_containment(ordered) == []
    res = stitch(ordered)
    assert res["superseded"] == ["aaa"]
    assert [r["_step"] for r in res["rows"]] == list(range(400, 1201))


def test_a_gap_is_reported_not_papered_over(tmp_path):
    a = _link(tmp_path, "arm", "aaa", 1, 100)
    b = _link(tmp_path, "arm", "bbb", 151, 200)
    res = stitch(_ordered([a, b]))

    assert res["gaps"] == [(101, 150)]
    assert len(res["rows"]) == 150


def test_a_step_logged_twice_inside_one_link_is_refused(tmp_path):
    # Truncation at the next link's start makes the ACROSS-link series
    # increasing by construction, so the invariant check exists for the one
    # thing it cannot rule out: a single export that carries the same step
    # twice.  That is a duplicate the ownership rule says nothing about, and
    # silently keeping both would double-count it in every paired statistic.
    a = _link(tmp_path, "arm", "aaa", 1, 100)
    a["rows"].insert(50, dict(a["rows"][50]))
    with pytest.raises(SystemExit) as e:
        stitch([a])
    assert "strictly increasing" in str(e.value)


def test_created_at_overrides_first_step_ordering(tmp_path):
    # A false start: launched from checkpoint_1,000, logged two steps, died.
    # The link that actually continued resumed from the SAME checkpoint, so
    # first-step ordering cannot separate them but wandb's clock can.
    dead = _link(tmp_path, "arm", "dead", 1001, 1002)
    live = _link(tmp_path, "arm", "live", 1001, 1400)
    created = {"dead": "2026-10-01T00:00:00", "live": "2026-10-02T00:00:00"}
    assert [l["id"] for l in _ordered([live, dead], created)] == ["dead", "live"]
    # and with no created_at the tie-break reaches the same answer by length:
    # the shorter link goes first so the longer one supersedes it.
    assert [l["id"] for l in _ordered([dead, live])] == ["dead", "live"]


def test_stitched_columns_survive_a_link_that_logged_an_extra_key(tmp_path):
    a = _link(tmp_path, "arm", "aaa", 1, 10)
    b = _link(tmp_path, "arm", "bbb", 11, 20)
    b["cols"].append("train/lr_group2")
    for r in b["rows"]:
        r["train/lr_group2"] = "1e-5"
    res = stitch(_ordered([a, b]))
    assert len(res["rows"]) == 20
    # the union is the caller's job; what matters here is that a row missing
    # the key does not take the key's value from its neighbour
    assert "train/lr_group2" not in res["rows"][0]


def test_paired_stats_and_the_phase_split(tmp_path, capsys):
    # Arm b is 0.10 ahead before the boundary and 0.02 ahead after it.  Pooled,
    # that is 0.06 -- a number neither phase ever had.
    a = _link(tmp_path, "arm_a", "a1", 1, 200, loss=3.0)
    rows_a = a["rows"]
    rows_b = []
    for r in rows_a:
        s = r["_step"]
        rows_b.append({"_step": s, "train/step": str(s),
                       "train/loss": f"{float(r['train/loss']) - (0.10 if s < 101 else 0.02):.8f}"})

    out = tmp_path / "paired.csv"
    pair_runs("a", rows_a, "b", rows_b, heal_end=101, out_path=str(out), tail=50)
    text = capsys.readouterr().out

    assert "heal  (step < 101)" in text and "mix   (step >= 101)" in text
    heal = paired_stats([-0.10] * 100)
    assert heal["mean"] == pytest.approx(-0.10)
    with open(out, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 200
    assert float(rows[0]["delta_b_minus_a"]) == pytest.approx(-0.10)
    assert float(rows[-1]["delta_b_minus_a"]) == pytest.approx(-0.02)


def test_paired_stats_needs_two_points():
    assert paired_stats([])["n"] == 0
    assert "mean" not in paired_stats([-0.1])
    st = paired_stats([-0.1, -0.2, -0.3])
    assert st["mean"] == pytest.approx(-0.2)
    assert st["ci_lo"] < st["mean"] < st["ci_hi"]


def test_diag_dedup_keeps_the_last_append(tmp_path):
    # train.py appends across links, so a re-walked step appears twice and the
    # file is not monotone.  The second append is the surviving lineage's.
    src = tmp_path / "cortex_diag.jsonl"
    with open(src, "w", encoding="ascii") as fh:
        for s, v in [(50, 1.0), (100, 1.0), (150, 1.0), (100, 2.0), (150, 2.0), (200, 2.0)]:
            fh.write(json.dumps({"step": s, "z_row_norm": v}) + "\n")
        fh.write("not json\n")

    out = tmp_path / "dedup.jsonl"
    dedup_diag(str(src), str(out))
    rows = [json.loads(l) for l in open(out, encoding="ascii")]

    assert [r["step"] for r in rows] == [50, 100, 150, 200]
    assert [r["z_row_norm"] for r in rows] == [1.0, 2.0, 2.0, 2.0]
