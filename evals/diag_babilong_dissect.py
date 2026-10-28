"""
BABILong dissection -- WHERE does the supporting fact get lost?

On BABILong (SEQ_LEN 512 x NUM_CHUNKS 8, T=8) both 5B arms sit near chance.
A pooled accuracy cannot say why.  This instrument scores the gold answer for
the SAME question chunk under five carry conditions and reads the failure off
their differences:

  normal          the benchmark's own chunked run with carry.  Chunking comes
                  from eval_babilong.split_context itself, so this IS the
                  benchmark's condition (and --records checks it against the
                  benchmark's own per-item records).
  nocarry         the question chunk with no incoming carry (m_cross_in=None).
                  On cortex this removes the carried COLUMNS as well as their
                  content, so it is the NO-COLUMN FLOOR, not a content control.
  teleport        the carry state produced right after the fact chunk, fed
                  straight to the question chunk, skipping every intervening
                  chunk.  Removes retention from the path: write + read only.
                  At distance 1 it is the normal state by construction, which
                  is checked as an identity (see SELF-CHECKS).
  teleport_donor  the same, but the state comes from ANOTHER item's fact chunk
                  in the same cell: same chunk index (or, failing that, same
                  row count and ring phase), so same column count, norms and
                  register -- different content.  The donor is REQUIRED to
                  not encode this item's answer (see DONOR RULE).
                  teleport - teleport_donor isolates information transfer.
  fact_in_window  the supporting fact sentence(s) inserted into the question
                  chunk, with the normal carry: the read/capability ceiling.

The control arm (no memory) runs `normal` and `fact_in_window` only; the
carry conditions are refused on a model without a prefix buffer.

WHERE THE FACT IS -- THE RULES (the dataset carries no supporting-fact ids)
  Fact sentences are found by regex over the raw context: one of the four bAbI
  names, a bAbI verb phrase, "the <location|object>", optional " there", ".".
  The vocabulary was tabulated from babilong-1k-samples (moves: moved / went /
  went back / journeyed / travelled to; takes: got / took / picked up /
  grabbed; drops: left / dropped / discarded / put down).  The parsed facts are
  then REPLAYED in story order:
    qa1  support = the LAST sentence moving the queried person.
    qa2  support = the event that fixes the object's state (the last take, or
         the drop) + the holder's last move before it.
  The replayed answer must equal the row's target or the item is SKIPPED as
  ambiguous (answer_mismatch) -- that catches a PG-19 sentence that happens to
  look like a fact, and a fact the parser missed.
  A fact's chunk is the chunk holding its tokens (via the tokenizer's offset
  map against the exact token stream split_context chunks).  A supporting fact
  that STRADDLES a chunk boundary is ambiguous and SKIPPED.  Every skip is
  counted per reason and reported.

DISTANCE.  Chunks 0..n_prime-1 are the priming chunks; the question chunk is
  index n_prime.  The teleport source is the LATEST chunk holding a supporting
  fact that is NOT already in the question window; distance = n_prime - src.
  Distance 0 = every supporting fact is already in the question window: those
  items run normal and nocarry only (teleport and fact_in_window are not
  defined).  Ring laps = distance / (gate_slots / accum_vecs).

FACT_IN_WINDOW.  ALL supporting facts, in story order, are appended to the END
  of the story part of the question window -- i.e. right before the benchmark
  suffix (instruction + demos + "Question: ...").  To keep the window length
  and every suffix position fixed, the same number of FILLER tokens is deleted
  from the FRONT of the window's story part, skipping any token that belongs
  to a parsed bAbI fact.  If the window holds too little filler -- always at
  1k, where num_chunks=8 sizes the window to ~98 tokens with 1-2 story tokens
  -- the window is instead EXTENDED by the fact's length (suffix positions
  shift by that many tokens).  Those items carry length_matched=False and are
  excluded from the "[len-matched]" contrast line; never pool them silently.

DONOR RULE.  The donor's state is its own teleport-source state (the state
  after its own fact chunk).  Required: a different row, a different gold, AND
  replaying the donor's facts up to its source chunk must NOT give this item's
  gold for this item's question.  That last clause is not optional: BABILong
  rows share bAbI story prefixes, so a neighbouring row's state usually holds
  this item's very fact.  Tier 0 = same source chunk index; tier 1 = same row
  count and same ring phase.  No compatible donor => the item has no
  teleport_donor score (counted).

SEEDING.  torch.manual_seed once per item before its chain, and
  seed_example(id) before EVERY scoring call -- exactly the benchmark's scoring
  seed, so the candidate scoring of `normal` replays the benchmark's s0 draws.
  The benchmark did NOT seed its priming chain, so the carry itself is a
  different s0 sample: `normal` reproduces the benchmark in distribution, not
  bit-for-bit (and the benchmark ran bf16).

HOW TO READ  (delta = A - B; dNLL < 0 and dTop1 > 0 both mean A is better)
  fact_in_window      READ FIRST.  Top-1 ~ chance (1/|candidates|) => the
                      CAPABILITY FLOOR binds: the model cannot answer even with
                      the fact visible, and no memory can win this task at
                      this model scale.
  teleport - donor    ~0 => the carried CONTENT is not used: a WRITE or READ
                      failure.  dNLL < 0 => information transfers once
                      retention is taken out of the path.
  teleport - normal   dNLL << 0 (teleport >> normal) => RETENTION loss: the
                      fact was written, then gated away by intervening chunks.
                      Identically 0 at distance 1.
  normal - nocarry    what the carry does at all (content + columns).

USAGE (cluster; see pace/eval_longcontext.sbatch for the data layout)
  python evals/diag_babilong_dissect.py \
      --model_name ckpts/olmo-retrofit-cortex \
      --checkpoint cortex-5b/cortex-final/checkpoint_38146 \
      --set use_memory=true ... --set latent_read_znorm_target=78.9 \
      --dataset_path data/babilong-1k --tasks qa1 qa2 --lengths 1k 2k 4k 8k \
      --records eval_results/longcontext_5b-final/cortex-final-T8-nc8-sl512/babilong \
      --out_dir eval_results/babilong_dissect/cortex-final-38146
"""
from __future__ import annotations

import argparse
import bisect
import contextlib
import io
import json
import math
import os
import random
import re
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from model_utils import (  # noqa: E402
    _unwrap, explain_missing_cortex, load_checkpoint, parse_config_overrides,
    rank_candidates, seed_example, to_num_steps,
)
from eval_babilong import (  # noqa: E402
    build_suffix, collect_candidates, contains_answer, load_bucket,
    split_context,
)

CONDITIONS = ("normal", "nocarry", "teleport", "teleport_donor",
              "fact_in_window")
CARRY_CONDITIONS = ("nocarry", "teleport", "teleport_donor")
CONTROL_CONDITIONS = ("normal", "fact_in_window")
DIST_BUCKETS = ("0", "1", "2-3", "4-7", "8+")
AGG_BUCKETS = DIST_BUCKETS + ("1+", "all")
# (A, B, label, row filter).  "equal rows" drops the items whose teleported
# state has FEWER carried columns than the normal one (gate_fill=grow: a
# source chunk inside the first lap), where teleport - normal also changes the
# column count, not only the retention path.
CONTRASTS = (
    ("teleport", "normal", "teleport - normal", None),
    ("teleport", "normal", "teleport - normal [equal rows]", "equal_rows"),
    ("teleport", "teleport_donor", "teleport - teleport_donor", None),
    ("fact_in_window", "normal", "fact_in_window - normal", None),
    ("fact_in_window", "normal", "fact_in_window - normal [len-matched]",
     "fiw_matched"),
    ("normal", "nocarry", "normal - nocarry", None),
)
# Skip reasons that say the PARSER may be wrong (vs geometry: a supporting
# fact cut by a chunk boundary is a property of the chunking, not the parse).
PARSER_SKIPS = ("question_unparsed", "no_facts_parsed", "no_supporting_fact",
                "answer_mismatch", "fact_not_tokenized")
LAP_DEFAULT = 4          # cortex-final: gate_slots 64 / accum_vecs 16
# ZNORM_TARGET is lineage-specific (memory: znorm-target-is-lineage-specific).
# The two values the cortex-final run actually trained with at these steps.
KNOWN_ZNORM = {11444: 98.0, 38146: 78.9}

NAMES = ("Mary", "John", "Daniel", "Sandra")
OBJECTS = ("football", "apple", "milk")
LOCATIONS = ("bathroom", "bedroom", "garden", "hallway", "kitchen", "office")
MOVE_VERBS = ("moved to", "went to", "went back to", "journeyed to",
              "travelled to")
TAKE_VERBS = ("got", "took", "picked up", "grabbed")
DROP_VERBS = ("left", "dropped", "discarded", "put down")


# ---------------------------------------------------------------------------
# Pure: fact parsing and replay
# ---------------------------------------------------------------------------

class Ambiguous(Exception):
    """An item whose supporting fact cannot be located unambiguously."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Fact:
    kind: str        # "move" | "take" | "drop"
    person: str
    target: str      # location for a move, object for take/drop
    start: int       # character span in the context, [start, end)
    end: int
    text: str


def _alt(words) -> str:
    # Longest first, so "went back to" wins over "went to".
    return "|".join(re.escape(w) for w in sorted(set(words), key=len,
                                                 reverse=True))


def _fact_regex(locations) -> re.Pattern:
    names = _alt(NAMES)
    move = (rf"(?P<mp>{names}) (?:{_alt(MOVE_VERBS)}) the "
            rf"(?P<loc>{_alt(locations)})\.")
    take = (rf"(?P<tp>{names}) (?:{_alt(TAKE_VERBS)}) the "
            rf"(?P<tobj>{_alt(OBJECTS)})(?: there)?\.")
    drop = (rf"(?P<dp>{names}) (?:{_alt(DROP_VERBS)}) the "
            rf"(?P<dobj>{_alt(OBJECTS)})(?: there)?\.")
    return re.compile(rf"\b(?:{move}|{take}|{drop})")


def parse_facts(context: str, locations=LOCATIONS) -> list:
    """Every bAbI fact sentence in `context`, in text order."""
    out = []
    for m in _fact_regex(tuple(locations) + LOCATIONS).finditer(context):
        if m.group("mp"):
            f = Fact("move", m.group("mp"), m.group("loc"), m.start(),
                     m.end(), m.group(0))
        elif m.group("tp"):
            f = Fact("take", m.group("tp"), m.group("tobj"), m.start(),
                     m.end(), m.group(0))
        else:
            f = Fact("drop", m.group("dp"), m.group("dobj"), m.start(),
                     m.end(), m.group(0))
        out.append(f)
    return out


def parse_question(task: str, question: str) -> str:
    """The queried entity: a person (qa1) or an object (qa2)."""
    q = question.strip()
    if task == "qa1":
        m = re.fullmatch(rf"Where is ({_alt(NAMES)})\?", q)
    elif task == "qa2":
        m = re.fullmatch(rf"Where is the ({_alt(OBJECTS)})\?", q)
    else:
        raise ValueError(f"no fact-location rule for {task}; qa1/qa2 only")
    if not m:
        raise Ambiguous("question_unparsed", q)
    return m.group(1)


def world_answer(task: str, facts: list, entity: str):
    """Replay `facts` in order -> (answer location or None, support indices).

    qa1: the last move of the person.  qa2: if the object is held, the take
    that made it held + the holder's latest move; if it was dropped, the drop +
    the dropper's last move before it.  A drop by someone not holding the
    object is an inconsistent story (a missed or spurious parse) -> (None, []).
    A drop whose dropper has no known location YET is not an error -- a later
    take/drop can still resolve the object -- it only leaves the object's
    location unknown until then.
    """
    if task == "qa1":
        last = None
        for i, f in enumerate(facts):
            if f.kind == "move" and f.person == entity:
                last = i
        if last is None:
            return None, []
        return facts[last].target, [last]
    if task != "qa2":
        raise ValueError(f"no replay rule for {task}")
    loc = {}                 # person -> index of their latest move
    state = None             # ("held", person, take_i) | ("dropped", loc, idx)
    for i, f in enumerate(facts):
        if f.kind == "move":
            loc[f.person] = i
        elif f.target == entity:
            if f.kind == "take":
                state = ("held", f.person, i)
            else:
                if state is None or state[0] != "held" or state[1] != f.person:
                    return None, []
                mv = loc.get(f.person)
                if mv is None:
                    state = ("dropped", None, [i])
                else:
                    state = ("dropped", facts[mv].target, sorted([mv, i]))
    if state is None:
        return None, []
    if state[0] == "held":
        mv = loc.get(state[1])
        if mv is None:
            return None, []
        return facts[mv].target, sorted([mv, state[2]])
    if state[1] is None:
        return None, []
    return state[1], state[2]


# ---------------------------------------------------------------------------
# Pure: chunk geometry (mirrors eval_babilong.split_context, VERIFIED per item)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    n_ctx: int
    n_sfx: int
    win: int          # effective window after the num_chunks sizing
    room: int         # story tokens the question window reserves
    head_len: int     # context tokens that go to the priming chunks
    n_prime: int
    final_len: int
    tail_cut: int     # front of (tail + suffix) dropped by the hard cap

    @property
    def n_tail(self) -> int:
        return self.n_ctx - self.head_len

    @property
    def story_in_final(self) -> int:
        """Context tokens actually present in the question window."""
        return self.n_tail - min(self.tail_cut, self.n_tail)


def chunk_geometry(n_ctx: int, n_sfx: int, seq_len: int,
                   max_new_tokens: int, num_chunks: int) -> Geometry:
    win = seq_len
    if num_chunks > 0:
        total = n_ctx + n_sfx + max_new_tokens
        win = min(seq_len, max(-(-total // num_chunks),
                               n_sfx + max_new_tokens + 1))
    room = max(win - n_sfx - max_new_tokens, 0)
    head_len = n_ctx - room if n_ctx > room else 0
    n_prime = -(-head_len // win) if head_len > 0 else 0
    final_len = max(win - max_new_tokens, 1)
    tail_cut = max(0, (n_ctx - head_len) + n_sfx - final_len)
    return Geometry(n_ctx, n_sfx, win, room, head_len, n_prime, final_len,
                    tail_cut)


def token_chunk(t: int, g: Geometry) -> int:
    """Chunk index holding context token t; n_prime = the question window;
    -1 = cut by the final-window cap and seen by no chunk."""
    if t < g.head_len:
        return t // g.win
    if t < g.head_len + min(g.tail_cut, g.n_tail):
        return -1
    return g.n_prime


def verify_geometry(g: Geometry, prime_chunks, final_ids, ctx_ids,
                    sfx_ids) -> None:
    """The token-to-chunk map must describe EXACTLY what split_context built.
    A drift here would put every fact in the wrong chunk, silently."""
    if len(prime_chunks) != g.n_prime:
        raise RuntimeError(f"CHUNKING DRIFT: split_context made "
                           f"{len(prime_chunks)} priming chunks, geometry says "
                           f"{g.n_prime}")
    flat = [int(t) for c in prime_chunks for t in c[0].tolist()]
    if flat != list(ctx_ids[:g.head_len]):
        raise RuntimeError("CHUNKING DRIFT: priming chunks != context head")
    for k, c in enumerate(prime_chunks):
        want = min(g.win, g.head_len - k * g.win)
        if c.shape[1] != want:
            raise RuntimeError(f"CHUNKING DRIFT: chunk {k} has {c.shape[1]} "
                               f"tokens, geometry says {want}")
    want_final = (list(ctx_ids[g.head_len:]) + list(sfx_ids))[-g.final_len:]
    if final_ids[0].tolist() != want_final:
        raise RuntimeError("CHUNKING DRIFT: question window != tail + suffix")


def char_span_to_tokens(offsets, start: int, end: int):
    """(first, last) token index overlapping chars [start, end), or None."""
    starts = [a for a, _ in offsets]
    ends = [b for _, b in offsets]
    i0 = bisect.bisect_right(ends, start)
    i1 = bisect.bisect_left(starts, end) - 1
    toks = [i for i in range(max(i0, 0), min(i1, len(offsets) - 1) + 1)
            if offsets[i][0] < end and offsets[i][1] > start]
    if not toks:   # non-monotone offsets (rare byte-level cases): full scan
        toks = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not toks:
        return None
    return toks[0], toks[-1]


def distance_bucket(d: Optional[int]) -> Optional[str]:
    if d is None:
        return None
    if d < 0:
        raise ValueError(f"negative distance {d}")
    if d == 0:
        return "0"
    if d == 1:
        return "1"
    if d <= 3:
        return "2-3"
    if d <= 7:
        return "4-7"
    return "8+"


def bucket_members(bucket: str, dist_bucket: Optional[str]) -> bool:
    if dist_bucket is None:
        return False
    if bucket == "all":
        return True
    if bucket == "1+":
        return dist_bucket != "0"
    return bucket == dist_bucket


# ---------------------------------------------------------------------------
# Pure: fact insertion
# ---------------------------------------------------------------------------

def insert_facts(story: list, protected: list, fact_ids: list):
    """Delete the first len(fact_ids) UNPROTECTED tokens of `story` and append
    `fact_ids`, so the length (and every position after the story) is fixed.

    Returns (new_story, deleted_positions), or None when the story holds fewer
    unprotected tokens than the fact needs.
    """
    if len(story) != len(protected):
        raise ValueError("story and protected mask differ in length")
    n = len(fact_ids)
    drop = []
    for i, p in enumerate(protected):
        if len(drop) == n:
            break
        if not p:
            drop.append(i)
    if len(drop) < n:
        return None
    dropset = set(drop)
    out = [t for i, t in enumerate(story) if i not in dropset] + list(fact_ids)
    if len(out) != len(story):
        raise AssertionError("insert_facts changed the window length")
    return out, drop


# ---------------------------------------------------------------------------
# Pure: scoring
# ---------------------------------------------------------------------------

def score_ranked(cands, nll_sum, n_tok, gold, nll_nocontext=None) -> dict:
    """Gold NLL / rank / top-1 (and their PMI versions) from candidate NLLs.

    top-1 is the argmin of the summed NLL (first index on ties), the same rule
    as tools/analyze_longcontext_pairs.scored_correct('rank').  rank counts
    candidates STRICTLY better than gold, +1.
    """
    norm = [str(c).strip().lower() for c in cands]
    g = norm.index(str(gold).strip().lower())   # ValueError = gold not a cand

    def _rank(score):
        pick = min(range(len(score)), key=lambda i: score[i])
        return pick, 1 + sum(1 for v in score if v < score[g])

    pick, rank = _rank(list(nll_sum))
    out = {"gold_nll": float(nll_sum[g]) / max(int(n_tok[g]), 1),
           "gold_nll_sum": float(nll_sum[g]), "rank": rank,
           "top1": bool(pick == g), "pick": cands[pick]}
    if nll_nocontext is not None:
        pmi = [a - b for a, b in zip(nll_sum, nll_nocontext)]
        ppick, prank = _rank(pmi)
        out.update({"pmi_rank": prank, "pmi_top1": bool(ppick == g),
                    "pmi_pick": cands[ppick]})
    return out


# ---------------------------------------------------------------------------
# Pure: donors
# ---------------------------------------------------------------------------

def donor_tier(item_src: int, item_rows: int, donor_src: int,
               donor_rows: int, lap: int) -> Optional[int]:
    """0 = same source chunk index; 1 = same carried row count AND same ring
    phase; None = not register-matched."""
    if item_rows != donor_rows:
        return None
    if item_src == donor_src:
        return 0
    if item_src % lap == donor_src % lap:
        return 1
    return None


def donor_compatible(task: str, item_id: str, item_gold: str,
                     item_entity: str, donor_id: str, donor_gold: str,
                     donor_facts_upto_src: list) -> bool:
    """The donor must not encode this item's answer to this item's question."""
    if donor_id == item_id:
        return False
    if donor_gold.strip().lower() == item_gold.strip().lower():
        return False
    ans, _ = world_answer(task, donor_facts_upto_src, item_entity)
    return ans is None or ans.lower() != item_gold.strip().lower()


def pick_donor(item: dict, pool: list, lap: int, rng: random.Random):
    """Best-tier compatible donor from `pool` (dicts with id, gold, src, rows,
    facts_upto_src), chosen with `rng` inside the tier.  -> (donor, tier)."""
    best, tier_best = [], None
    for d in pool:
        tier = donor_tier(item["src"], item["rows"], d["src"], d["rows"], lap)
        if tier is None:
            continue
        if not donor_compatible(item["task"], item["id"], item["gold"],
                                item["entity"], d["id"], d["gold"],
                                d["facts_upto_src"]):
            continue
        if tier_best is None or tier < tier_best:
            best, tier_best = [d], tier
        elif tier == tier_best:
            best.append(d)
    if not best:
        return None, None
    best.sort(key=lambda d: d["id"])
    return rng.choice(best), tier_best


# ---------------------------------------------------------------------------
# Pure: aggregation
# ---------------------------------------------------------------------------

def boot_ci(values, n_boot: int, seed: int):
    """(mean, lo, hi) percentile bootstrap of the mean; lo/hi None if n < 2."""
    import numpy as np
    n = len(values)
    if n == 0:
        return None, None, None
    v = np.asarray(values, dtype=np.float64)
    mean = float(v.mean())
    if n < 2 or n_boot <= 0:
        return mean, None, None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return mean, float(lo), float(hi)


def paired_deltas(records: list, a: str, b: str, row_filter=None) -> dict:
    """Per-item A - B on gold NLL, top-1 and PMI top-1, over items scored in
    BOTH conditions.  Sign: dNLL < 0 and dTop1 > 0 both mean A is better."""
    d_nll, d_top, d_pmi, ids = [], [], [], []
    for r in records:
        ca, cb = r["conditions"].get(a), r["conditions"].get(b)
        if ca is None or cb is None:
            continue
        if row_filter == "equal_rows":
            rows = r.get("state_rows") or {}
            if rows.get(a) != rows.get(b):
                continue
        elif row_filter == "fiw_matched":
            if not (r.get("fiw") or {}).get("length_matched"):
                continue
        elif row_filter is not None:
            raise ValueError(f"unknown row filter {row_filter!r}")
        ids.append(r["id"])
        d_nll.append(ca["gold_nll"] - cb["gold_nll"])
        d_top.append(float(ca["top1"]) - float(cb["top1"]))
        if "pmi_top1" in ca and "pmi_top1" in cb:
            d_pmi.append(float(ca["pmi_top1"]) - float(cb["pmi_top1"]))
    return {"ids": ids, "nll": d_nll, "top1": d_top, "pmi_top1": d_pmi}


def summarize_condition(records: list, cond: str) -> dict:
    rows = [r["conditions"][cond] for r in records
            if r["conditions"].get(cond) is not None]
    n = len(rows)
    if n == 0:
        return {"n": 0}
    out = {"n": n,
           "gold_nll": sum(x["gold_nll"] for x in rows) / n,
           "top1": sum(x["top1"] for x in rows) / n,
           "rank": sum(x["rank"] for x in rows) / n}
    pm = [x["pmi_top1"] for x in rows if "pmi_top1" in x]
    if pm:
        out["pmi_top1"] = sum(pm) / len(pm)
    return out


def aggregate(records: list, conditions, n_boot: int, seed: int,
              group_key=lambda r: r["task"]) -> dict:
    """{group: {bucket: {"conditions": {...}, "contrasts": {...}}}}."""
    groups = {}
    for r in records:
        groups.setdefault(group_key(r), []).append(r)
    out = {}
    for gk in sorted(groups):
        out[gk] = {}
        for b in AGG_BUCKETS:
            sub = [r for r in groups[gk]
                   if bucket_members(b, r.get("dist_bucket"))]
            if not sub:
                continue
            cell = {"n_items": len(sub),
                    "conditions": {c: summarize_condition(sub, c)
                                   for c in conditions},
                    "contrasts": {}}
            for a, bb, label, filt in CONTRASTS:
                if a not in conditions or bb not in conditions:
                    continue
                d = paired_deltas(sub, a, bb, filt)
                if not d["ids"]:
                    continue
                s = zlib.crc32(f"{gk}|{b}|{label}".encode()) + seed
                cell["contrasts"][label] = {
                    "n": len(d["ids"]),
                    "d_nll": boot_ci(d["nll"], n_boot, s),
                    "d_top1": boot_ci(d["top1"], n_boot, s + 1),
                    "d_pmi_top1": boot_ci(d["pmi_top1"], n_boot, s + 2),
                }
            out[gk][b] = cell
    return out


# ---------------------------------------------------------------------------
# Item preparation (tokenizer only, no model)
# ---------------------------------------------------------------------------

def prepare_item(tok, task: str, cfg: str, row_idx: int, ex: dict,
                 seq_len: int, max_new_tokens: int, num_chunks: int,
                 locations) -> dict:
    """Everything about one item that does not need the model.  Raises
    Ambiguous for items whose supporting fact cannot be pinned down."""
    ctx = ex.get("input", ex.get("context", ex.get("text", "")))
    question = ex.get("question", "")
    answer = str(ex.get("target", ex.get("answer", "")))
    example_id = f"{task}/{cfg}/{row_idx}"
    suffix = build_suffix(task, question)
    prime_chunks, final_ids = split_context(tok, ctx, suffix, seq_len,
                                            max_new_tokens,
                                            num_chunks=num_chunks)
    enc = tok(ctx, add_special_tokens=False, return_offsets_mapping=True)
    ctx_ids, offsets = list(enc.input_ids), list(enc.offset_mapping)
    if ctx_ids != tok(ctx, add_special_tokens=False).input_ids:
        raise RuntimeError("tokenizing with offsets changed the ids")
    sfx_ids = tok(suffix, add_special_tokens=False).input_ids
    geom = chunk_geometry(len(ctx_ids), len(sfx_ids), seq_len,
                          max_new_tokens, num_chunks)
    verify_geometry(geom, prime_chunks, final_ids, ctx_ids, sfx_ids)

    final_text = tok.decode(final_ids[0], skip_special_tokens=True)
    base = {"id": example_id, "task": task, "length": cfg, "row": row_idx,
            "gold": answer.strip(), "question": question.strip(),
            "n_prime_chunks": geom.n_prime, "window": geom.win,
            "n_ctx_tokens": geom.n_ctx, "n_sfx_tokens": geom.n_sfx,
            "story_tokens_in_window": geom.story_in_final,
            # the benchmark's own stratum, and why it misleads: the demos in
            # the suffix name rooms too
            "gold_in_final_window": bool(contains_answer(final_text, answer)),
            "gold_in_suffix": bool(contains_answer(suffix, answer))}

    entity = parse_question(task, question)
    facts = parse_facts(ctx, locations)
    if not facts:
        raise Ambiguous("no_facts_parsed")
    ans, support = world_answer(task, facts, entity)
    if ans is None:
        raise Ambiguous("no_supporting_fact", entity)
    if ans.lower() != answer.strip().lower():
        raise Ambiguous("answer_mismatch", f"replay={ans} target={answer}")

    spans, chunks = [], []
    for f in facts:
        sp = char_span_to_tokens(offsets, f.start, f.end)
        if sp is None:
            raise Ambiguous("fact_not_tokenized", f.text)
        spans.append(sp)
        chunks.append((token_chunk(sp[0], geom), token_chunk(sp[1], geom)))
    for i in support:
        c0, c1 = chunks[i]
        if c0 == -1 or c1 == -1:
            raise Ambiguous("fact_unseen", facts[i].text)
        if c0 != c1:
            raise Ambiguous("fact_straddles_chunk", facts[i].text)

    sup_chunks = [chunks[i][0] for i in support]
    carried = [c for c in sup_chunks if c < geom.n_prime]
    src = max(carried) if carried else None
    distance = geom.n_prime - src if src is not None else 0

    # fact_in_window: all supporting facts, story order, at the story's end.
    fiw_ids, fiw = None, {"applied": False}
    if distance >= 1:
        n_story = geom.story_in_final
        story = final_ids[0, :n_story].tolist()
        first_ctx_tok = geom.n_ctx - n_story      # context index of story[0]
        protected = [False] * n_story
        for (t0, t1), (c0, c1) in zip(spans, chunks):
            if c1 != geom.n_prime and c0 != geom.n_prime:
                continue
            for t in range(max(t0, first_ctx_tok), t1 + 1):
                protected[t - first_ctx_tok] = True
        fact_text = " " + " ".join(facts[i].text for i in support)
        fact_ids = tok(fact_text, add_special_tokens=False).input_ids
        ins = insert_facts(story, protected, fact_ids)
        rest = final_ids[0, n_story:].tolist()
        if ins is None:
            # NO FILLER TO TRADE.  At 1k the num_chunks=8 sizing leaves a
            # ~98-token window with 1-2 story tokens, so a length-matched
            # insertion is impossible.  Rather than lose the ceiling for the
            # whole bucket, EXTEND the window by the fact's length -- and say
            # so: length_matched=False, and the aggregates carry a separate
            # [length-matched] contrast that excludes these items.
            new = torch.tensor(story + list(fact_ids) + rest,
                               dtype=torch.long).unsqueeze(0)
            if new.shape[1] != final_ids.shape[1] + len(fact_ids):
                raise AssertionError("extended fact_in_window has the wrong "
                                     "length")
            fiw = {"applied": True, "length_matched": False,
                   "n_tokens": len(fact_ids), "extended_by": len(fact_ids),
                   "filler": n_story - sum(protected),
                   "inserted": fact_text.strip(), "deleted": ""}
        else:
            new_story, dropped = ins
            new = torch.tensor(new_story + rest,
                               dtype=torch.long).unsqueeze(0)
            if new.shape != final_ids.shape:
                raise AssertionError("fact_in_window changed the length")
            fiw = {"applied": True, "length_matched": True,
                   "n_tokens": len(fact_ids), "extended_by": 0,
                   "inserted": fact_text.strip(),
                   "deleted": tok.decode([story[i] for i in dropped])}
        if rest and new[0, -len(rest):].tolist() != rest:
            raise AssertionError("fact_in_window altered the suffix tokens")
        fiw_ids = new

    return {**base, "entity": entity, "suffix": suffix,
            "prime_chunks": prime_chunks, "final_ids": final_ids,
            "fiw_ids": fiw_ids, "fiw": fiw, "facts": facts,
            "fact_chunks": chunks,
            "support": [{"text": facts[i].text, "kind": facts[i].kind,
                         "chunk": chunks[i][0], "tokens": list(spans[i])}
                        for i in support],
            "support_chunks": sup_chunks, "src_chunk": src,
            "distance": distance, "dist_bucket": distance_bucket(distance),
            "support_in_window": sum(1 for c in sup_chunks
                                     if c == geom.n_prime),
            "n_facts": len(facts)}


def facts_upto(item: dict, k: int) -> list:
    """Facts any part of which is in chunks <= k (conservative: a partly seen
    fact counts as seen for the donor-compatibility check)."""
    return [f for f, (c0, _c1) in zip(item["facts"], item["fact_chunks"])
            if 0 <= c0 <= k]


# ---------------------------------------------------------------------------
# Model side
# ---------------------------------------------------------------------------

def _get_m_cross(out):
    return out.get("m_cross") if isinstance(out, dict) \
        else getattr(out, "m_cross", None)


@torch.no_grad()
def run_chain(model, prime_chunks, num_steps, device, buf=None) -> list:
    """prime_cross_state, keeping every intermediate state.  states[k] is the
    carry after chunk k -- what chunk k+1 would read.

    RING-POINTER HAZARD.  The gated ring's write pointer (`buf._chunk`,
    `_z_chunk`, `_z_pending`) is mutable state ON THE BUFFER: a forward with
    m_cross_in=None resets it, every merge advances it.  A side forward
    interleaved into a chain would make the chain's later writes land in the
    wrong rows, silently.  This tool never does that -- each chain runs start
    to finish here, and every other forward is a scoring forward
    (prefix_write=False, which never calls merge) issued after the chain is
    done.  The pointer is CHECKED rather than trusted: after n chunks it must
    read n."""
    states, m = [], None
    for ch in prime_chunks:
        out = model(input_ids=ch.to(device), num_steps=num_steps,
                    m_cross_in=m, return_m_cross=True)
        m = _get_m_cross(out)
        if m is None:
            raise RuntimeError("a carry arm returned no m_cross")
        states.append(m)
    if buf is not None and hasattr(buf, "_chunk") and prime_chunks and \
            int(buf._chunk) != len(prime_chunks):
        raise RuntimeError(
            f"ring pointer reads {buf._chunk} after {len(prime_chunks)} "
            f"chained chunks: something forwarded inside the chain, so its "
            f"writes landed in the wrong ring rows")
    return states


def state_stats(state, hidden: int) -> Optional[dict]:
    if state is None:
        return None
    s = state.detach().float()
    e = s[..., :hidden]
    out = {"rows": int(s.shape[1]),
           "e_norm": float(e.norm(dim=-1).mean())}
    if s.shape[-1] > hidden:
        out["z_norm"] = float(s[..., hidden:].norm(dim=-1).mean())
    return out


def score_condition(model, tok, prompt_ids, cands, num_steps, m_cross,
                    example_id) -> list:
    seed_example(example_id)          # the benchmark's own scoring seed
    ranked = rank_candidates(model, tok, prompt_ids, cands, num_steps,
                             m_cross=m_cross)
    return ranked


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def load_bench_records(path: Optional[str]) -> dict:
    if not path:
        return {}
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.rglob("records.jsonl"))
    if not files:
        raise SystemExit(f"--records {path}: no records.jsonl found")
    recs = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    recs[r["id"]] = r
    print(f"[records] {len(recs):,} benchmark records from "
          f"{', '.join(str(f) for f in files)}")
    return recs


def check_structure(item: dict, bench: Optional[dict], cands) -> None:
    """Fail FAST if this item's chunking is not the benchmark's."""
    if bench is None:
        return
    bad = []
    if int(bench.get("n_prime_chunks", -1)) != item["n_prime_chunks"]:
        bad.append(f"n_prime_chunks bench={bench.get('n_prime_chunks')} "
                   f"here={item['n_prime_chunks']}")
    if "gold_in_final_window" in bench and \
            bool(bench["gold_in_final_window"]) != item["gold_in_final_window"]:
        bad.append("gold_in_final_window differs")
    if bench.get("candidates") and list(bench["candidates"]) != list(cands):
        bad.append(f"candidate set differs: {bench['candidates']} vs {cands}")
    if str(bench.get("gold", "")).strip() != item["gold"]:
        bad.append("gold differs")
    if bad:
        raise SystemExit(
            f"REFUSING: {item['id']} does not match the benchmark record -- "
            f"{'; '.join(bad)}.  The dissection's `normal` would not be the "
            f"benchmark's condition.  Check --seq_len/--num_chunks/"
            f"--max_new_tokens and that --dataset_path is the benchmark's "
            f"data (babilong-1k-samples).")


def reproduction_report(records: list, bench: dict, cands_by_cell: dict,
                        dtype: str) -> dict:
    """normal vs the benchmark's own scores on the shared ids."""
    pairs = [(r, bench[r["id"]]) for r in records
             if r["id"] in bench and r["conditions"].get("normal")]
    if not pairs:
        return {"n": 0}
    rep = {"n": len(pairs)}
    d_nc, d_cand, agree, agree_pmi, top_b, top_o = [], [], 0, 0, 0, 0
    pmi_b, pmi_o, n_pmi = 0, 0, 0
    for r, b in pairs:
        nrm = r["conditions"]["normal"]
        if b.get("cand_nll_nocontext") and r.get("cand_nll_nocontext"):
            d_nc += [abs(x - y) for x, y in zip(r["cand_nll_nocontext"],
                                                b["cand_nll_nocontext"])]
        if b.get("cand_nll"):
            d_cand += [abs(x - y) for x, y in zip(nrm["cand_nll"],
                                                  b["cand_nll"])]
            sb = score_ranked(b["candidates"], b["cand_nll"],
                              b.get("cand_ntok") or [1] * len(b["cand_nll"]),
                              b["gold"], b.get("cand_nll_nocontext"))
            agree += int(sb["pick"] == nrm["pick"])
            top_b += int(sb["top1"])
            top_o += int(nrm["top1"])
            if "pmi_pick" in sb and "pmi_pick" in nrm:
                n_pmi += 1
                agree_pmi += int(sb["pmi_pick"] == nrm["pmi_pick"])
                pmi_b += int(sb["pmi_top1"])
                pmi_o += int(nrm["pmi_top1"])
    n = len(pairs)
    rep.update({
        "nocontext_mean_abs_diff": (sum(d_nc) / len(d_nc)) if d_nc else None,
        "nocontext_max_abs_diff": max(d_nc) if d_nc else None,
        "normal_cand_mean_abs_diff": (sum(d_cand) / len(d_cand))
        if d_cand else None,
        "normal_pick_agreement": agree / n,
        "bench_top1": top_b / n, "ours_top1": top_o / n,
        "pmi_pick_agreement": (agree_pmi / n_pmi) if n_pmi else None,
        "bench_pmi_top1": (pmi_b / n_pmi) if n_pmi else None,
        "ours_pmi_top1": (pmi_o / n_pmi) if n_pmi else None,
        "this_dtype": dtype, "bench_dtype": "bfloat16 (eval_babilong default)",
    })
    return rep


def parse_lengths(items) -> list:
    out = []
    for x in items:
        s = str(x).strip().lower()
        if s.endswith("k"):
            out.append(s)
        else:
            out.append(f"{int(s) // 1000}k")
    return out


def chain_seed(seed: int, example_id: str) -> int:
    return (seed * 1_000_003 + zlib.crc32(example_id.encode())) & 0x7fffffff


def expected_znorm(checkpoint: Optional[str]) -> Optional[float]:
    if not checkpoint:
        return None
    # cortex-5b/cortex-final/checkpoint_38146 (cluster) or a slim copy named
    # cortex-final_38146 (local smoke)
    m = re.search(r"cortex-final(?:/+checkpoint_|_)0*(\d+)",
                  checkpoint.replace("\\", "/"))
    if not m:
        return None
    return KNOWN_ZNORM.get(int(m.group(1)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser("BABILong write/retention/read dissection")
    p.add_argument("--model_name", required=True,
                   help="HF dir (the BASE dir for an overlay checkpoint)")
    p.add_argument("--checkpoint", default=None,
                   help="train.py checkpoint dir or chkpt.pt, overlaid on "
                        "--model_name.  REQUIRED for a memory arm (without it "
                        "reset_cortex_graft_init wipes the trained cortex).")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="graft config override (repeatable), parsed by "
                        "parse_config_overrides -- as eval_babilong --set.")
    p.add_argument("--dataset_path", default=None,
                   help="local BABILong snapshot (cluster: data/babilong-1k); "
                        "layout auto-detected by eval_babilong.load_bucket")
    p.add_argument("--dataset_repo", default="RMT-team/babilong-1k-samples",
                   help="hub repo when --dataset_path is absent (the "
                        "benchmark's BABILONG_REPO)")
    p.add_argument("--tasks", nargs="+", default=["qa1", "qa2"],
                   choices=["qa1", "qa2"])
    p.add_argument("--lengths", nargs="+", default=["1k", "2k", "4k", "8k"])
    p.add_argument("--max_items", type=int, default=100,
                   help="items EVALUATED per task x length (ambiguous rows "
                        "are skipped and do not count)")
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--num_chunks", type=int, default=8,
                   help="the benchmark's NUM_CHUNKS; must match it")
    p.add_argument("--max_new_tokens", type=int, default=12,
                   help="eval_babilong's default; it sizes the window, so it "
                        "must match the benchmark too")
    p.add_argument("--conditions", nargs="+", default=["auto"],
                   choices=["auto", *CONDITIONS],
                   help="auto = all five on a memory arm, normal + "
                        "fact_in_window on a model without a prefix buffer")
    p.add_argument("--records", default=None,
                   help="the benchmark's records.jsonl (or its run dir): "
                        "chunking is checked per item (fail fast) and "
                        "`normal` is compared against it")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--allow_no_overlay", action="store_true")
    p.add_argument("--allow_key_mismatch", action="store_true")
    p.add_argument("--allow_znorm_mismatch", action="store_true")
    p.add_argument("--allow_scrambled", action="store_true")
    return p.parse_args(argv)


def refuse(msg: str) -> None:
    print(f"\nREFUSING: {msg}", flush=True)
    raise SystemExit(2)


def resolve_conditions(requested, has_buffer: bool) -> list:
    if requested == ["auto"]:
        return list(CONDITIONS) if has_buffer else list(CONTROL_CONDITIONS)
    if "auto" in requested:
        refuse("--conditions auto cannot be combined with named conditions")
    if not has_buffer:
        bad = [c for c in requested if c in CARRY_CONDITIONS]
        if bad:
            refuse(f"{bad} requested on a model with NO prefix buffer.  There "
                   f"is no carry to remove or teleport; on this model those "
                   f"conditions would silently equal `normal`.")
    if "teleport_donor" in requested and "teleport" not in requested:
        refuse("teleport_donor needs teleport (the donor is defined against it)")
    return [c for c in CONDITIONS if c in requested]


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt_ci(t, scale=1.0, nd=3):
    m, lo, hi = t
    if m is None:
        return "--"
    if lo is None:
        return f"{m * scale:+.{nd}f} [  n<2  ]"
    return f"{m * scale:+.{nd}f} [{lo * scale:+.{nd}f},{hi * scale:+.{nd}f}]"


def print_tables(agg: dict, conditions, chance: dict, title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    for gk, buckets in agg.items():
        ch = chance.get(str(gk).split("/")[0])
        print(f"\n--- {gk}   (chance top-1 = "
              f"{'%.1f%%' % (100 * ch) if ch else '?'}) ---")
        print(f"  {'dist':<5} {'condition':<16} {'n':>5} {'goldNLL':>8} "
              f"{'top1%':>7} {'pmi%':>7} {'rank':>6}")
        for b, cell in buckets.items():
            for c in conditions:
                s = cell["conditions"].get(c, {})
                if not s.get("n"):
                    continue
                pmi = (f"{100 * s['pmi_top1']:>7.1f}"
                       if "pmi_top1" in s else f"{'--':>7}")
                print(f"  {b:<5} {c:<16} {s['n']:>5} {s['gold_nll']:>8.3f} "
                      f"{100 * s['top1']:>7.1f} {pmi} {s['rank']:>6.2f}")
        print(f"\n  paired contrasts (A - B; dNLL<0 / dTop1>0 = A better), "
              f"95% bootstrap CI")
        print(f"  {'dist':<5} {'contrast':<32} {'n':>5} "
              f"{'dNLL nats/tok':>24} {'dTop1 pt':>22} {'dPMI-top1 pt':>22}")
        for b, cell in buckets.items():
            for label, c in cell["contrasts"].items():
                print(f"  {b:<5} {label:<32} {c['n']:>5} "
                      f"{_fmt_ci(c['d_nll']):>24} "
                      f"{_fmt_ci(c['d_top1'], 100, 1):>22} "
                      f"{_fmt_ci(c['d_pmi_top1'], 100, 1):>22}")


HOW_TO_READ = """
HOW TO READ  (delta = A - B; dNLL < 0 and dTop1 > 0 both mean A is better)
  1. fact_in_window top-1 vs chance: ~chance => the CAPABILITY FLOOR binds --
     the model cannot answer even with the fact in its window, and no memory
     can win this task at this model scale.  Stop here if so.
  2. teleport - teleport_donor: ~0 => the carried CONTENT is not used (WRITE
     or READ failure: the fact never entered the carry, or the question chunk
     cannot use it).  dNLL < 0 => information transfers when retention is
     removed from the path.
  3. teleport - normal: dNLL << 0 (teleport >> normal) => RETENTION loss: the
     fact was written, then gated away over the intervening chunks.  Exactly 0
     at dist 1 by construction.  Use the [equal rows] line when the source
     chunk sits inside the first lap (gate_fill=grow: fewer carried columns).
  4. normal - nocarry: the carry's total effect, content AND columns -- not a
     content effect (that is 2).
  dist = question chunk - latest carried fact chunk; dist 0 = every
  supporting fact already in the question window.  laps = dist / 4."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    if args.dtype == "bfloat16":
        print("WARNING: bfloat16.  Every contrast here is a difference of two "
              "nearly equal losses -- the quantity bf16 got wrong by 6x in "
              "P0.1.  Smoke only; do not quote these numbers.", flush=True)
    lengths = parse_lengths(args.lengths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    overrides = parse_config_overrides(args.set)
    num_steps = to_num_steps(args.T)

    # ---- PREFLIGHT every cell before the model loads -----------------------
    # A missing (task, length) file used to surface only when the loop reached
    # it, after every earlier cell had spent its GPU time.  Resolve them all
    # now: seconds, not hours.
    cands_by_cell = {}
    for task in args.tasks:
        for cfg_name in lengths:
            c = collect_candidates(load_bucket(task, cfg_name,
                                               args.dataset_path,
                                               args.dataset_repo))
            if not c:
                refuse(f"{task}/{cfg_name}: empty candidate set")
            cands_by_cell[f"{task}/{cfg_name}"] = c
    print(f"[data] preflight ok: {len(cands_by_cell)} cells "
          f"({', '.join(cands_by_cell)})", flush=True)

    # ---- load, capturing the overlay's key counts --------------------------
    cap = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, cap)):
        model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                     dtype, device,
                                     config_overrides=overrides or None)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    buf = getattr(cortex, "prefix", None) if cortex is not None else None
    m = re.search(r"\[load\] overlaid .*?: (\d+) missing / (\d+) unexpected",
                  cap.getvalue())
    key_counts = None
    if args.checkpoint:
        if not m:
            refuse("could not find the '[load] overlaid ...' line, so the "
                   "overlay cannot be confirmed.")
        key_counts = {"missing": int(m.group(1)),
                      "unexpected": int(m.group(2))}
        if (key_counts["missing"] or key_counts["unexpected"]) and \
                not args.allow_key_mismatch:
            refuse(f"overlay reported {key_counts['missing']} missing / "
                   f"{key_counts['unexpected']} unexpected keys.  Unexpected "
                   f"keys on a memory checkpoint mean the graft was built "
                   f"WITHOUT the arm's geometry (missing --set) and the "
                   f"trained cortex was dropped; missing keys mean part of the "
                   f"model is at init.  --allow_key_mismatch to override.")
    elif cortex is not None and not args.allow_no_overlay:
        refuse("a memory model with NO --checkpoint overlay: "
               "reset_cortex_graft_init has just wiped the trained cortex "
               "(memory: hf-dir-evals-wipe-trained-cortex).  Pass the "
               "train.py checkpoint, or --allow_no_overlay knowingly.")
    if overrides.get("use_memory") and buf is None:
        refuse("use_memory was forced but no prefix buffer was built: "
               + explain_missing_cortex(cfg, overrides))
    if cortex is not None:
        from cortex_memory.health import refuse_if_scrambled
        refuse_if_scrambled(cortex, "babilong_dissect", args.allow_scrambled)
    znorm = None
    if cortex is not None and getattr(cortex, "latent_carry", False):
        znorm = float(getattr(cortex, "latent_read_znorm_target", 0.0))
        if not hasattr(cfg, "latent_read_znorm_target"):
            refuse("latent_carry is on but latent_read_znorm_target was never "
                   "set, so the graft fell back to its CODE DEFAULT "
                   f"({znorm}).  Pass --set latent_read_znorm_target=<the "
                   "lineage's measured value>.")
        want = expected_znorm(args.checkpoint)
        if want is not None and abs(want - znorm) > 1e-6 and \
                not args.allow_znorm_mismatch:
            refuse(f"latent_read_znorm_target={znorm} but this cortex-final "
                   f"checkpoint trained with {want} (lineage-specific; see "
                   f"KNOWN_ZNORM).  --allow_znorm_mismatch to override.")
        print(f"[z] latent_read_znorm_target = {znorm}"
              + (f" (expected {want} for this step: OK)" if want else
                 " (step not in KNOWN_ZNORM: verify by hand)"))
    conditions = resolve_conditions(args.conditions, buf is not None)
    hidden = int(getattr(buf, "hidden_size", getattr(cfg, "n_embd", 0)))
    lap = (int(buf.n_slots) // max(int(buf.n_vec), 1)
           if buf is not None and getattr(buf, "route", "") == "ring"
           else (1 if buf is not None else LAP_DEFAULT))
    arm = "memory" if buf is not None else "no-memory"
    print(f"[arm] {arm}; conditions = {conditions}; T={args.T} "
          f"seq_len={args.seq_len} num_chunks={args.num_chunks} "
          f"max_new_tokens={args.max_new_tokens}; lap = {lap} chunks"
          + (f"; buffer {type(buf).__name__} n_vec={buf.n_vec} "
             f"n_slots={getattr(buf, 'n_slots', '?')} "
             f"fill={getattr(buf, 'fill', '?')}" if buf is not None else ""),
          flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name)
    bench = load_bench_records(args.records)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_fh = open(out_dir / "items.jsonl", "w", encoding="utf-8")

    # seconds per item, per length (window size and chain depth vary by
    # length, so a pooled mean would not extrapolate to a different mix)
    records, skipped, timing_by_len = [], {}, {}
    chance = {}
    self_check = {"teleport_d1_max_abs": 0.0, "teleport_d1_n": 0,
                  "normal_vs_nocarry_max_abs": 0.0, "nocarry_n": 0,
                  "no_donor": 0, "donor_tiers": {}}

    for task in args.tasks:
        for cfg_name in lengths:
            cell = f"{task}/{cfg_name}"
            cands = cands_by_cell[cell]          # resolved in the preflight
            chance[task] = 1.0 / len(cands)
            locs = tuple(c.lower() for c in cands)
            print(f"\n--- {cell}: {len(cands)} candidates "
                  f"(chance {100.0 / len(cands):.1f}%): {', '.join(cands)}",
                  flush=True)
            cell_items, pool = [], []
            sk = skipped.setdefault(cell, {})
            timing = timing_by_len.setdefault(
                cfg_name, {c: [] for c in ("chain", "pmi_nocontext",
                                           *conditions)})
            n_rows = 0
            for row_idx, ex in enumerate(load_bucket(task, cfg_name,
                                                     args.dataset_path,
                                                     args.dataset_repo)):
                if len(cell_items) >= args.max_items:
                    break
                ctx = ex.get("input", ex.get("context", ex.get("text", "")))
                if not ctx or not ex.get("question") or not str(
                        ex.get("target", ex.get("answer", ""))):
                    continue      # the benchmark skips these rows too
                n_rows += 1
                try:
                    item = prepare_item(tok, task, cfg_name, row_idx, ex,
                                        args.seq_len, args.max_new_tokens,
                                        args.num_chunks, locs)
                except Ambiguous as e:
                    sk.setdefault(e.reason, []).append(
                        f"{task}/{cfg_name}/{row_idx}")
                    continue
                check_structure(item, bench.get(item["id"]) if bench else None,
                                cands)
                iid = item["id"]
                res = {c: None for c in conditions}
                rows = {}

                # ---- chain --------------------------------------------------
                _sync(); t0 = time.perf_counter()
                states = []
                if buf is not None:
                    torch.manual_seed(chain_seed(args.seed, iid))
                    states = run_chain(model, item["prime_chunks"], num_steps,
                                       device, buf)
                _sync(); timing["chain"].append(time.perf_counter() - t0)
                normal_state = states[-1] if states else None

                # ---- PMI denominator (condition-invariant) ------------------
                _sync(); t0 = time.perf_counter()
                nc_ids = torch.tensor(
                    tok(item["suffix"], add_special_tokens=False).input_ids,
                    dtype=torch.long).unsqueeze(0)
                nc = score_condition(model, tok, nc_ids, cands, num_steps,
                                     None, iid)
                nc_nll = [c["nll_sum"] for c in nc]
                _sync(); timing["pmi_nocontext"].append(
                    time.perf_counter() - t0)

                def run(cond, prompt, state):
                    _sync(); t1 = time.perf_counter()
                    ranked = score_condition(model, tok, prompt, cands,
                                             num_steps, state, iid)
                    _sync(); timing[cond].append(time.perf_counter() - t1)
                    nll = [c["nll_sum"] for c in ranked]
                    ntok = [c["n_tok"] for c in ranked]
                    sc = score_ranked(cands, nll, ntok, item["gold"], nc_nll)
                    sc["cand_nll"] = nll
                    sc["cand_ntok"] = ntok
                    return sc

                fin = item["final_ids"]
                res["normal"] = run("normal", fin, normal_state)
                rows["normal"] = int(normal_state.shape[1]) \
                    if normal_state is not None else 0
                stats = {"normal": state_stats(normal_state, hidden)}
                if "nocarry" in conditions:
                    res["nocarry"] = run("nocarry", fin, None)
                    rows["nocarry"] = 0
                    if normal_state is not None:
                        dmax = max(abs(a - b) for a, b in zip(
                            res["normal"]["cand_nll"],
                            res["nocarry"]["cand_nll"]))
                        self_check["normal_vs_nocarry_max_abs"] = max(
                            self_check["normal_vs_nocarry_max_abs"], dmax)
                        self_check["nocarry_n"] += 1
                if "teleport" in conditions and item["distance"] >= 1:
                    tstate = states[item["src_chunk"]]
                    res["teleport"] = run("teleport", fin, tstate)
                    rows["teleport"] = int(tstate.shape[1])
                    stats["teleport"] = state_stats(tstate, hidden)
                    if item["distance"] == 1:
                        d1 = max(abs(a - b) for a, b in zip(
                            res["normal"]["cand_nll"],
                            res["teleport"]["cand_nll"]))
                        self_check["teleport_d1_max_abs"] = max(
                            self_check["teleport_d1_max_abs"], d1)
                        self_check["teleport_d1_n"] += 1
                    if "teleport_donor" in conditions:
                        pool.append({"id": iid, "gold": item["gold"],
                                     "src": item["src_chunk"],
                                     "rows": int(tstate.shape[1]),
                                     "facts_upto_src": facts_upto(
                                         item, item["src_chunk"]),
                                     "state": tstate.detach().to("cpu")})
                if "fact_in_window" in conditions and item["fiw_ids"] is not None:
                    res["fact_in_window"] = run("fact_in_window",
                                                item["fiw_ids"], normal_state)
                    rows["fact_in_window"] = rows["normal"]
                del states, normal_state

                rec = {k: v for k, v in item.items()
                       if k not in ("prime_chunks", "final_ids", "fiw_ids",
                                    "facts", "fact_chunks", "suffix")}
                rec.update({"laps": item["distance"] / lap, "lap_chunks": lap,
                            "candidates": cands, "cand_nll_nocontext": nc_nll,
                            "conditions": res, "state_rows": rows,
                            "state_stats": stats})
                b = bench.get(iid) if bench else None
                if b is not None:
                    rec["bench"] = {k: b.get(k) for k in
                                    ("correct", "cand_nll",
                                     "cand_nll_nocontext", "gold_nll_per_tok")}
                cell_items.append((item, rec))
                if len(cell_items) % 10 == 0:
                    print(f"  [{cell}] {len(cell_items)} items "
                          f"({time.time() - t_start:.0f}s elapsed)",
                          flush=True)

            # ---- phase B: donors -------------------------------------------
            if "teleport_donor" in conditions:
                for item, rec in cell_items:
                    if item["distance"] < 1:
                        continue
                    me = next(p for p in pool if p["id"] == item["id"])
                    rng = random.Random(chain_seed(args.seed + 17, item["id"]))
                    donor, tier = pick_donor(
                        {"id": item["id"], "task": task, "gold": item["gold"],
                         "entity": item["entity"], "src": me["src"],
                         "rows": me["rows"]}, pool, lap, rng)
                    if donor is None:
                        self_check["no_donor"] += 1
                        rec["donor"] = None
                        continue
                    dstate = donor["state"].to(device=device, dtype=dtype)
                    if dstate.shape != me["state"].shape:
                        raise AssertionError(
                            f"donor state {tuple(dstate.shape)} != teleport "
                            f"state {tuple(me['state'].shape)} for "
                            f"{item['id']}")
                    _sync(); t1 = time.perf_counter()
                    ranked = score_condition(model, tok, item["final_ids"],
                                             cands, num_steps, dstate,
                                             item["id"])
                    _sync(); timing["teleport_donor"].append(
                        time.perf_counter() - t1)
                    nll = [c["nll_sum"] for c in ranked]
                    ntok = [c["n_tok"] for c in ranked]
                    sc = score_ranked(cands, nll, ntok, item["gold"],
                                      rec["cand_nll_nocontext"])
                    sc["cand_nll"], sc["cand_ntok"] = nll, ntok
                    rec["conditions"]["teleport_donor"] = sc
                    rec["state_rows"]["teleport_donor"] = int(dstate.shape[1])
                    rec["state_stats"]["teleport_donor"] = state_stats(
                        dstate, hidden)
                    rec["donor"] = {"id": donor["id"], "gold": donor["gold"],
                                    "src_chunk": donor["src"], "tier": tier}
                    self_check["donor_tiers"][str(tier)] = \
                        self_check["donor_tiers"].get(str(tier), 0) + 1
                pool.clear()

            for _item, rec in cell_items:
                items_fh.write(json.dumps(rec) + "\n")
                records.append(rec)
            items_fh.flush()
            n_skip = sum(len(v) for v in sk.values())
            n_parse = sum(len(v) for k, v in sk.items() if k in PARSER_SKIPS)
            print(f"  [{cell}] {len(cell_items)} evaluated, {n_skip} skipped "
                  f"of {n_rows} rows read: "
                  + (", ".join(f"{k}={len(v)}" for k, v in sk.items())
                     or "none"), flush=True)
            if n_rows and n_parse / n_rows > 0.05:
                print(f"  *** WARNING: {cell}: {100 * n_parse / n_rows:.0f}% of "
                      f"rows failed the PARSE/REPLAY ({n_parse}) -- check the "
                      f"fact parser before trusting this cell.", flush=True)
    items_fh.close()

    if not records:
        print("ERROR: 0 items evaluated.  Check --dataset_path / --tasks / "
              "--lengths.")
        return 1

    # ---- aggregates ---------------------------------------------------------
    agg_task = aggregate(records, conditions, args.boot, args.seed)
    agg_len = aggregate(records, conditions, args.boot, args.seed,
                        group_key=lambda r: f"{r['task']}/{r['length']}")
    # only the pooled buckets for the per-length view
    agg_len = {k: {b: v for b, v in d.items() if b in ("1+", "all")}
               for k, d in agg_len.items()}

    normal_nll = [r["conditions"]["normal"]["gold_nll"] for r in records]
    mean_nll = sum(normal_nll) / len(normal_nll)
    ln_v = math.log(max(vocab, 2))
    health = {"mean_gold_nll_normal": mean_nll, "ln_vocab": ln_v,
              "margin_below_chance": ln_v - mean_nll,
              "at_chance": bool(ln_v - mean_nll < 1.0)}
    repro = reproduction_report(records, bench, cands_by_cell, args.dtype) \
        if bench else None
    tsum = {L: {c: {"n": len(v), "mean_s": (sum(v) / len(v)) if v else None,
                    "median_s": (sorted(v)[len(v) // 2]) if v else None}
                for c, v in tm.items()}
            for L, tm in timing_by_len.items()}
    for L, tm in tsum.items():
        # one item's total = chain + PMI + every condition it ran, averaged
        n_items = sum(1 for r in records if r["length"] == L)
        tot = sum(sum(v) for v in timing_by_len[L].values())
        tm["_total_per_item_s"] = (tot / n_items) if n_items else None

    d1 = self_check["teleport_d1_max_abs"]
    invalid = []
    if self_check["teleport_d1_n"] and d1 > 0.05:
        invalid.append(f"teleport at distance 1 differs from normal by {d1:.4f}"
                       f" nats -- the states are not what this tool thinks")
    if "nocarry" in conditions and self_check["nocarry_n"] and \
            self_check["normal_vs_nocarry_max_abs"] == 0.0:
        invalid.append("normal == nocarry on every item: the carry is INERT "
                       "(not wired, or the overlay did not load)")

    skip_counts = {cell: {k: len(v) for k, v in d.items()}
                   for cell, d in skipped.items()}
    report = {
        "instrument": "babilong dissection (write / retention / read)",
        "when": datetime.now().isoformat(timespec="seconds"),
        "config": {**{k: v for k, v in vars(args).items()},
                   "lengths": lengths, "conditions": conditions,
                   "config_overrides": overrides, "arm": arm,
                   "lap_chunks": lap, "hidden": hidden,
                   "overlay_keys": key_counts, "znorm_target": znorm,
                   "buffer": (type(buf).__name__ if buf is not None else None)},
        "health": health, "self_checks": self_check,
        "invalid": invalid, "reproduction": repro,
        "skipped": skip_counts, "skipped_ids": skipped,
        "timing_seconds_per_item": tsum,
        "aggregates_by_task": {str(k): v for k, v in agg_task.items()},
        "aggregates_by_length": {str(k): v for k, v in agg_len.items()},
        "chance_top1": chance,
        "records": records,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    # ---- printed summary ----------------------------------------------------
    print_tables(agg_task, conditions, chance,
                 "BABILONG DISSECTION by task x distance bucket "
                 "(pooled over lengths)")
    print_tables(agg_len, conditions, chance,
                 "by task/length, pooled over distance (1+ = memory needed)")
    print(HOW_TO_READ)
    print(f"\nHEALTH  mean gold NLL (normal) {mean_nll:.3f} nats/tok vs "
          f"ln(vocab) {ln_v:.2f}: margin {ln_v - mean_nll:+.2f}"
          + ("   *** AT CHANCE -- nothing below is readable ***"
             if health["at_chance"] else "   (ok)"))
    gin = [r for r in records if r["gold_in_final_window"]]
    print(f"NOTE    the benchmark's gold_in_final_window is true for "
          f"{len(gin)}/{len(records)} items here, but the gold word is in the "
          f"SUFFIX DEMOS for {sum(r['gold_in_suffix'] for r in records)} and a "
          f"supporting fact is actually in the window for "
          f"{sum(1 for r in records if r['support_in_window'])}.  Use dist, "
          f"not gold_in_final_window.")
    if "teleport" in conditions:
        print(f"SELF    teleport==normal at dist 1: max |d cand NLL| = "
              f"{d1:.2e} over {self_check['teleport_d1_n']} items"
              + ("   (WARN > 1e-4)" if d1 > 1e-4 else ""))
    if "nocarry" in conditions:
        print(f"SELF    normal vs nocarry: max |d cand NLL| = "
              f"{self_check['normal_vs_nocarry_max_abs']:.3f} over "
              f"{self_check['nocarry_n']} items (must be > 0)")
    if "teleport_donor" in conditions:
        print(f"DONORS  tiers {self_check['donor_tiers']}; no compatible "
              f"donor for {self_check['no_donor']} items")
    fiw_ext = sum(1 for r in records if r["fiw"].get("applied")
                  and not r["fiw"].get("length_matched"))
    fiw_n = sum(1 for r in records if r["fiw"].get("applied"))
    print(f"FIW     fact_in_window on {fiw_n} dist>=1 items; {fiw_ext} of them "
          f"EXTENDED the window (no filler to trade; see [len-matched])")
    print("SKIPS   " + "; ".join(f"{c}: " + (", ".join(
        f"{k}={n}" for k, n in d.items()) or "none")
        for c, d in skip_counts.items()))
    if repro:
        print(f"\nREPRODUCTION vs benchmark records (n={repro['n']}; bench "
              f"bf16 with UNSEEDED priming, this run {args.dtype})")
        if repro["n"]:
            def _f(x, fmt):
                return fmt % x if x is not None else "--"
            print(f"  no-context cand NLL |diff|: mean "
                  f"{_f(repro['nocontext_mean_abs_diff'], '%.4f')} max "
                  f"{_f(repro['nocontext_max_abs_diff'], '%.4f')}  "
                  f"(no carry involved: dtype/hardware only)")
            print(f"  normal cand NLL |diff| mean "
                  f"{_f(repro['normal_cand_mean_abs_diff'], '%.4f')}; argmin "
                  f"agreement {100 * repro['normal_pick_agreement']:.1f}%; "
                  f"top-1 bench {100 * repro['bench_top1']:.1f}% vs here "
                  f"{100 * repro['ours_top1']:.1f}%")
            if repro.get("pmi_pick_agreement") is not None:
                print(f"  PMI argmin agreement "
                      f"{100 * repro['pmi_pick_agreement']:.1f}%; PMI top-1 "
                      f"bench {100 * repro['bench_pmi_top1']:.1f}% vs here "
                      f"{100 * repro['ours_pmi_top1']:.1f}%")
            if repro["normal_pick_agreement"] < 0.8:
                print("  *** WARNING: under 80% argmin agreement with the "
                      "benchmark -- `normal` may not be its condition.")
    print(f"\nTIMING  seconds per item, mean ({args.dtype}, "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    for L, tm in tsum.items():
        tot = tm.get("_total_per_item_s")
        print(f"  {L:>4}: " + ", ".join(
            f"{c}={v['mean_s']:.3f}" for c, v in tm.items()
            if isinstance(v, dict) and v["mean_s"] is not None)
            + (f"  | TOTAL {tot:.2f} s/item" if tot is not None else ""))
    print(f"\nwrote {out_dir / 'results.json'} and {out_dir / 'items.jsonl'} "
          f"({time.time() - t_start:.0f}s total)")
    if invalid:
        print("\n*** RESULTS INVALID ***\n  " + "\n  ".join(invalid))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
