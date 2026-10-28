"""
GPU-free tests for evals/diag_babilong_dissect.py -- the pure pieces that
decide WHICH chunk a fact is in, WHAT the fact_in_window prompt is, and WHICH
WAY a delta points.  Every one of those is a silent-wrong-answer site: a fact
put one chunk off, an insertion that shifts the question, or a flipped sign
all print a tidy table.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_babilong_dissect.py -q
"""
from __future__ import annotations

import os
import random
import re
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))

from evals import diag_babilong_dissect as D  # noqa: E402
from eval_babilong import build_suffix, split_context  # noqa: E402


# ---------------------------------------------------------------------------
# A whitespace/punctuation tokenizer with real offsets, so prepare_item and
# split_context run end-to-end without downloading anything.
# ---------------------------------------------------------------------------

class FakeTok:
    _pat = re.compile(r"\s*(?:\w+|[^\w\s])")

    def __init__(self):
        self.vocab, self.inv = {}, {}

    def _id(self, s):
        if s not in self.vocab:
            self.vocab[s] = len(self.vocab)
            self.inv[self.vocab[s]] = s
        return self.vocab[s]

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False):
        ids, offs = [], []
        for m in self._pat.finditer(text):
            ids.append(self._id(m.group(0)))
            offs.append((m.start(), m.end()))
        return SimpleNamespace(input_ids=ids, offset_mapping=offs)

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.inv[int(i)] for i in ids)


def filler(n_words: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    words = ["the", "old", "house", "stood", "near", "a", "river", "and",
             "many", "people", "walked", "by", "it", "every", "day"]
    return [rng.choice(words) for _ in range(n_words)]


def make_context(n_words: int, facts_at: dict, seed: int = 0) -> str:
    """Filler words with fact sentences spliced in BEFORE word index k."""
    w = filler(n_words, seed)
    out = []
    for i, word in enumerate(w):
        if i in facts_at:
            out.append(facts_at[i])
        out.append(word)
    return " ".join(out) + "."


# ---------------------------------------------------------------------------
# parsing and replay
# ---------------------------------------------------------------------------

def test_parse_facts_vocab_and_boundaries():
    ctx = ("a number of them here Sandra journeyed to the kitchen. Then "
           "John went back to the hallway. Johnny went to the office. "
           "Mary got the milk there. Mary put down the milk. "
           "Daniel left the football there. Daniel went to the market.")
    f = D.parse_facts(ctx)
    got = [(x.kind, x.person, x.target) for x in f]
    assert got == [("move", "Sandra", "kitchen"), ("move", "John", "hallway"),
                   ("take", "Mary", "milk"), ("drop", "Mary", "milk"),
                   ("drop", "Daniel", "football")]
    # "went back to" must win over "went to" and the span must be exact
    assert f[1].text == "John went back to the hallway."
    assert ctx[f[0].start:f[0].end] == "Sandra journeyed to the kitchen."


def test_parse_question():
    assert D.parse_question("qa1", "Where is John? ") == "John"
    assert D.parse_question("qa2", "Where is the milk? ") == "milk"
    with pytest.raises(D.Ambiguous):
        D.parse_question("qa1", "Who is John?")
    with pytest.raises(ValueError):
        D.parse_question("qa3", "Where was the milk before the garden?")


def _facts(*specs):
    out = []
    for kind, person, target in specs:
        out.append(D.Fact(kind, person, target, 0, 0, f"{person} {target}"))
    return out


def test_world_answer_qa1_is_last_move():
    f = _facts(("move", "Mary", "kitchen"), ("move", "John", "office"),
               ("move", "Mary", "garden"), ("move", "John", "hallway"))
    assert D.world_answer("qa1", f, "Mary") == ("garden", [2])
    assert D.world_answer("qa1", f, "Sandra") == (None, [])


def test_world_answer_qa2_held_dropped_and_late_resolution():
    held = _facts(("move", "Mary", "kitchen"), ("take", "Mary", "milk"),
                  ("move", "Mary", "office"))
    assert D.world_answer("qa2", held, "milk") == ("office", [1, 2])
    held_before = _facts(("move", "Mary", "kitchen"), ("take", "Mary", "milk"))
    assert D.world_answer("qa2", held_before, "milk") == ("kitchen", [0, 1])
    dropped = _facts(("move", "Mary", "kitchen"), ("take", "Mary", "milk"),
                     ("drop", "Mary", "milk"), ("move", "Mary", "office"))
    assert D.world_answer("qa2", dropped, "milk") == ("kitchen", [0, 2])
    # REGRESSION: a drop before the dropper has any known location used to
    # abort the replay; a later take must still resolve it (qa2/1k/14).
    late = _facts(("take", "Mary", "apple"), ("drop", "Mary", "apple"),
                  ("move", "Mary", "bathroom"), ("take", "Mary", "apple"),
                  ("drop", "Mary", "apple"))
    assert D.world_answer("qa2", late, "apple") == ("bathroom", [2, 4])
    unresolved = _facts(("take", "Mary", "apple"), ("drop", "Mary", "apple"))
    assert D.world_answer("qa2", unresolved, "apple") == (None, [])
    inconsistent = _facts(("move", "John", "office"),
                          ("drop", "John", "milk"))
    assert D.world_answer("qa2", inconsistent, "milk") == (None, [])


# ---------------------------------------------------------------------------
# geometry: must describe exactly what split_context built
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_words", [5, 60, 300, 1500, 4000])
@pytest.mark.parametrize("seq_len,num_chunks", [(512, 8), (512, 4), (512, 0),
                                                (64, 0), (200, 8)])
def test_geometry_matches_split_context(n_words, seq_len, num_chunks):
    tok = FakeTok()
    ctx = make_context(n_words, {})
    sfx = build_suffix("qa1", "Where is Mary? ")
    prime, final = split_context(tok, ctx, sfx, seq_len, 12,
                                 num_chunks=num_chunks)
    ctx_ids = tok(ctx).input_ids
    sfx_ids = tok(sfx).input_ids
    g = D.chunk_geometry(len(ctx_ids), len(sfx_ids), seq_len, 12, num_chunks)
    D.verify_geometry(g, prime, final, ctx_ids, sfx_ids)   # raises on drift
    # every context token maps to exactly the chunk that holds it
    for t in range(len(ctx_ids)):
        c = D.token_chunk(t, g)
        assert c != -1        # split_context never drops a context token
        if c < g.n_prime:
            k = t - c * g.win
            assert prime[c][0, k].item() == ctx_ids[t]
        else:
            pos = t - (g.n_ctx - g.story_in_final)
            assert final[0, pos].item() == ctx_ids[t]


def test_verify_geometry_catches_drift():
    tok = FakeTok()
    ctx = make_context(1500, {})
    sfx = build_suffix("qa1", "Where is Mary? ")
    prime, final = split_context(tok, ctx, sfx, 512, 12, num_chunks=8)
    ctx_ids, sfx_ids = tok(ctx).input_ids, tok(sfx).input_ids
    wrong = D.chunk_geometry(len(ctx_ids), len(sfx_ids), 512, 12, 4)
    with pytest.raises(RuntimeError, match="CHUNKING DRIFT"):
        D.verify_geometry(wrong, prime, final, ctx_ids, sfx_ids)


def test_char_span_to_tokens_leading_space_and_period():
    tok = FakeTok()
    s = "blah here Sandra journeyed to the kitchen. His"
    enc = tok(s, return_offsets_mapping=True)
    f = D.parse_facts(s)[0]
    t0, t1 = D.char_span_to_tokens(enc.offset_mapping, f.start, f.end)
    assert tok.decode(enc.input_ids[t0:t1 + 1]).strip() == f.text


def test_distance_bucket_edges():
    want = {0: "0", 1: "1", 2: "2-3", 3: "2-3", 4: "4-7", 7: "4-7", 8: "8+",
            30: "8+"}
    for d, b in want.items():
        assert D.distance_bucket(d) == b
    assert D.distance_bucket(None) is None
    with pytest.raises(ValueError):
        D.distance_bucket(-1)
    assert D.bucket_members("1+", "4-7") and not D.bucket_members("1+", "0")
    assert D.bucket_members("all", "0") and not D.bucket_members("all", None)


# ---------------------------------------------------------------------------
# prepare_item end to end, on the fake tokenizer
# ---------------------------------------------------------------------------

def _item(n_words, facts_at, question="Where is Mary? ", target="garden",
          task="qa1", seed=0):
    tok = FakeTok()
    ex = {"input": make_context(n_words, facts_at, seed), "question": question,
          "target": target}
    return tok, D.prepare_item(tok, task, "1k", 0, ex, 512, 12, 8,
                               D.LOCATIONS)


def test_prepare_item_locates_last_move_and_distance():
    # earlier Mary move in chunk 0, the LAST one near word 700
    tok, it = _item(1500, {20: "Mary went to the office.",
                           700: "Mary journeyed to the garden.",
                           900: "John moved to the garden."})
    assert it["support"][0]["text"] == "Mary journeyed to the garden."
    tok_idx = it["support"][0]["tokens"][1]
    assert it["src_chunk"] == tok_idx // it["window"]
    assert it["distance"] == it["n_prime_chunks"] - it["src_chunk"]
    assert it["distance"] >= 1
    assert it["dist_bucket"] == D.distance_bucket(it["distance"])


def test_prepare_item_answer_mismatch_is_skipped():
    with pytest.raises(D.Ambiguous) as e:
        _item(1500, {700: "Mary journeyed to the garden."}, target="kitchen")
    assert e.value.reason == "answer_mismatch"


def test_prepare_item_fact_in_question_window_is_distance_zero():
    # put the fact in the last few words: it lands in the question window
    tok, it = _item(1500, {1495: "Mary journeyed to the garden."})
    assert it["distance"] == 0 and it["src_chunk"] is None
    assert it["support_in_window"] == 1
    assert it["fiw_ids"] is None and it["dist_bucket"] == "0"


def test_fact_in_window_length_matched_and_suffix_untouched():
    tok, it = _item(1500, {700: "Mary journeyed to the garden.",
                           1497: "John moved to the office."})
    assert it["fiw"]["applied"] and it["fiw"]["length_matched"]
    fin, fiw = it["final_ids"][0].tolist(), it["fiw_ids"][0].tolist()
    assert len(fin) == len(fiw)
    n_sfx = it["n_sfx_tokens"]
    assert fiw[-n_sfx:] == fin[-n_sfx:]               # question untouched
    n_fact = it["fiw"]["n_tokens"]
    story_end = len(fin) - n_sfx
    assert tok.decode(fiw[story_end - n_fact:story_end]).strip() == \
        "Mary journeyed to the garden."
    # the in-window (protected) fact survived the deletion
    assert "John moved to the office." in tok.decode(fiw[:story_end])


def test_fact_in_window_extends_when_no_room_and_says_so():
    # 1k-like: ~600 words -> ~90-token window holding ~1 story token
    tok, it = _item(600, {300: "Mary journeyed to the garden."})
    fiw = it["fiw"]
    assert fiw["applied"] and fiw["length_matched"] is False
    assert fiw["extended_by"] == fiw["n_tokens"] > 0
    assert it["fiw_ids"].shape[1] == it["final_ids"].shape[1] + fiw["n_tokens"]
    n_sfx = it["n_sfx_tokens"]
    assert it["fiw_ids"][0, -n_sfx:].tolist() == \
        it["final_ids"][0, -n_sfx:].tolist()


def test_insert_facts_rules():
    story = [10, 11, 12, 13, 14, 15]
    prot = [False, True, False, False, True, False]
    new, dropped = D.insert_facts(story, prot, [90, 91])
    assert dropped == [0, 2]                      # first UNPROTECTED tokens
    assert new == [11, 13, 14, 15, 90, 91]        # protected kept, in order
    assert len(new) == len(story)
    assert D.insert_facts(story, [True] * 5 + [False], [90, 91]) is None
    with pytest.raises(ValueError):
        D.insert_facts(story, prot[:-1], [90])


# ---------------------------------------------------------------------------
# scoring and sign conventions
# ---------------------------------------------------------------------------

def test_score_ranked_rank_top1_pmi_and_ties():
    cands = ["bathroom", "garden", "office"]
    s = D.score_ranked(cands, [3.0, 1.0, 2.0], [1, 1, 1], "garden",
                       [0.5, 2.5, 0.1])
    assert s["top1"] and s["rank"] == 1 and s["gold_nll"] == 1.0
    # PMI: 2.5, -1.5, 1.9 -> garden still the argmin
    assert s["pmi_top1"] and s["pmi_rank"] == 1
    s2 = D.score_ranked(cands, [1.0, 1.0, 2.0], [1, 1, 1], "garden")
    assert not s2["top1"] and s2["rank"] == 1     # tie: first index wins
    s3 = D.score_ranked(cands, [3.0, 4.0, 2.0], [2, 2, 1], "garden")
    assert s3["rank"] == 3 and s3["gold_nll"] == 2.0
    with pytest.raises(ValueError):
        D.score_ranked(cands, [1, 2, 3], [1, 1, 1], "kitchen")


def _rec(i, dist, conds, rows=None, fiw=None, task="qa1"):
    return {"id": f"{task}/1k/{i}", "task": task, "length": "1k",
            "dist_bucket": D.distance_bucket(dist), "conditions": conds,
            "state_rows": rows or {}, "fiw": fiw or {}}


def _sc(nll, top1, pmi=None):
    d = {"gold_nll": nll, "top1": top1, "rank": 1 if top1 else 3}
    if pmi is not None:
        d["pmi_top1"] = pmi
    return d


def test_paired_delta_signs():
    recs = [_rec(0, 2, {"teleport": _sc(1.0, True, True),
                        "normal": _sc(3.0, False, False)})]
    d = D.paired_deltas(recs, "teleport", "normal")
    assert d["nll"] == [-2.0]          # A better -> negative dNLL
    assert d["top1"] == [1.0]          # A better -> positive dTop1
    assert d["pmi_top1"] == [1.0]
    # an item missing either condition is not paired
    recs.append(_rec(1, 2, {"teleport": None, "normal": _sc(2.0, True)}))
    assert len(D.paired_deltas(recs, "teleport", "normal")["ids"]) == 1


def test_paired_delta_filters():
    recs = [_rec(0, 6, {"teleport": _sc(1.0, True), "normal": _sc(2.0, False)},
                 rows={"teleport": 32, "normal": 64}),
            _rec(1, 6, {"teleport": _sc(1.0, True), "normal": _sc(2.0, False)},
                 rows={"teleport": 64, "normal": 64}),
            _rec(2, 6, {"fact_in_window": _sc(1.0, True),
                        "normal": _sc(2.0, False)},
                 fiw={"applied": True, "length_matched": False})]
    assert D.paired_deltas(recs, "teleport", "normal",
                           "equal_rows")["ids"] == ["qa1/1k/1"]
    assert D.paired_deltas(recs, "fact_in_window", "normal",
                           "fiw_matched")["ids"] == []
    with pytest.raises(ValueError):
        D.paired_deltas(recs, "teleport", "normal", "bogus")


def test_aggregate_buckets_and_contrasts():
    conds = ["normal", "nocarry", "teleport", "teleport_donor",
             "fact_in_window"]
    recs = [
        _rec(0, 0, {"normal": _sc(2.0, True), "nocarry": _sc(2.5, False),
                    "teleport": None, "teleport_donor": None,
                    "fact_in_window": None}),
        _rec(1, 1, {"normal": _sc(3.0, False), "nocarry": _sc(3.0, False),
                    "teleport": _sc(3.0, False),
                    "teleport_donor": _sc(3.5, False),
                    "fact_in_window": _sc(1.0, True)},
             rows={"normal": 64, "teleport": 64, "teleport_donor": 64},
             fiw={"applied": True, "length_matched": True}),
        _rec(2, 5, {"normal": _sc(4.0, False), "nocarry": _sc(4.0, False),
                    "teleport": _sc(2.0, True),
                    "teleport_donor": _sc(4.0, False),
                    "fact_in_window": _sc(1.0, True)},
             rows={"normal": 64, "teleport": 48, "teleport_donor": 48},
             fiw={"applied": True, "length_matched": False}),
    ]
    agg = D.aggregate(recs, conds, n_boot=200, seed=0)["qa1"]
    assert set(agg) == {"0", "1", "4-7", "1+", "all"}
    assert agg["all"]["n_items"] == 3 and agg["1+"]["n_items"] == 2
    assert agg["0"]["conditions"]["teleport"] == {"n": 0}
    c = agg["1+"]["contrasts"]
    assert c["teleport - normal"]["n"] == 2
    assert c["teleport - normal"]["d_nll"][0] == pytest.approx(-1.0)
    assert c["teleport - normal [equal rows]"]["n"] == 1
    assert c["teleport - teleport_donor"]["d_nll"][0] == pytest.approx(-1.25)
    assert c["fact_in_window - normal [len-matched]"]["n"] == 1
    assert c["normal - nocarry"]["d_top1"][0] == 0.0
    # bucket 0 has no teleport, so no teleport contrast at all
    assert "teleport - normal" not in agg["0"]["contrasts"]


def test_boot_ci_edges():
    assert D.boot_ci([], 100, 0) == (None, None, None)
    assert D.boot_ci([0.3], 100, 0) == (0.3, None, None)
    m, lo, hi = D.boot_ci([0.5] * 10, 100, 0)
    assert m == lo == hi == pytest.approx(0.5)
    m, lo, hi = D.boot_ci(list(range(20)), 500, 0)
    assert lo < m < hi


# ---------------------------------------------------------------------------
# donors
# ---------------------------------------------------------------------------

def test_donor_tier():
    assert D.donor_tier(5, 64, 5, 64, 4) == 0
    assert D.donor_tier(5, 64, 1, 32, 4) is None      # row count differs
    assert D.donor_tier(5, 64, 9, 64, 4) == 1         # same phase, full ring
    assert D.donor_tier(5, 64, 6, 64, 4) is None      # different phase


def test_donor_compatible_rejects_shared_story():
    mary_garden = _facts(("move", "Mary", "garden"))
    # the donor's own answer differs, but its state holds THIS item's fact
    assert not D.donor_compatible("qa1", "a", "garden", "Mary", "b",
                                  "kitchen", mary_garden)
    assert not D.donor_compatible("qa1", "a", "garden", "Mary", "b",
                                  "garden", [])          # same gold
    assert not D.donor_compatible("qa1", "a", "garden", "Mary", "a",
                                  "kitchen", [])         # same row
    assert D.donor_compatible("qa1", "a", "garden", "Mary", "b", "kitchen",
                              _facts(("move", "Mary", "office"),
                                     ("move", "John", "kitchen")))


def test_pick_donor_prefers_tier0_and_is_deterministic():
    item = {"id": "a", "task": "qa1", "gold": "garden", "entity": "Mary",
            "src": 5, "rows": 64}
    pool = [{"id": "b", "gold": "kitchen", "src": 9, "rows": 64,
             "facts_upto_src": []},
            {"id": "c", "gold": "office", "src": 5, "rows": 64,
             "facts_upto_src": []},
            {"id": "d", "gold": "garden", "src": 5, "rows": 64,
             "facts_upto_src": []},
            {"id": "a", "gold": "garden", "src": 5, "rows": 64,
             "facts_upto_src": []}]
    d, tier = D.pick_donor(item, pool, 4, random.Random(0))
    assert (d["id"], tier) == ("c", 0)
    d2, tier2 = D.pick_donor(item, pool[:1], 4, random.Random(0))
    assert (d2["id"], tier2) == ("b", 1)
    assert D.pick_donor(item, pool[2:], 4, random.Random(0)) == (None, None)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_resolve_conditions_refuses_carry_without_buffer():
    assert D.resolve_conditions(["auto"], False) == ["normal",
                                                     "fact_in_window"]
    assert D.resolve_conditions(["auto"], True) == list(D.CONDITIONS)
    with pytest.raises(SystemExit):
        D.resolve_conditions(["normal", "teleport"], False)
    with pytest.raises(SystemExit):
        D.resolve_conditions(["teleport_donor", "normal"], True)


def test_expected_znorm_paths():
    assert D.expected_znorm("cortex-5b/cortex-final/checkpoint_38146") == 78.9
    assert D.expected_znorm("cortex-5b\\cortex-final\\checkpoint_11444") == 98.0
    assert D.expected_znorm("C:/tmp/slim/cortex-final_38146") == 78.9
    assert D.expected_znorm("cortex-5b/c-chunked/checkpoint_38146") is None
    assert D.expected_znorm("cortex-5b/cortex-final/checkpoint_22800") is None
    assert D.expected_znorm(None) is None


def test_check_structure_fails_fast_on_chunking_mismatch():
    item = {"id": "qa1/1k/0", "n_prime_chunks": 7,
            "gold_in_final_window": True, "gold": "garden"}
    D.check_structure(item, None, ["garden"])          # no record: no check
    D.check_structure(item, {"n_prime_chunks": 7, "gold": "garden",
                             "gold_in_final_window": True,
                             "candidates": ["garden"]}, ["garden"])
    with pytest.raises(SystemExit):
        D.check_structure(item, {"n_prime_chunks": 8, "gold": "garden"},
                          ["garden"])


def test_parse_lengths():
    assert D.parse_lengths(["1k", "2000", "16K"]) == ["1k", "2k", "16k"]
