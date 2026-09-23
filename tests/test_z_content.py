"""
Z attempt 2, Step 0: D1 (seed vs document) and D2 (what Z holds beyond E).

Written WITH the instrument.  Every earlier Z instrument that returned a
confident wrong number (REDs 8-15) did so through a control that did not do
what its label said.  So the controls get pinned here first: the reseed must
move s0 and only s0, the donor must be another book at the same chunk index,
the split must not leak a book, and the probe must find content where content
was planted and not where it was not.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_z_content.py -q
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "evals"))

from evals.diag_z_content import (  # noqa: E402
    CONTENT_CUT, ENCODINGS, NOISE_CUT, Z_ENCODINGS, book_ids, build_vocab,
    is_book_end, pack_eos,
    collect, d1_reading, d1_report, d2_report, donor_index, print_report,
    reseed_verdict,
)


# ─── books, donors ──────────────────────────────────────────────────────────

class TestBooks:
    def test_a_new_book_starts_after_each_ragged_last_window(self):
        # stride_windows: every book ends in exactly one ragged row.
        ragged = [False, False, True, False, True, False, False, True]
        books, note = book_ids(ragged)
        assert books == [0, 0, 0, 1, 1, 2, 2, 2]
        assert "document ends: 3 books" in note

    def test_it_agrees_with_the_real_packer(self):
        """Pin against stride_windows itself, not a description of it.  The
        9-token book is the edge case that shaped is_book_end: 9 tokens + the
        eos marker fill a 10-token row exactly, so nothing is padded and a
        padding-only rule merged it into the next book."""
        from tools.prepare_pg19_dataset import stride_windows
        rows = []
        for length in (25, 9, 31):          # three books, row_len 10
            rows += stride_windows(list(range(1, length + 1)), 10, eos=0)
        ends = [is_book_end(ids, m, eos=0) for ids, m in rows]
        books, _ = book_ids(ends)
        assert books == [0, 0, 0, 1, 2, 2, 2, 2]
        padding_only = [min(m) == 0 for _, m in rows]
        assert book_ids(padding_only)[0] != books

    def test_the_eos_comes_from_the_pack_not_the_config(self):
        # B2-family configs say 65505 (Huginn); the pack pads with OLMo's.
        rows = [{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1]},
                {"input_ids": [5, 9, 9], "attention_mask": [1, 0, 0]}]
        assert pack_eos(rows, 65505)[0] == 9
        assert pack_eos(rows[:1], 65505)[0] == 65505

    def test_a_full_mid_book_window_is_not_an_end(self):
        assert not is_book_end([5, 6, 7], [1, 1, 1], eos=0)
        assert is_book_end([5, 6, 0], [1, 1, 1], eos=0)
        assert is_book_end([5, 0, 0], [1, 1, 0], eos=None)

    def test_a_bos_is_used_only_when_no_ends_mark_books(self):
        books, note = book_ids([False] * 7, [7, 3, 4, 7, 9, 7, 1], bos_id=7)
        assert books == [0, 0, 0, 1, 1, 2, 2] and "bos 7" in note

    def test_no_marker_falls_back_loudly(self):
        books, note = book_ids([False] * 45, fallback_block=20)
        assert books[:20] == [0] * 20 and books[20] == 1 and books[44] == 2
        assert "FALLBACK" in note and "leak" in note

    def test_a_bos_that_marks_one_book_is_not_trusted(self):
        _, note = book_ids([False] * 4, [7, 1, 2, 3], bos_id=7)
        assert "FALLBACK" in note


class TestDonors:
    def test_the_donor_is_another_book_at_the_same_chunk(self):
        books = [0, 0, 0, 1, 1, 1, 2, 2, 2] * 2
        chunks = [0, 1, 2] * 6
        d = donor_index(books, chunks)
        for i, j in enumerate(d):
            assert books[i] != books[j]
            assert chunks[i] == chunks[j]

    def test_one_book_has_no_donor_and_says_so(self):
        with pytest.raises(ValueError, match="one book"):
            donor_index([0, 0, 0], [0, 0, 0])

    def test_a_lopsided_book_mix_still_finds_other_books(self):
        books = [0] * 9 + [1]
        d = donor_index(books, [0] * 10)
        assert all(books[i] != books[j] for i, j in enumerate(d))


# ─── D1 ─────────────────────────────────────────────────────────────────────

def _d1_samples(content: float, noise: float, n_books=6, per_book=4, R=3,
                D=16, T=3, seed=0):
    """Fake D1 samples whose encoding is content(book, chunk) + noise(seed).

    seed_sq is what `collect` records: the squared row distance between two
    independent noise draws of the same sample, i.e. 2 * noise^2 * D.
    """
    g = torch.Generator().manual_seed(seed)
    out = []
    for b in range(n_books):
        for c in range(per_book):
            base = torch.randn(R, D, generator=g) * content
            rec = {"book": b, "chunk": c, "row": b, "tape_seed_sq": []}
            seed_sq = {}
            for k in ENCODINGS:
                xa = base + torch.randn(R, D, generator=g) * noise
                xb = base + torch.randn(R, D, generator=g) * noise
                rec[k] = xa
                seed_sq[k] = float((xa - xb).pow(2).sum(-1).mean())
            rec["seed_sq"] = seed_sq
            tape_a, ts = [], []
            for _ in range(T):
                ta = base + torch.randn(R, D, generator=g) * noise
                tb = base + torch.randn(R, D, generator=g) * noise
                tape_a.append(ta)
                ts.append(float((ta - tb).pow(2).sum(-1).mean()))
            rec["tape"] = torch.stack(tape_a)
            rec["tape_seed_sq"] = ts
            out.append(rec)
    return out


class TestD1:
    def test_pure_noise_reads_near_one(self):
        rep = d1_report(_d1_samples(content=0.0, noise=1.0), 200, 0)
        for k in ENCODINGS:
            assert 0.8 < rep["encodings"][k]["ratio"] < 1.2
            assert rep["encodings"][k]["reading"] == "NOISE-DOMINATED"

    def test_no_noise_reads_zero(self):
        rep = d1_report(_d1_samples(content=1.0, noise=0.0), 200, 0)
        for k in ENCODINGS:
            assert rep["encodings"][k]["ratio"] == 0.0
            assert rep["encodings"][k]["reading"] == "CONTENT-DETERMINED"

    def test_the_ratio_is_the_noise_share_of_the_variance(self):
        # noise^2 / (noise^2 + content^2) = 1 / (1 + 4) = 0.2
        rep = d1_report(_d1_samples(content=2.0, noise=1.0, n_books=12), 200, 0)
        r = rep["encodings"]["Z_delta"]["ratio"]
        assert 0.14 < r < 0.27
        lo, hi = rep["encodings"]["Z_delta"]["ci_lo"], rep["encodings"]["Z_delta"]["ci_hi"]
        assert lo <= r <= hi

    def test_it_reports_every_depth_and_names_the_best_z(self):
        rep = d1_report(_d1_samples(content=1.0, noise=0.5, T=4), 100, 0)
        assert [d["t"] for d in rep["depths"]] == [1, 2, 3, 4]
        assert rep["lowest_ratio_z"] in Z_ENCODINGS

    def test_the_cuts_read_as_documented(self):
        assert d1_reading(NOISE_CUT + 0.01) == "NOISE-DOMINATED"
        assert d1_reading(CONTENT_CUT - 0.01) == "CONTENT-DETERMINED"
        assert d1_reading(0.3) == "MIXED"

    def test_too_few_samples_refuse(self):
        with pytest.raises(ValueError):
            d1_report(_d1_samples(1.0, 1.0, n_books=1, per_book=2), 10, 0)


class TestReseedVerdict:
    def _check(self, repeat, seed):
        return {k: {"repeat_sq": repeat, "seed_sq": seed} for k in ENCODINGS}

    def test_a_clean_reseed_passes(self):
        assert reseed_verdict(self._check(0.0, 1.0)) is None

    def test_a_seed_that_moves_nothing_fails(self):
        assert "did not move" in reseed_verdict(self._check(0.0, 0.0))

    def test_other_randomness_in_the_forward_fails(self):
        assert "other than the seed" in reseed_verdict(self._check(0.5, 1.0))

    def test_no_check_at_all_fails(self):
        assert reseed_verdict({}) is not None


# ─── D2 ─────────────────────────────────────────────────────────────────────

V_TOY = 60


def _d2_samples(n_books=10, rows_per_book=3, n_chunks=4, R=2, D=24, seed=0):
    """Planted content.  Each book draws its words from its own topic, and
    each chunk has its own random mix within that topic, so a chunk's bag of
    words is not predictable from the book alone.

      E        a linear image of THIS chunk's bag of words
      Z_delta  a linear image of the NEXT chunk's bag of words (content E lacks)
      Z_end    pure noise
      Z_tok    a copy of E (redundant: no content beyond E)
    """
    g = torch.Generator().manual_seed(seed)
    WE = torch.randn(V_TOY, R * D, generator=g)
    WZ = torch.randn(V_TOY, R * D, generator=g)
    samples = []
    ri = 0
    for b in range(n_books):
        topic = torch.randperm(V_TOY, generator=g)[:20]
        for _ in range(rows_per_book):
            chunk_ids = []
            for c in range(n_chunks):
                sub = topic[torch.randperm(20, generator=g)[:6]]
                chunk_ids.append(sub[torch.randint(0, 6, (30,), generator=g)])
            bows = [torch.log1p(torch.bincount(ids, minlength=V_TOY).float())
                    for ids in chunk_ids]
            for c in range(n_chunks):
                nxt = bows[c + 1] if c + 1 < n_chunks else torch.zeros(V_TOY)
                e = (bows[c] @ WE).view(R, D)
                samples.append({
                    "row": ri, "book": b, "chunk": c, "ids": chunk_ids[c],
                    "nll": float(nxt.sum()) / 10.0,
                    "E": e,
                    "Z_delta": (nxt @ WZ).view(R, D),
                    "Z_end": torch.randn(R, D, generator=g),
                    "Z_tok": e.clone(),
                })
            ri += 1
    return samples


@pytest.fixture(scope="module")
def rep():
    # 15 books, every 3rd held out = 5 test books, the MIN_TEST_BOOKS floor.
    return d2_report(_d2_samples(n_books=15), V_TOY, torch.device("cpu"),
                     test_every=3, n_folds=3, bow_vocab=40, bow_skip_top=0,
                     n_boot=300, seed=0)


class TestD2:

    def test_the_probe_health_control_sees_e_record_its_own_chunk(self, rep):
        assert rep["probe_health_ok"]
        assert rep["r2"]["E"]["bow_self"] > 0.5

    def test_planted_next_chunk_content_is_found_beyond_e(self, rep):
        g = rep["gains"]["Z_delta"]["bow_next"]
        assert g["holds_content"], g
        assert g["content_gain"] > 0.1

    def test_noise_holds_no_content(self, rep):
        g = rep["gains"]["Z_end"]["bow_next"]
        assert not g["holds_content"], g

    def test_a_copy_of_e_holds_nothing_beyond_e(self, rep):
        # THE CASE THAT SHAPED THE RULE.  Real-minus-donor is POSITIVE here
        # (the donor block hurts the fit), so a rule on the donor alone would
        # call a redundant copy of E "content".  The gain over E alone is ~0,
        # and holds_content requires both.
        for t in ("bow_self", "bow_next"):
            g = rep["gains"]["Z_tok"][t]
            assert not g["holds_content"], (t, g)
            assert abs(g["gain_over_E"]) < 0.05, (t, g)

    def test_planted_content_also_clears_the_gain_over_e(self, rep):
        g = rep["gains"]["Z_delta"]["bow_next"]
        assert g["gain_over_E_ci"][0] > 0, g

    def test_the_split_is_by_book_and_both_sides_have_books(self, rep):
        assert rep["n_test_books"] == 5 and rep["enough_test_books"]
        assert rep["n_train_books"] == 10
        assert rep["n_train"] + rep["n_test"] == 15 * 3 * 3   # last chunk has no next

    def test_every_z_encoding_has_every_target(self, rep):
        for z in Z_ENCODINGS:
            assert set(rep["gains"][z]) == {"bow_self", "bow_next", "nll_next"}
            for name in (z, f"E+{z}", f"E+donor:{z}"):
                assert name in rep["r2"]

    def test_a_failed_probe_health_clears_every_flag(self):
        s = _d2_samples()
        g = torch.Generator().manual_seed(9)
        for x in s:                      # E carries nothing about its chunk
            x["E"] = torch.randn(x["E"].shape, generator=g)
        r = d2_report(s, V_TOY, torch.device("cpu"), test_every=3, n_folds=3,
                      bow_vocab=40, bow_skip_top=0, n_boot=100, seed=0)
        assert not r["probe_health_ok"]
        assert not any(r["gains"][z][t]["holds_content"]
                       for z in r["gains"] for t in r["gains"][z])

    def test_too_few_test_books_clear_every_flag(self):
        r = d2_report(_d2_samples(n_books=6), V_TOY, torch.device("cpu"),
                      test_every=3, n_folds=2, bow_vocab=40, bow_skip_top=0,
                      n_boot=100, seed=0)
        assert r["n_test_books"] == 2 and not r["enough_test_books"]
        assert not any(r["gains"][z][t]["holds_content"]
                       for z in r["gains"] for t in r["gains"][z])

    def test_too_few_books_to_split_refuse(self):
        with pytest.raises(ValueError, match="each side"):
            d2_report(_d2_samples(n_books=3), V_TOY, torch.device("cpu"),
                      test_every=5, n_folds=2, bow_vocab=20, bow_skip_top=0,
                      n_boot=10, seed=0)

    def test_the_vocab_drops_the_most_frequent_first(self):
        ids = [torch.tensor([1, 1, 2, 3]), torch.tensor([1, 2]),
               torch.tensor([1, 4])]
        v = build_vocab(ids, 10, skip_top=1, size=2)
        assert 1 not in v.tolist() and v.tolist()[0] == 2

    def test_it_prints_and_stays_ascii(self, rep, capsys):
        d1 = d1_report(_d1_samples(1.0, 0.5), 50, 0)
        print_report({"d1": d1, "d2": rep})
        out = capsys.readouterr().out
        out.encode("ascii")
        assert "D1 -- seed vs document" in out and "D2 -- what Z holds" in out


# ─── the real capture, on a tiny real model ─────────────────────────────────

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

NV, K, CL, EOS, T = 4, 16, 16, VOCAB - 1, 4
NC = 4


def _model(gated=True):
    torch.manual_seed(1234)
    common = dict(use_memory=True, memory_slots=0, accum_vecs=NV,
                  latent_carry=True, eos_token_id=EOS)
    if gated:
        common.update(prefix_memory="gated", gate_slots=K, gate_route="ring",
                      gate_init="zero", gate_fill="grow")
    else:
        common.update(prefix_memory="accum", accum_max=K * 4)
    return _build_raven(**common).eval()


def _rows(n=6):
    torch.manual_seed(0)
    return [(i, i // 2, torch.randint(0, VOCAB - 1, (CL * NC + 1,)))
            for i in range(n)]


@pytest.mark.parametrize("gated", [True, False])
class TestCaptureOnTheRealLoop:
    def _collect(self, gated):
        m = _model(gated)
        rows = _rows()
        samples, check = collect(m, m.cortex, rows, NC, torch.tensor([T, 0]),
                                 torch.device("cpu"), 1234, {0, 2, 4},
                                 tok_rows=8, tok_pool=2, log_every=0)
        return m, samples, check

    def test_one_record_per_chunk_with_every_encoding_at_equal_width(self, gated):
        _, samples, _ = self._collect(gated)
        assert len(samples) == 6 * NC
        for s in samples:
            shapes = {k: tuple(s[k].shape) for k in ENCODINGS}
            assert len(set(shapes.values())) == 1, shapes
            assert all(s[k].dtype == torch.float32 for k in ENCODINGS)

    def test_the_seed_moves_s0_and_nothing_else_does(self, gated):
        _, _, check = self._collect(gated)
        assert reseed_verdict(check) is None, check
        for k in Z_ENCODINGS:
            assert check[k]["repeat_sq"] == 0.0
            assert check[k]["seed_sq"] > 0.0

    def test_only_the_d1_rows_carry_the_seed_rerun(self, gated):
        _, samples, _ = self._collect(gated)
        for s in samples:
            assert ("seed_sq" in s) == (s["row"] in {0, 2, 4})
            if "tape" in s:
                assert s["tape"].shape[0] == T
                assert len(s["tape_seed_sq"]) == T

    def test_the_merge_hook_is_removed_afterwards(self, gated):
        m, _, _ = self._collect(gated)
        assert "merge" not in vars(m.cortex.prefix)

    def test_the_chain_is_the_same_whether_or_not_d1_reruns(self, gated):
        """The seed-b re-run must not leak into the chain: the carry the next
        chunk reads comes from seed a either way."""
        m = _model(gated)
        rows = _rows(2)
        a, _ = collect(m, m.cortex, rows, NC, torch.tensor([T, 0]),
                       torch.device("cpu"), 1234, {0, 1}, 8, 2, log_every=0)
        b, _ = collect(m, m.cortex, rows, NC, torch.tensor([T, 0]),
                       torch.device("cpu"), 1234, set(), 8, 2, log_every=0)
        for x, y in zip(a, b):
            for k in ENCODINGS:
                assert torch.equal(x[k], y[k]), (k, x["row"], x["chunk"])
            assert x["nll"] == y["nll"]

    def test_d1_runs_end_to_end_on_the_capture(self, gated):
        _, samples, _ = self._collect(gated)
        rep = d1_report(samples, 50, 0)
        assert rep["n_books"] == 3
        for k in ENCODINGS:
            assert rep["encodings"][k]["doc_var"] > 0


class TestTheLauncher:
    SB = os.path.join(REPO, "pace", "z2_diag_content.sbatch")

    def _src(self):
        with open(self.SB, encoding="utf-8") as fh:
            return fh.read()

    def test_it_scores_the_probe_dirs_by_their_real_names(self):
        # The P1 Z arms exist ONLY as 400-step probes, so the probe- prefix is
        # right here -- the opposite of RED 13, and pinned for the same reason.
        s = self._src()
        assert "probe-p1-a2-accum-w16-cc8-z" in s
        assert "probe-p1-a3z-gated-w16k64-cc8-z" in s

    def test_it_forces_the_geometry_and_the_z_channel(self):
        s = self._src()
        for flag in ("use_memory=true", "latent_carry=true",
                     "prefix_memory=$PREFIX_MODE", "latent_depth_lo=2",
                     "latent_depth_hi=9"):
            assert flag in s, flag

    def test_it_reads_the_held_out_strided_pack_in_fp32(self):
        s = self._src()
        assert "pg19_olmo_validation_len4096_strided" in s
        assert "--dtype float32" in s

    def test_no_var_assignment_prefix_on_the_python_line(self):
        # Bit three times already (cortex_final_prelaunch_suite).
        for line in self._src().splitlines():
            if "python evals/diag_z_content.py" in line:
                assert "=" not in line.split("python")[0]
