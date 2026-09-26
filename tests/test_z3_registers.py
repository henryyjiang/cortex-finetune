"""
D3, the register probe (evals/diag_z_registers.py; j4_handoff.md step 1).

What has to be true before its numbers can decide J4's write:
  1. The register replay reads rows exactly as tools/prepare_carry_task.py wrote
     them -- answer positions match answer_dep, values obey the arithmetic.
  2. Targets and categories are taken at the chunk END with the eval's chunking.
  3. The probe decodes a planted code, stays at the baseline on noise, and
     never lets a row sit on both sides of a split.
  4. The capture runs end to end on a real (tiny) model: E, Z_write, Z_tok,
     Z_end every chunk, the ring from chunk RING_FROM_CHUNK, and on a 'tokens'
     model the write IS Z_tok (the probe reads the columns the write reads).

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/test_z3_registers.py -q
"""
from __future__ import annotations

import os
import re
import random
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diag_z_registers as Z  # noqa: E402
from prepare_carry_task import NOT_ANSWER, make_row  # noqa: E402

# Piece ids inside the tiny test model's vocabulary (256; EOS = 255).
PT = {"regs": list(range(10, 26)), "digits": list(range(40, 50)),
      "plus": 60, "eq": 61, "nl": 62}


def _row(seed=0, n_tokens=4097, ops=1):
    return make_row(random.Random(seed), n_tokens, PT["regs"], PT["digits"],
                    PT["plus"], PT["eq"], PT["nl"], ops_per_line=ops)


# ─── 1. the replay ──────────────────────────────────────────────────────────

class TestReplay:
    @pytest.mark.parametrize("ops", [1, 2, 3])
    def test_answers_match_the_generator(self, ops):
        ids, dep = _row(seed=ops, ops=ops)
        ups = Z.replay(ids, PT)
        Z.check_answer_positions(ups, dep, NOT_ANSWER)       # raises on mismatch
        # one line = REG (+ d) x ops = v NL = 4 + 2*ops tokens
        assert len(ups) >= 4097 // (4 + 2 * ops) - 2
        for p, r, v in ups:
            assert ids[p] == PT["digits"][v] and ids[p - 1] == PT["eq"]

    def test_the_last_value_per_register_is_the_last_update(self):
        ids, _ = _row(seed=5)
        last = {}
        for p, r, v in Z.replay(ids, PT):
            last[r] = v
        # Independent recomputation: walk lines, sum digits mod 10.
        val = [0] * 16
        i = 0
        while i < len(ids):
            if ids[i] in PT["regs"] and (i == 0 or ids[i - 1] == PT["nl"]):
                r = PT["regs"].index(ids[i])
                j, s = i + 1, 0
                while j + 1 < len(ids) and ids[j] == PT["plus"]:
                    s += PT["digits"].index(ids[j + 1])
                    j += 2
                if j + 1 < len(ids) and ids[j] == PT["eq"]:
                    val[r] = (val[r] + s) % 10
                i = j
            i += 1
        for r, v in last.items():
            assert val[r] == v

    def test_a_wrong_answer_is_refused(self):
        ids, _ = _row(seed=2)
        ups = Z.replay(ids, PT)
        p = ups[3][0]
        bad = list(ids)
        bad[p] = PT["digits"][(PT["digits"].index(ids[p]) + 1) % 10]
        with pytest.raises(ValueError, match="line grammar"):
            Z.replay(bad, PT)

    def test_misaligned_answer_dep_is_refused(self):
        ids, dep = _row(seed=3)
        dep = [NOT_ANSWER] + dep[:-1]                        # shifted by one
        with pytest.raises(ValueError, match="answer_dep"):
            Z.check_answer_positions(Z.replay(ids, PT), dep, NOT_ANSWER)


# ─── 2. targets at the chunk end ────────────────────────────────────────────

class TestTargets:
    def test_values_and_categories(self):
        L, tok_rows = 10, 3
        ups = [(2, 0, 4), (8, 1, 7), (12, 0, 5), (18, 2, 9), (27, 3, 1)]
        t = Z.chunk_targets(ups, n_chunks=3, L=L, tok_rows=tok_rows)
        (v0, c0), (v1, c1), (v2, c2) = t
        assert v0[:4] == [4, 7, 0, 0] and c0[:4] == ["chunk", "recent", "never", "never"]
        # chunk 1 = positions 10..19; reg 2 at 18 (>= 20-3) is recent, reg 0 at 12 is chunk
        assert v1[:4] == [5, 7, 9, 0] and c1[:4] == ["chunk", "older", "recent", "never"]
        assert v2[:4] == [5, 7, 9, 1] and c2[:4] == ["older", "older", "older", "recent"]

    def test_an_update_at_the_boundary_belongs_to_the_next_chunk(self):
        t = Z.chunk_targets([(10, 0, 3)], n_chunks=2, L=10, tok_rows=2)
        assert t[0][0][0] == 0 and t[0][1][0] == "never"
        assert t[1][0][0] == 3 and t[1][1][0] == "chunk"


# ─── 3. the probe ───────────────────────────────────────────────────────────

def _planted(n_rows=100, chunks=4, signal=1.0, seed=0, d=512):
    g = torch.Generator().manual_seed(seed)
    n = n_rows * chunks
    Y = torch.randint(0, 10, (n, Z.N_REGS), generator=g)
    rows = torch.arange(n_rows).repeat_interleave(chunks)
    code = torch.nn.functional.one_hot(Y, 10).reshape(n, -1).float()
    proj = torch.randn(code.shape[1], d, generator=g)
    X = signal * (code @ proj) + torch.randn(n, d, generator=g)
    return X, Y, rows


class TestProbe:
    def test_rows_never_straddle_a_split(self):
        rows = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        f = Z.row_folds(rows, 3)
        for r in range(5):
            assert len(set(f[rows == r].tolist())) == 1

    def test_a_planted_code_is_decoded(self):
        X, Y, rows = _planted(signal=1.0)
        p = Z.probe(X, Y, rows, torch.device("cpu"), n_folds=5)
        acc = float((p["pred"] == Y).double().mean())
        assert acc > 0.9 and (p["pred"] >= 0).all()

    def test_noise_stays_at_the_baseline(self):
        X, Y, rows = _planted(signal=0.0)
        p = Z.probe(X, Y, rows, torch.device("cpu"), n_folds=5)
        mask = torch.ones_like(Y, dtype=torch.bool)
        m = Z.boot_margin(p["pred"] == Y, p["maj"] == Y, rows, mask, 500, 0)
        assert m["lo"] <= 0 <= m["hi"] + 0.02
        assert abs(m["margin"]) < 0.05

    def test_boot_margin_arithmetic_and_empty_mask(self):
        c = torch.tensor([[True, False], [True, True]])
        b = torch.tensor([[False, False], [True, False]])
        rows = torch.tensor([0, 1])
        m = Z.boot_margin(c, b, rows, torch.ones_like(c), 200, 0)
        assert m["acc"] == 0.75 and m["baseline"] == 0.25 and m["margin"] == 0.5
        assert Z.boot_margin(c, b, rows, torch.zeros_like(c), 200, 0) is None


# ─── the cuts ───────────────────────────────────────────────────────────────

def _m(margin, lo):
    return {"margin": margin, "lo": lo, "hi": margin + 0.05, "acc": 0, "baseline": 0}


class TestTheCuts:
    def test_status(self):
        e = _m(0.40, 0.35)
        assert Z.status(_m(0.25, 0.20), e) == "HOLDS"
        assert Z.status(_m(0.10, 0.05), e) == "PARTIAL"
        assert Z.status(_m(0.02, -0.01), e) == "NONE"

    def test_health(self):
        assert Z.healthy(_m(0.40, 0.35))
        assert not Z.healthy(_m(0.10, 0.05))          # under HEALTH_MARGIN
        assert not Z.healthy(_m(0.20, -0.01))         # CI crosses zero

    @pytest.mark.parametrize("e, sw, se, st, want", [
        (0.40, "HOLDS", "HOLDS", "NONE", "Z_WRITE_HOLDS"),
        (0.40, "NONE", "HOLDS", "NONE", "ONLY_Z_END_HOLDS"),
        (0.40, "PARTIAL", "PARTIAL", "HOLDS", "ONLY_Z_TOK_HOLDS"),
        (0.40, "PARTIAL", "NONE", "NONE", "NO_LATENT_HOLDS"),
        (0.05, "HOLDS", "HOLDS", "HOLDS", "PROBE_BROKEN"),
    ])
    def test_reading(self, e, sw, se, st, want):
        rep = {"write": {"E": {"all": _m(e, e - 0.03)},
                         "Z_write": {"status": sw}, "Z_end": {"status": se},
                         "Z_tok": {"status": st}}}
        assert Z.reading(rep) == want
        assert want in Z.TRIGGERS


# ─── 3b. probe (b), the redundancy probe (j4_prereg.md S4.6) ────────────────

def _planted_pair(kind, n_rows=100, chunks=4, seed=0, d=128, signal=1.0):
    """E carries a planted register code.  `kind` decides what Z_end is:

      recode  Z_end is a fixed linear re-coding of E, E @ W, and nothing else.
              This is EXACTLY S4.6's objection ("the same conclusion twice in
              two coordinate systems") and must read REDUNDANT.
      extra   the same code plus a second, independent code the probe can only
              get from Z_end -- must read ADDITIONAL.
      noisy   an independent noisy view of the same code.  Reads ADDITIONAL,
              and that is CORRECT: two noisy views of one signal do carry more
              than one.  It cannot arise between E and Z_end -- E is coda+ln_f
              applied to Z_end at the same columns, a DETERMINISTIC map, so E
              holds no noise Z_end does not -- but it is pinned here so nobody
              later reads "ADDITIONAL" as "statistically independent content".
      noise   no signal at all -- Z_end is at chance, so Z_END_EMPTY.
    """
    g = torch.Generator().manual_seed(seed)
    n = n_rows * chunks
    Y = torch.randint(0, 10, (n, Z.N_REGS), generator=g)
    rows = torch.arange(n_rows).repeat_interleave(chunks)
    code = torch.nn.functional.one_hot(Y, 10).reshape(n, -1).float()
    pe = torch.randn(code.shape[1], d, generator=g)
    E = signal * (code @ pe) + torch.randn(n, d, generator=g)
    if kind == "recode":
        W = torch.randn(d, d, generator=g) / d ** 0.5
        Zf = E @ W
    elif kind == "noisy":
        pz = torch.randn(code.shape[1], d, generator=g)
        Zf = signal * (code @ pz) + torch.randn(n, d, generator=g)
    elif kind == "extra":
        # Half the registers are carried ONLY by Z_end.
        half = code.reshape(n, Z.N_REGS, 10).clone()
        half[:, : Z.N_REGS // 2] = 0
        pz = torch.randn(code.shape[1], d, generator=g)
        Zf = signal * (half.reshape(n, -1) @ pz) + torch.randn(n, d, generator=g)
    elif kind == "noise":
        Zf = torch.randn(n, d, generator=g)
    else:
        raise AssertionError(kind)
    return E, Zf, Y, rows


def _run_b(kind, **kw):
    E, Zf, Y, rows = _planted_pair(kind, **kw)
    dev = torch.device("cpu")
    samples = [{"row": int(rows[i]), "values": Y[i].tolist(),
                "cats": ["recent"] * Z.N_REGS,
                "E": E[i], "Z_end": Zf[i]} for i in range(len(Y))]
    return Z.report_incremental(samples, dev, 5, 500, 0)


class TestProbeB:
    def test_the_residual_kernel_matches_feature_space(self):
        """The Gram-space residual is not an approximation: with few enough
        dims the coefficient matrix can be formed, and the two must agree."""
        torch.manual_seed(0)
        dev = torch.device("cpu")
        n, dc, dx = 40, 7, 5
        C, X = torch.randn(n, dc), torch.randn(n, dx)
        tr = torch.zeros(n, dtype=torch.bool)
        tr[:27] = True
        Xz, Cz = Z._zs(X, tr, dev).double(), Z._zs(C, tr, dev).double()
        ti = torch.nonzero(tr)[:, 0]
        Gc_tt = (Cz @ Cz.T).index_select(0, ti).index_select(1, ti)
        lam = Z.RESID_LAM_REL * Gc_tt.diagonal().mean()
        I = torch.eye(len(ti), dtype=torch.float64)
        B = Cz[tr].T @ torch.linalg.inv(Cz[tr] @ Cz[tr].T + lam * I) @ Xz[tr]
        R = Xz - Cz @ B
        want = (R @ R.T) / dx
        got = Z._resid_kernel(X, C, tr, dev).double()
        assert (got - want).abs().max() < 1e-3 * want.abs().max()

    def test_the_fit_uses_training_rows_only(self):
        """Scrambling the TEST rows' features must not move the train block:
        that is the whole no-leakage claim."""
        torch.manual_seed(0)
        dev = torch.device("cpu")
        n, d = 40, 5
        C, X = torch.randn(n, 7), torch.randn(n, d)
        tr = torch.zeros(n, dtype=torch.bool)
        tr[:27] = True
        ti = torch.nonzero(tr)[:, 0]
        blk = lambda K: K.index_select(0, ti).index_select(1, ti)
        X2 = X.clone()
        X2[~tr] = torch.randn(int((~tr).sum()), d) * 5
        a = blk(Z._resid_kernel(X, C, tr, dev))
        b = blk(Z._resid_kernel(X2, C, tr, dev))
        assert torch.equal(a, b)

    def test_a_perfect_copy_residualises_to_nothing(self):
        dev = torch.device("cpu")
        torch.manual_seed(0)
        C = torch.randn(40, 7)
        tr = torch.zeros(40, dtype=torch.bool)
        tr[:27] = True
        K = Z._resid_kernel(C, C, tr, dev, k=7)
        plain = Z._pca_kernel(C, tr, dev, 7)
        assert float(K.diagonal().mean()) < 1e-6 * float(plain.diagonal().mean())

    def test_kernel_is_unchanged_for_the_D3_path(self):
        """covar=None must be the D3 code path, or the committed D3 numbers
        stop being reproducible."""
        X, Y, rows = _planted(signal=1.0, d=64, n_rows=40)
        dev = torch.device("cpu")
        a = Z.probe(X, Y, rows, dev, n_folds=5)
        b = Z.probe(X, Y, rows, dev, n_folds=5, covar=None)
        assert torch.equal(a["pred"], b["pred"])

    def test_the_pca_residual_cancels_a_linear_recoding(self):
        """A projection, not a shrinkage: if X lies in C's retained subspace the
        residual is ~0, so a re-coded copy cannot decode."""
        torch.manual_seed(0)
        dev = torch.device("cpu")
        n, d = 60, 12
        C = torch.randn(n, d)
        W = torch.randn(d, d) / d ** 0.5
        tr = torch.zeros(n, dtype=torch.bool)
        tr[:40] = True
        K = Z._resid_kernel(C @ W, C, tr, dev, k=d)
        assert float(K.diagonal().mean()) < 1e-6 * float(
            Z._pca_kernel(C @ W, tr, dev, d).diagonal().mean())

    def test_the_pca_reduction_is_lossless_at_full_rank(self):
        """The reduction is a rotation, and the linear kernel is invariant to
        one: at k >= the features' rank the PCA probe must reproduce the full
        probe exactly.  So E_pca below E's margin means k was too small -- which
        is what pca_retains_E reports -- and never an artefact of reducing."""
        dev = torch.device("cpu")
        X, Y, rows = _planted(signal=1.0, d=64, n_rows=60)   # rank 64 < n_tr
        full = Z.probe(X, Y, rows, dev, n_folds=5)
        red = Z.probe(X, Y, rows, dev, n_folds=5, pca_k=64)
        assert float((full["pred"] == red["pred"]).double().mean()) > 0.99

    def test_too_few_components_lose_signal_and_the_control_says_so(self):
        dev = torch.device("cpu")
        X, Y, rows = _planted(signal=1.0, d=512, n_rows=60)
        full = float((Z.probe(X, Y, rows, dev, n_folds=5)["pred"] == Y).double().mean())
        few = float((Z.probe(X, Y, rows, dev, n_folds=5, pca_k=16)["pred"] == Y).double().mean())
        assert few < full - 0.1

    def test_resid_frac_separates_a_recoding_from_new_directions(self):
        """The residual's honest quantity: a linear re-coding keeps ~none of its
        own variance, independent features keep most of it."""
        torch.manual_seed(0)
        dev = torch.device("cpu")
        n, d = 200, 32
        rows = torch.arange(50).repeat_interleave(4)
        C = torch.randn(n, d)
        W = torch.randn(d, d) / d ** 0.5
        recode = Z.resid_frac(C @ W, C, rows, dev, 5, d)
        indep = Z.resid_frac(torch.randn(n, d), C, rows, dev, 5, d)
        assert recode < Z.RESID_FRAC_MIN
        assert indep > 0.5

    def test_a_recoded_copy_reads_redundant(self):
        """S4.6's objection, planted: Z_end carries E's content in another
        coordinate system and nothing more.  It decodes well on its own, and
        adds nothing over E."""
        inc = _run_b("recode")
        assert inc["increment"]["all"]["lo"] <= 0
        # the residual keeps no direction of its own, so it is not readable --
        # which is the sharpest REDUNDANT there is
        assert inc["resid_frac"] < Z.RESID_FRAC_MIN
        assert not inc["resid_readable"]
        # ... and E is not the problem: it decodes, before and after reduction
        assert inc["E"]["all"]["margin"] > 0.3 and inc["pca_retains_E"]

    def test_an_independently_noisy_view_reads_additional(self):
        """STATED LIMIT, not a defect.  Two independent noisy views of one code
        do carry more than one, so this reads ADDITIONAL.  Between E and Z_end
        it cannot happen (E is a deterministic function of Z_end), which is why
        ADDITIONAL there means "the coda made content linearly inaccessible
        that Z_end still exposes", not "independent content"."""
        inc = _run_b("noisy")
        assert inc["increment"]["all"]["lo"] > 0

    def test_independent_content_reads_additional(self):
        inc = _run_b("extra")
        assert inc["increment"]["all"]["lo"] > 0
        assert inc["increment_share"] > 0
        # here the residual IS readable, and it agrees
        assert inc["resid_frac"] > Z.RESID_FRAC_MIN and inc["resid_readable"]
        assert inc["Z_end_resid_E"]["all"]["lo"] > 0
        assert not inc["residual_disagrees"]

    def test_noise_adds_nothing(self):
        inc = _run_b("noise")
        assert inc["increment"]["all"]["lo"] <= 0

    def test_boot_delta_arithmetic_and_empty_mask(self):
        a = torch.tensor([[True, True], [True, False]])
        b = torch.tensor([[True, False], [False, False]])
        rows = torch.tensor([0, 1])
        d = Z.boot_delta(a, b, rows, torch.ones_like(a), 200, 0)
        assert d["delta"] == 0.5 and d["cells"] == 4
        assert Z.boot_delta(a, b, rows, torch.zeros_like(a), 200, 0) is None

    def test_boot_delta_is_paired(self):
        """Identical encodings have a ZERO-WIDTH interval when paired; two
        unpaired bootstraps of the same margin would not."""
        torch.manual_seed(0)
        c = torch.rand(60, 4) > 0.5
        rows = torch.arange(20).repeat_interleave(3)
        d = Z.boot_delta(c, c, rows, torch.ones_like(c), 300, 0)
        assert d["delta"] == 0 and d["lo"] == 0 and d["hi"] == 0

    @pytest.mark.parametrize("e, zend_lo, inc_lo, res_lo, want", [
        (0.40, 0.20, +0.02, +0.02, "ADDITIONAL"),
        (0.40, 0.20, -0.01, -0.01, "REDUNDANT"),
        # the RESIDUAL never moves the label -- the increment is registered
        (0.40, 0.20, +0.02, -0.01, "ADDITIONAL"),
        (0.40, 0.20, -0.01, +0.02, "REDUNDANT"),
        (0.40, -0.01, +0.02, +0.02, "Z_END_EMPTY"),
        (0.05, 0.20, +0.02, +0.02, "PROBE_BROKEN"),
    ])
    def test_redundancy_reading(self, e, zend_lo, inc_lo, res_lo, want):
        rep = {"write": {"E": {"all": _m(e, e - 0.03)},
                         "Z_end": {"all": _m(0.20, zend_lo)}},
               "incremental": {"increment": {"all": {"delta": inc_lo + 0.01,
                                                     "lo": inc_lo, "hi": 0.1}},
                               "Z_end_resid_E": {"all": _m(0.05, res_lo)}}}
        assert Z.redundancy(rep) == want
        assert want in Z.TRIGGERS_B

    def test_redundancy_is_not_run_without_the_block(self):
        assert Z.redundancy({"write": {"E": {"all": _m(0.4, 0.37)}}}) == "NOT_RUN"
        assert "NOT_RUN" in Z.TRIGGERS_B


# ─── 4. the capture, end to end on a tiny model ─────────────────────────────

NV, K, CL, D, T = 4, 16, 16, 64, 4
N_CHUNKS = 5                                  # the ring is full from chunk 4


def _model(encoding):
    from test_cortex_eval import VOCAB, _build_raven
    flags = dict(use_memory=True, memory_slots=0, accum_vecs=NV, latent_carry=True,
                 eos_token_id=VOCAB - 1, prefix_memory="gated", gate_slots=K,
                 gate_route="ring", gate_init="zero", gate_fill="grow",
                 latent_encoding=encoding, latent_tok_pool=2, latent_s0_read=False,
                 latent_read_heads=4, latent_read_znorm="rms")
    flags["latent_read"] = "scratch" if encoding == "scratch" else "xattn"
    torch.manual_seed(1234)
    return _build_raven(**flags).eval()


def _collect(encoding, n_rows=4):
    m = _model(encoding)
    rows = []
    for i in range(n_rows):
        ids, dep = _row(seed=10 + i, n_tokens=CL * N_CHUNKS + 1)
        rows.append((i, ids, dep))
    return Z.collect(m, m.cortex, rows, PT, N_CHUNKS, torch.tensor([T, 0]),
                     torch.device("cpu"), 7, tok_rows=NV * 2, tok_pool=2,
                     log_every=0)


class TestCaptureEndToEnd:
    @pytest.mark.parametrize("encoding", ["tokens", "scratch"])
    def test_every_encoding_every_chunk_and_the_ring_when_full(self, encoding):
        samples, check = _collect(encoding)
        assert len(samples) == 4 * N_CHUNKS
        for s in samples:
            for k in Z.WRITE_ENCODINGS:
                assert s[k].shape == (NV, D), k
            has_ring = s["chunk"] + 1 >= Z.RING_FROM_CHUNK
            assert ("E_ring" in s) == has_ring
            if has_ring:
                assert s["E_ring"].shape == (K, D) and s["Z_ring"].shape == (K, D)
            assert len(s["values"]) == Z.N_REGS and len(s["cats"]) == Z.N_REGS
        if encoding == "tokens":
            assert check["tokens_write_equals_z_tok"] is True
        else:
            assert check["tokens_write_equals_z_tok"] is None
            s = samples[-1]
            assert not torch.allclose(s["Z_write"].float(), s["Z_tok"].float())

    def test_the_report_runs_on_real_samples(self):
        samples, _ = _collect("tokens", n_rows=6)
        rep = {"write": Z.report_encodings(samples, Z.WRITE_ENCODINGS,
                                           torch.device("cpu"), 3, 100, 0),
               "ring": Z.report_encodings(samples, Z.RING_ENCODINGS,
                                          torch.device("cpu"), 3, 100, 0)}
        assert rep["write"]["_n"]["samples"] == 6 * N_CHUNKS
        assert rep["ring"]["_n"]["samples"] == 6 * (N_CHUNKS - Z.RING_FROM_CHUNK + 1)
        assert rep["write"]["E"]["status"] == "control"
        assert Z.reading(rep) in Z.TRIGGERS


# ─── the launcher ───────────────────────────────────────────────────────────

# ─── the cluster's Python ────────────────────────────────────────────────────

def _fstring_brace_hazards(path):
    """Lines whose f-string literal opens a replacement field it does not close.

    PEP 701 (Python 3.12) let a replacement field span physical lines and hold
    implicitly concatenated literals.  PACE runs **3.11.15**, where an f-string
    is one STRING token that must close on its own line -- so that construct is
    a SyntaxError there and the module will not even import.  A 3.13 dev box
    accepts it, and ast.parse(feature_version=(3, 11)) does NOT catch it: the
    change is in the tokenizer, which feature_version does not reach.
    """
    bad, in_doc = [], False
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        if line.count('"""') % 2:
            in_doc = not in_doc
            continue
        if in_doc:
            continue
        for m in re.finditer(r'\b(?:f|fr|rf)(["\'])', line):
            q = m.group(1)
            rest = line[m.end():]
            end = rest.find(q)
            if end < 0:                       # literal does not close on this line
                bad.append((n, line.rstrip()))
                break
            body = rest[:end].replace("{{", "").replace("}}", "")
            if body.count("{") != body.count("}"):
                bad.append((n, line.rstrip()))
                break
    return bad


class TestClusterPython:
    def test_no_312_only_fstrings_in_the_probe(self):
        for f in ("evals/diag_z_registers.py", "evals/score_j1.py",
                  "tests/test_z3_registers.py"):
            bad = _fstring_brace_hazards(os.path.join(REPO, f))
            assert not bad, f"{f}: 3.12-only f-string(s) -- PACE is 3.11: {bad}"

    def test_the_check_catches_the_bug_it_was_written_for(self):
        import tempfile
        src = ('x = 1\n'
               'print(f"a {\'Y\' if x else \'N -- the \'\n'
               '      \'rest of it\'}")\n')
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(src)
            name = fh.name
        try:
            assert _fstring_brace_hazards(name)
        finally:
            os.unlink(name)


class TestTheLauncher:
    SB = os.path.join(REPO, "pace", "diag_z_registers.sbatch")

    def _src(self):
        with open(self.SB, encoding="utf-8") as fh:
            return fh.read()

    def test_it_builds_the_read_outs_flags(self):
        s = self._src()
        assert "--set latent_read=scratch" in s
        assert "--set latent_read_gate_lr_mult=$GATE_LR_MULT" in s
        assert "GATE_LR_MULT=${GATE_LR_MULT:-0.1}" in s
        assert "--set latent_read=none --set latent_write_only=true" in s
        assert "z_gate_lr_mult" in s                  # the trained-mult guard

    def test_the_tasks_are_the_registered_four(self):
        s = self._src()
        for t in ('"j3 scratch real"', '"j1 tokens real"', '"j1 tokens noread"',
                  '"j3 scratch edrop0.25-real"'):
            assert t in s

    def test_probe_c_does_not_disturb_the_registered_four(self):
        """(c), the heal parent, is index 4 and nothing below it moves.

        A bare `sbatch pace/diag_z_registers.sbatch` must still be exactly D3:
        the default array is 0-3, the four entries keep their order (the index
        IS the task, and eval_results dirs are named from it), and the parent's
        own path comes from its own defaults rather than STEP.
        """
        s = self._src()
        assert "#SBATCH --array=0-3" in s
        tasks = s.split("TASKS=(")[1].split(")")[0].strip().splitlines()
        assert [t.strip() for t in tasks] == ['"j3 scratch real"',
                                              '"j1 tokens real"',
                                              '"j1 tokens noread"',
                                              '"j3 scratch edrop0.25-real"',
                                              '"heal tokens parent"']
        # the SLICED w16 parent every J arm branched from, not checkpoint_91552
        assert "${PARENT_CKPT:-checkpoint_91552_w16}" in s
        assert "RUN=${PARENT_RUN:-retro-b2-heal}" in s
        # and it reads NOTHING: the parent has no trained read module
        heal = s.split("heal:parent)")[1].split(";;")[0]
        assert "--set latent_read=none --set latent_write_only=true" in heal
        assert "latent_read=xattn" not in heal and "latent_read=scratch" not in heal

    def test_the_window_matches_the_write(self):
        s = self._src()
        assert "--tok_rows 64 --tok_pool \"$TOK_POOL\"" in s and "TOK_POOL=${TOK_POOL:-4}" in s

    def test_probe_b_is_opt_in_and_wired(self):
        s = self._src()
        assert "INCREMENTAL=${INCREMENTAL:-}" in s      # off by default
        assert '[ -n "$INCREMENTAL" ] && INC_ARGS="--incremental"' in s
        # the real invocation is the LAST occurrence; the earlier ones are the
        # usage comments at the top of the file
        assert "$INC_ARGS" in s.split("python evals/diag_z_registers.py")[-1]

    def test_no_var_assignment_prefix_on_the_python_line(self):
        for line in self._src().splitlines():
            if line.strip().startswith("python "):
                assert "=" not in line.split("python")[0]
