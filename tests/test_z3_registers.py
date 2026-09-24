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

    def test_the_window_matches_the_write(self):
        s = self._src()
        assert "--tok_rows 64 --tok_pool \"$TOK_POOL\"" in s and "TOK_POOL=${TOK_POOL:-4}" in s

    def test_no_var_assignment_prefix_on_the_python_line(self):
        for line in self._src().splitlines():
            if line.strip().startswith("python "):
                assert "=" not in line.split("python")[0]
