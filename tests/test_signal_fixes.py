"""
Tests for the 2026-07-13 signal/gate fixes:

  * tools/prepare_recall_mix.splice_facts_probe — the pure token splice behind
    the recall-supervised data mix (tested directly; the tool's module level
    imports nothing heavy).
  * train.py distill_kl_loss + the distilled cross-chunk chain — train.py
    hard-fails at import on a box without the cluster env (torchdata etc.), so
    per repo convention (see test_cortex_train.py) the EXACT logic is
    replicated here and exercised against FakeRaven.  KEEP THE MIRRORS IN SYNC
    with train.py.
  * cortex.read_init_scale — the read projections get N(0, scale^2) instead of
    exact zero (mirror of reset_cortex_graft_init's _read_init).

Run: /c/Users/henry/miniconda3/envs/cortex/python.exe -m pytest tests/test_signal_fixes.py -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tests.test_cortex_graft import FakeRaven, _ids, _activate_cross_read, K, VOCAB
from tools.prepare_recall_mix import splice_facts_probe


# ---------------------------------------------------------------------------
# MIRROR of train.py distill_kl_loss — keep byte-identical to the original.
# ---------------------------------------------------------------------------
def distill_kl_loss(student_logits, teacher_logits, labels, temp=1.0):
    mask = (labels != -100)
    if not mask.any():
        return student_logits.sum() * 0.0
    s = torch.nn.functional.log_softmax(student_logits[mask].float() / temp, dim=-1)
    t = torch.nn.functional.log_softmax(teacher_logits[mask].float() / temp, dim=-1)
    kl = torch.nn.functional.kl_div(s, t, log_target=True, reduction="batchmean")
    return kl * (temp ** 2)


# ---------------------------------------------------------------------------
# splice_facts_probe
# ---------------------------------------------------------------------------
class TestSpliceFactsProbe:
    ROW = 128

    def _row(self):
        return list(range(1000, 1000 + self.ROW))

    def test_row_length_preserved(self):
        out = splice_facts_probe(self._row(), [(10, [7, 7, 7])], [9, 9])
        assert len(out) == self.ROW

    def test_probe_at_tail(self):
        probe = [901, 902, 903]
        out = splice_facts_probe(self._row(), [(10, [7, 7])], probe)
        assert out[-3:] == probe

    def test_single_fact_at_exact_position(self):
        fact = [7, 7, 7, 7]
        out = splice_facts_probe(self._row(), [(20, fact)], [9])
        assert out[20:24] == fact
        assert out[:20] == self._row()[:20]           # prefix untouched
        assert out[24:30] == self._row()[20:26]       # suffix shifted right

    def test_multi_fact_drift_equals_lower_lengths(self):
        f1, f2 = [7, 7, 7], [8, 8]
        out = splice_facts_probe(self._row(), [(10, f1), (50, f2)], [9])
        assert out[10:13] == f1                        # lowest fact: exact pos
        # higher fact drifts right by len(f1)
        assert out[50 + len(f1): 50 + len(f1) + len(f2)] == f2

    def test_no_room_raises(self):
        with pytest.raises(AssertionError):
            splice_facts_probe(self._row(), [(self.ROW - 2, [7, 7, 7])],
                               [9] * 4)


# ---------------------------------------------------------------------------
# distill_kl_loss
# ---------------------------------------------------------------------------
class TestDistillKL:

    def test_zero_for_identical_logits(self):
        logits = torch.randn(2, 8, VOCAB)
        labels = torch.randint(0, VOCAB, (2, 8))
        kl = distill_kl_loss(logits, logits.clone(), labels)
        assert float(kl) == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different_logits(self):
        kl = distill_kl_loss(torch.randn(2, 8, VOCAB), torch.randn(2, 8, VOCAB),
                             torch.randint(0, VOCAB, (2, 8)))
        assert float(kl) > 0

    def test_masked_positions_ignored(self):
        s = torch.randn(1, 8, VOCAB)
        t = s.clone()
        t[0, 3] += 100.0                              # perturb ONLY a masked position
        labels = torch.randint(0, VOCAB, (1, 8))
        labels[0, 3] = -100
        assert float(distill_kl_loss(s, t, labels)) == pytest.approx(0.0, abs=1e-6)

    def test_all_masked_returns_zero_on_graph(self):
        s = torch.randn(1, 4, VOCAB, requires_grad=True)
        kl = distill_kl_loss(s, torch.randn(1, 4, VOCAB),
                             torch.full((1, 4), -100))
        assert float(kl) == 0.0
        kl.backward()                                  # graph intact, no crash
        assert s.grad is not None

    def test_grad_flows_to_student_only(self):
        s = torch.randn(1, 4, VOCAB, requires_grad=True)
        t = torch.randn(1, 4, VOCAB, requires_grad=True)
        # train.py computes the teacher under no_grad; enforce the same contract
        distill_kl_loss(s, t.detach(), torch.randint(0, VOCAB, (1, 4))).backward()
        assert s.grad is not None and s.grad.abs().sum() > 0
        assert t.grad is None

    def test_temperature_scaling_bounded(self):
        s, t = torch.randn(1, 4, VOCAB), torch.randn(1, 4, VOCAB)
        labels = torch.randint(0, VOCAB, (1, 4))
        k1 = float(distill_kl_loss(s, t, labels, temp=1.0))
        k2 = float(distill_kl_loss(s, t, labels, temp=2.0))
        assert k2 > 0
        # temp softens both distributions; the temp^2 factor keeps the scale
        # in the same ballpark rather than collapsing quadratically
        assert k2 < k1 * 4


# ---------------------------------------------------------------------------
# Distilled cross-chunk chain (mirror of train.py cortex_fwd_bwd's new path)
# ---------------------------------------------------------------------------
def _distilled_chain_backward(student, teacher, input_ids, labels, n_chunks,
                              distill_window, distill_coeff, num_steps=(0, 3)):
    """Mirror of cortex_fwd_bwd with distillation: student runs the chunked
    carry chain; the frozen teacher runs a plain full-window forward ending at
    each chunk >= 2 boundary; KL on the chunk's supervised positions."""
    x_chunks = torch.chunk(input_ids, n_chunks, dim=1)
    y_chunks = torch.chunk(labels, n_chunks, dim=1)
    m_cross, chunk_losses, kl_terms, chunk_end = None, [], [], 0
    for gi, (xc, yc) in enumerate(zip(x_chunks, y_chunks)):
        chunk_end += xc.shape[1]
        out = student(xc, num_steps, labels=yc, m_cross_in=m_cross,
                      return_m_cross=True)
        m_cross = out.get("m_cross")
        if (yc != -100).any():
            chunk_losses.append(out["loss"])
            if gi > 0:
                t_start = max(0, chunk_end - distill_window)
                with torch.no_grad():
                    t_out = teacher(input_ids[:, t_start:chunk_end], num_steps)
                t_logits = t_out["logits"][:, -xc.shape[1]:]
                kl_terms.append(distill_kl_loss(out["logits"], t_logits, yc))
    total = torch.stack(chunk_losses).mean()
    objective = total
    if kl_terms:
        objective = objective + distill_coeff * torch.stack(kl_terms).mean()
    objective.backward()
    return total, kl_terms


class TestDistilledChain:
    N_CHUNKS = 4
    SEQ = 32                                          # 4 chunks x 8

    def _pair(self):
        torch.manual_seed(0)
        student = FakeRaven(use_memory=True, memory_slots=K)
        _activate_cross_read(student)                 # nonzero read: full pathway live
        teacher = FakeRaven()                         # plain, no memory
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return student, teacher

    def test_kl_terms_on_later_chunks_only(self):
        student, teacher = self._pair()
        ids = _ids(s=self.SEQ)
        _, kl_terms = _distilled_chain_backward(
            student, teacher, ids, ids.clone(), self.N_CHUNKS,
            distill_window=16, distill_coeff=1.0)
        assert len(kl_terms) == self.N_CHUNKS - 1     # chunk 1 never distilled
        assert all(float(k) > 0 for k in kl_terms)

    def test_teacher_gets_no_grad_student_memory_does(self):
        student, teacher = self._pair()
        ids = _ids(s=self.SEQ)
        _distilled_chain_backward(student, teacher, ids, ids.clone(),
                                  self.N_CHUNKS, 16, 1.0)
        assert all(p.grad is None for p in teacher.parameters())
        mem_grads = [p.grad for n, p in student.named_parameters()
                     if "cortex" in n and p.grad is not None]
        assert mem_grads and any(g.abs().sum() > 0 for g in mem_grads)

    def test_distill_changes_the_gradient(self):
        """The KL term must actually alter the memory-path gradient (it is a
        different signal, not a rescaled LM loss)."""
        ids = _ids(s=self.SEQ)
        grads = []
        for coeff in (0.0, 5.0):
            torch.manual_seed(0)
            student = FakeRaven(use_memory=True, memory_slots=K)
            _activate_cross_read(student)
            teacher = FakeRaven()
            for p in teacher.parameters():
                p.requires_grad_(False)
            torch.manual_seed(1)                      # identical h0 draws
            _distilled_chain_backward(student, teacher, ids, ids.clone(),
                                      self.N_CHUNKS, 16, coeff)
            g = torch.cat([p.grad.flatten() for n, p in
                           sorted(student.named_parameters()) if "cortex" in n
                           and p.grad is not None])
            grads.append(g)
        assert not torch.allclose(grads[0], grads[1])


# ---------------------------------------------------------------------------
# read_init_scale (mirror of reset_cortex_graft_init's _read_init)
# ---------------------------------------------------------------------------
def _read_init(w, read_init_scale):
    if read_init_scale > 0:
        torch.nn.init.normal_(w, std=read_init_scale)
    else:
        torch.nn.init.zeros_(w)


class TestReadInitScale:

    def test_zero_scale_is_designed_zero_init(self):
        model = FakeRaven(use_memory=True, memory_slots=K)
        w = model.cortex.m_cross.out_proj.weight
        with torch.no_grad():
            nn.init.normal_(w)                        # simulate clobber
            _read_init(w, 0.0)
        assert w.abs().sum() == 0

    def test_nonzero_scale_std_and_step0_divergence(self):
        scale = 1e-3
        model = FakeRaven(use_memory=True, memory_slots=K)
        w = model.cortex.m_cross.out_proj.weight
        with torch.no_grad():
            _read_init(w, scale)
        assert w.abs().sum() > 0
        assert float(w.std()) == pytest.approx(scale, rel=0.5)
        # the point of the knob: the write path gets nonzero gradient at step 0
        ids = _ids()
        out = model(ids, (0, 2), labels=ids.clone(), m_cross_in=None,
                    return_m_cross=True)
        out2 = model(ids, (0, 2), labels=ids.clone(),
                     m_cross_in=out["m_cross"])
        out2["loss"].backward()
        write_grads = [p.grad for n, p in model.named_parameters()
                       if "cortex" in n and "out_proj" not in n
                       and p.grad is not None]
        assert write_grads and any(g.abs().sum() > 0 for g in write_grads)
