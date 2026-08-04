"""
prefix_pack(write=, read=) and the n_sum==0 unpack path (2026-08-03).

These flags exist so KV-cached generation is CORRECT, not merely faster, so the
properties worth pinning are the ones whose violation is silent:

  * the defaults are the pre-existing packing, exactly — cortex_graft.py is
    imported live from the repo root (it is NOT snapshotted into prepared
    checkpoint dirs the way the modeling file is), so a resumed training run
    picks these edits up mid-flight.  If the default path moved, B2 would
    change arrangement at its resume step and nothing in the loss curve would
    say so;
  * write=False must still strip the carried columns on unpack.  Greedy
    decoding only reads logits[:, -1] and the carry sits at the FRONT, so a
    missing strip is invisible in generation and corrupts every scoring path;
  * read=False must not splice the carry (it is already in the cache) while
    still advancing positions, or a cached decode step restarts numbering and
    silently re-uses RoPE positions the prefill already consumed.

Run: /c/Users/henry/miniconda3/envs/cortex-retro/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from test_cortex_graft import B, H, S, VOCAB, AttnRaven, _ids  # noqa: E402

NV = 4
EOS = VOCAB - 1


def _model(mode="accum", n_vec=NV, accum_max=32, **kw):
    return AttnRaven(use_memory=True, memory_slots=0, prefix_memory=mode,
                     accum_vecs=n_vec, accum_max=accum_max, eos_token_id=EOS,
                     **kw)


def _pack(model, carry=None, s=S, ids=None, **kw):
    """Run prefix_pack through the graft's own runtime setup.

    `ids` must be passed when two packs are compared to each other — drawing
    fresh ids per call would compare different token sequences and the
    assertion would fail for a reason that has nothing to do with the flags."""
    ids = _ids(B, s) if ids is None else ids
    emb = model.wte(ids)
    model.cortex.begin(carry, None, s, emb.device, emb.dtype)
    if not bool(model.cortex.prefix.summary_seeded):
        model.cortex.init_summary_from_embedding(model.wte.weight)
    pos = torch.arange(s).unsqueeze(0).expand(B, s)
    return model.cortex.prefix_pack(emb, pos, 1.0, **kw)


class TestDefaultsUnchanged:

    def test_default_layout_is_carry_tokens_slots(self):
        m = _model()
        carry = torch.randn(B, 2 * NV, H)
        packed, pos, n_pre, n_sum = _pack(m, carry)
        assert (n_pre, n_sum) == (2 * NV, NV)
        assert packed.shape[1] == 2 * NV + S + NV
        # carry at the front, verbatim; slots at the back
        assert torch.allclose(packed[:, :2 * NV], carry)
        assert pos[0, :n_pre].tolist() == [0] * n_pre
        assert pos[0, n_pre:n_pre + S].tolist() == list(range(1, S + 1))
        # summary slots CONTINUE the token numbering (prefix_pos='tail'), so the
        # write reads the chunk at positive relative offsets
        assert pos[0, -n_sum:].tolist() == list(range(S + 1, S + 1 + n_sum))

    def test_explicit_true_matches_default(self):
        """The training call site must be bit-identical under the new kwargs."""
        torch.manual_seed(0)
        m = _model()
        carry, ids = torch.randn(B, NV, H), _ids(B, S)
        a = _pack(m, carry, ids=ids)
        b = _pack(m, carry, ids=ids, write=True, read=True)
        assert a[2] == b[2] and a[3] == b[3]
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


class TestWriteFalse:

    def test_no_summary_slots_appended(self):
        m = _model()
        carry = torch.randn(B, NV, H)
        packed, pos, n_pre, n_sum = _pack(m, carry, write=False)
        assert n_sum == 0 and n_pre == NV
        assert packed.shape[1] == NV + S
        assert pos[0, -1].item() == S          # last real token, still shifted

    def test_unpack_strips_carry_when_no_slots(self):
        """The failure this catches: n_sum==0 used to short-circuit and return
        the packed tensor whole, leaving n_pre carry rows in front of the
        logits."""
        m = _model()
        carry = torch.randn(B, NV, H)
        packed, _, n_pre, n_sum = _pack(m, carry, write=False)
        real, new = m.cortex.prefix_unpack(packed, n_pre, n_sum)
        assert new is None
        assert real.shape[1] == S
        assert torch.allclose(real, packed[:, n_pre:])

    def test_slots_cannot_reach_real_tokens(self):
        """Dropping the slots must not change real-token outputs, because the
        causal mask already forbade them from being read.  Compared at the
        packed-tensor level (not logits) because a shorter packed sequence
        draws a different initialize_state — see smoke_kv_cache.py."""
        m = _model()
        carry, ids = torch.randn(B, NV, H), _ids(B, S)
        with_slots, _, p1, s1 = _pack(m, carry, ids=ids, write=True)
        no_slots, _, p2, s2 = _pack(m, carry, ids=ids, write=False)
        assert s1 == NV and s2 == 0 and p1 == p2
        assert torch.equal(with_slots[:, :p1 + S], no_slots)


class TestReadFalse:

    def test_carry_not_spliced_but_positions_advance(self):
        m = _model()
        carry = torch.randn(B, 3 * NV, H)
        packed, pos, n_pre, n_sum = _pack(m, carry, write=False, read=False,
                                          s=1)
        assert (n_pre, n_sum) == (0, 0)
        assert packed.shape[1] == 1
        # the +1 shift still applies, so a cached step continues the prefill's
        # 1..S numbering instead of restarting
        assert pos[0].tolist() == [1]

    def test_incremental_position_continues_prefill(self):
        """Prefill over S tokens ends at position S; the next token must be
        S+1, which is what passing absolute index S produces."""
        m = _model()
        _, prefill_pos, n_pre, _ = _pack(m, torch.randn(B, NV, H))
        last = prefill_pos[0, n_pre:n_pre + S][-1].item()
        emb = m.wte(_ids(B, 1))
        m.cortex.begin(None, None, 1, emb.device, emb.dtype)
        pos = torch.tensor([[S]]).expand(B, 1)
        _, step_pos, _, _ = m.cortex.prefix_pack(emb, pos, 1.0,
                                                 write=False, read=False)
        assert step_pos[0, 0].item() == last + 1

    def test_no_prefix_model_is_untouched_by_flags(self):
        """Base / control arms have no prefix; the flags must be inert so the
        same decode loop drives every arm."""
        m = AttnRaven(use_memory=False, memory_slots=0)
        assert m.cortex is None or m.cortex.prefix is None


class TestSummarySlotPositions:
    """prefix_pos — the 2026-08-04 write-position fix.

    Under the old 'zero' layout the summary slots sat at position 0 while
    physically trailing the chunk, so every summary->token RoPE offset was
    NEGATIVE (-1..-S), which a causal LM never sees (pos_q >= pos_k always).
    A wrong layout here trains perfectly happily and shows nothing in the loss.
    """

    def test_tail_offsets_to_real_tokens_are_all_positive(self):
        m = _model()
        _, pos, n_pre, n_sum = _pack(m, torch.randn(B, NV, H))
        tok, summ = pos[0, n_pre:n_pre + S], pos[0, -n_sum:]
        assert int((summ.unsqueeze(1) - tok.unsqueeze(0)).min()) > 0

    def test_legacy_zero_layout_is_reproducible(self):
        """The cancelled B2 phase-0 arrangement must stay reachable, or those
        weights become uninterpretable."""
        m = _model(prefix_pos="zero")
        _, pos, n_pre, n_sum = _pack(m, torch.randn(B, NV, H))
        assert pos[0, -n_sum:].tolist() == [0] * n_sum
        tok = pos[0, n_pre:n_pre + S]
        assert int((pos[0, -n_sum:].unsqueeze(1) - tok.unsqueeze(0)).max()) < 0

    def test_carry_and_token_positions_are_layout_independent(self):
        """Only the slots move.  Real tokens must stay at 1..S under both, so
        the cached-decode continuation rule keeps working."""
        ids = _ids(B, S)
        a = _pack(_model(), torch.randn(B, NV, H), ids=ids)[1]
        b = _pack(_model(prefix_pos="zero"), torch.randn(B, NV, H), ids=ids)[1]
        assert torch.equal(a[:, :NV + S], b[:, :NV + S])

    def test_write_false_leaves_positions_alone(self):
        """No slots to place during cached decode, so the layouts coincide."""
        ids = _ids(B, S)
        a = _pack(_model(), torch.randn(B, NV, H), ids=ids, write=False)[1]
        b = _pack(_model(prefix_pos="zero"), torch.randn(B, NV, H), ids=ids,
                  write=False)[1]
        assert torch.equal(a, b)

    def test_bad_layout_name_raises(self):
        try:
            _model(prefix_pos="middle")
        except ValueError as e:
            assert "prefix_pos" in str(e)
        else:
            raise AssertionError("an unknown prefix_pos must not be ignored")


class TestEosCarryReset:
    """prefix_eos_reset — the 2026-08-04 read fix.

    write_reset is has_eos for the WHOLE chunk, so the old unconditional reset
    zeroed the carry for every position whenever a document boundary appeared
    anywhere in the chunk.  On EOS-separated packed data that is most chunks.
    """

    def _carry_norm(self, m, eos_at):
        carry = torch.randn(B, 2 * NV, H)
        ids = _ids(B, S)
        eos = torch.zeros(B, S, dtype=torch.bool)
        if eos_at is not None:
            eos[:, eos_at] = True
        emb = m.wte(ids)
        m.cortex.begin(carry, eos, S, emb.device, emb.dtype)
        if not bool(m.cortex.prefix.summary_seeded):
            m.cortex.init_summary_from_embedding(m.wte.weight)
        pos = torch.arange(S).unsqueeze(0).expand(B, S)
        packed, _, n_pre, _ = m.cortex.prefix_pack(emb, pos, 1.0)
        return float(packed[:, :n_pre].detach().norm()), n_pre

    def test_default_keeps_the_carry_across_a_boundary(self):
        m = _model()
        live, n_pre = self._carry_norm(m, eos_at=S // 2)
        assert n_pre == 2 * NV and live > 1e-6

    def test_opt_in_reset_still_zeroes(self):
        m = _model(prefix_eos_reset=True)
        dead, n_pre = self._carry_norm(m, eos_at=S // 2)
        assert n_pre == 2 * NV and dead == 0.0

    def test_reset_fires_wherever_the_boundary_is(self):
        """The old behaviour was position-independent — an EOS at the very last
        token killed the carry for the whole chunk just as thoroughly."""
        m = _model(prefix_eos_reset=True)
        assert self._carry_norm(m, eos_at=S - 1)[0] == 0.0
        assert self._carry_norm(m, eos_at=0)[0] == 0.0

    def test_no_eos_is_unaffected_either_way(self):
        assert self._carry_norm(_model(), None)[0] > 1e-6
        assert self._carry_norm(_model(prefix_eos_reset=True), None)[0] > 1e-6

    def test_write_side_masking_is_untouched(self):
        """pool_mask / valid_write still restrict the WRITE to the open
        document's suffix — the fix is read-side only."""
        m = _model()
        emb = m.wte(_ids(B, S))
        eos = torch.zeros(B, S, dtype=torch.bool)
        eos[:, S - 1] = True                      # nothing open after the EOS
        m.cortex.begin(torch.randn(B, NV, H), eos, S, emb.device, emb.dtype)
        assert m.cortex._valid_write is not None
        assert not bool(m.cortex._valid_write.any())
