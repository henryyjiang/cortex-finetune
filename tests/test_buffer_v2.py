"""PrefixGatedBufferV2 (v2_buffer_handoff.md B1): hard keep/replace."""
from __future__ import annotations

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cortex_memory.buffers import PrefixGatedBuffer, PrefixGatedBufferV2  # noqa: E402

D, W, K = 8, 4, 16
L = K // W


def _v2(pairing="paired", latent=True, random_heads=False, seed=0):
    torch.manual_seed(seed)
    b = PrefixGatedBufferV2(D, n_vec=W, n_slots=K, carries_latent=latent,
                            pairing=pairing)
    if random_heads:
        for h in b.score_heads:
            torch.nn.init.normal_(h.weight, std=1.0)
    return b


def _writes(n, latent=True, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(2, W, D, generator=g),
             torch.randn(2, W, D, generator=g) if latent else None) for _ in range(n)]


def _fifo_ring(latent=True):
    ring = PrefixGatedBuffer(D, n_vec=W, n_slots=K, route="ring", fill="grow",
                             carries_latent=latent)
    ring.gate = lambda s, c: (c, None, None)          # fg = 0, ig = 1
    ring.gate_latent = lambda s, c: (c, None, None)
    return ring


@pytest.mark.parametrize("pairing,latent", [("paired", True), ("independent", True),
                                            ("paired", False)])
def test_fifo_at_init_is_the_ring_with_fg0_ig1_up_to_block_rotation(pairing, latent):
    v2, ring = _v2(pairing, latent), _fifo_ring(latent)
    sv = sr = None
    for n, (e, z) in enumerate(_writes(13, latent), start=1):
        sv, sr = v2.merge(sv, e, z), ring.merge(sr, e, z)
        if n >= L:
            # the ring's newest block is (n-1) mod L; v2 keeps age order
            order = [(n + i) % L for i in range(L)]
            rot = sr.view(2, L, W, -1)[:, order].reshape(sr.shape)
            assert torch.equal(sv, rot), f"chunk {n}"
            if n > L:
                assert all(float(r) == 1.0 for r in v2.last_replace_rate)
        else:
            assert torch.equal(sv, sr)


@pytest.mark.parametrize("pairing", ["paired", "independent"])
def test_every_row_is_a_write_bit_for_bit_and_stays_in_its_slot_class(pairing):
    v2 = _v2(pairing, random_heads=True)
    st, seen, rates = None, [], []
    for n, (e, z) in enumerate(_writes(40, seed=3)):
        st = v2.merge(st, e, z)
        seen.append(torch.cat([e, z], -1))
        if n >= L:
            rates += [float(r) for r in v2.last_replace_rate]
    assert min(rates) < 1.0, "random heads must drop some writes, or only FIFO is tested"
    writes = torch.stack(seen, 1)                      # [B, n, W, 2D]
    for b in range(2):
        for r in range(K):
            for ch in (slice(0, D), slice(D, 2 * D)):
                hits = (writes[b, :, :, ch] == st[b, r, ch]).all(-1).nonzero()
                assert len(hits) == 1, f"row {r} is not exactly one write"
                assert int(hits[0, 1]) == r % W, "a row left its slot class"
            if pairing == "paired":
                pair = (writes[b] == st[b, r]).all(-1).nonzero()
                assert len(pair) == 1, f"row {r}: E and Z came from different writes"


def test_independent_channels_choose_only_on_their_own_head():
    a = _v2("independent", random_heads=True, seed=0)
    b = _v2("independent", random_heads=True, seed=0)
    torch.nn.init.normal_(b.score_heads[1].weight, std=5.0)   # a different Z head
    sa = sb = None
    for e, z in _writes(30, seed=4):
        sa, sb = a.merge(sa, e, z), b.merge(sb, e, z)
    assert torch.equal(sa[..., :D], sb[..., :D])
    assert not torch.equal(sa[..., D:], sb[..., D:])


def test_paired_can_still_change_with_either_half():
    a, b = _v2("paired", random_heads=True), _v2("paired", random_heads=True)
    with torch.no_grad():
        b.score_heads[0].weight[0, D:2 * D] += 3.0            # weight on the Z half
    sa = sb = None
    for e, z in _writes(30, seed=5):
        sa, sb = a.merge(sa, e, z), b.merge(sb, e, z)
    assert not torch.equal(sa, sb)


def test_the_score_gets_a_gradient_at_init():
    v2 = _v2()
    st = None
    for e, z in _writes(L, seed=6):
        st = v2.merge(st, e, z)
    e, z = _writes(1, seed=7)[0]
    out = v2.merge(st, e.requires_grad_(), z)
    (out * torch.randn_like(out)).sum().backward()
    assert v2.score_heads[0].weight.grad.abs().sum() > 0
    assert v2.age_prior.grad.abs().sum() > 0


def test_stored_rows_stay_bounded_over_200_chunks():
    v2 = _v2(random_heads=True)
    st, peak = None, 0.0
    for e, z in _writes(200, seed=8):
        st = v2.merge(st, e, z)
        peak = max(peak, float(torch.cat([e, z], -1).norm(dim=-1).max()))
    assert float(st.detach().norm(dim=-1).max()) <= peak


def test_apply_gate_init_resets_the_score_and_the_age_prior():
    v2 = _v2(random_heads=True)
    with torch.no_grad():
        v2.age_prior.fill_(-3.0)
    v2.apply_gate_init()
    assert all(float(h.weight.abs().sum()) == 0 for h in v2.score_heads)
    assert torch.equal(v2.age_prior, torch.full_like(v2.age_prior, v2.age_prior_init))


def test_v2_refuses_geometries_it_cannot_keep_in_order():
    with pytest.raises(ValueError, match="gate_route=ring"):
        PrefixGatedBufferV2(D, n_vec=W, n_slots=K, route="mix")
    with pytest.raises(ValueError, match="v2_pairing"):
        PrefixGatedBufferV2(D, n_vec=W, n_slots=K, pairing="loose")


# -- through the graft, on a tiny raven -----------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _raven(**kw):
    from test_cortex_eval import VOCAB, _build_raven
    torch.manual_seed(1234)
    flags = dict(use_memory=True, memory_slots=0, accum_vecs=4, latent_carry=True,
                 eos_token_id=VOCAB - 1, prefix_memory="gated", gate_slots=16,
                 gate_route="ring", gate_init="zero", gate_fill="grow")
    flags.update(kw)
    return _build_raven(**flags), VOCAB


def _chain(m, vocab, n=10, grad=False):
    torch.manual_seed(0)
    ids = torch.randint(0, vocab - 2, (2, 16 * n))
    st, loss = None, 0.0
    for x in torch.chunk(ids, n, dim=1):
        out = m(input_ids=x.contiguous(), labels=x.contiguous(),
                num_steps=torch.tensor([2, 1 if grad else 0]),
                m_cross_in=st, return_m_cross=True)
        st, loss = out["m_cross"], loss + out["loss"]
    return st, loss


def test_graft_builds_v2_and_a_gated_checkpoint_loads_into_it():
    gated, vocab = _raven()
    v2, _ = _raven(prefix_memory="gated_v2", v2_pairing="independent")
    assert type(v2.cortex.prefix).__name__ == "PrefixGatedBufferV2"
    assert v2.cortex.prefix.pairing == "independent"
    missing, unexpected = v2.load_state_dict(gated.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith("cortex.prefix.") and ("score_heads" in k or "age_prior" in k)
               for k in missing), missing


def test_v2_chain_trains_and_the_reset_reapplies_its_init():
    from cortex_graft import reset_cortex_graft_init
    m, vocab = _raven(prefix_memory="gated_v2")
    m.train()
    st, loss = _chain(m, vocab, grad=True)
    loss.backward()
    buf = m.cortex.prefix
    assert st.shape[1] == 16 and torch.isfinite(loss)
    assert buf.score_heads[0].weight.grad is not None
    with torch.no_grad():
        buf.score_heads[0].weight.fill_(1.0)
        buf.age_prior.fill_(9.0)
    reset_cortex_graft_init(m)
    assert float(buf.score_heads[0].weight.abs().sum()) == 0
    assert float(buf.age_prior[0]) == buf.age_prior_init


def test_z_store_norm_target_pins_the_stored_z_rows():
    m, vocab = _raven(z_store_norm_target=3.0)
    st, _ = _chain(m.eval(), vocab, n=6)
    D = m.cortex.prefix.hidden_size
    zn = st[0, :, D:].float().norm(dim=-1)
    assert torch.allclose(zn, torch.full_like(zn, 3.0), rtol=1e-4)
    assert float(st[0, :, :D].float().norm(dim=-1).std()) > 0   # E untouched


# -- B3: WarmCarry ----------------------------------------------------------------

def test_warm_carry_restores_the_pointer_a_side_forward_reset():
    from recipe_utils import WarmCarry
    ring = PrefixGatedBuffer(D, n_vec=W, n_slots=K, route="ring", fill="grow",
                             carries_latent=True)
    wc = WarmCarry(0.99, max_chunks=40, n_chunks=6, seed=0)
    wc.rng.random = lambda: 0.0                         # always warm when allowed
    assert wc.start(ring, 2) is None                    # nothing to carry yet
    st = None
    for e, z in _writes(6):
        st = ring.merge(st, e, z)
    wc.end(ring, st)
    ptr = ring._chunk
    ring.merge(None, *_writes(1, seed=9)[0])            # a side forward: pointer -> 1
    assert ring._chunk != ptr
    got = wc.start(ring, 2)
    assert torch.equal(got, st) and ring._chunk == ptr and not got.requires_grad
    assert wc.start(ring, 3) is None                    # batch size changed: cold


def test_warm_carry_resets_past_max_chunks_and_counts_the_share():
    from recipe_utils import WarmCarry
    ring = PrefixGatedBuffer(D, n_vec=W, n_slots=K, route="ring", fill="grow",
                             carries_latent=True)
    wc = WarmCarry(0.99, max_chunks=16, n_chunks=8, seed=0)
    wc.rng.random = lambda: 0.0
    depths = []
    for _ in range(7):
        m = wc.start(ring, 2)
        depths.append(wc.depth)
        st = m
        for e, z in _writes(8):
            st = ring.merge(st, e, z)
        wc.end(ring, st)
    assert depths == [0, 8, 16, 0, 8, 16, 0]
    s = wc.stats()
    assert s["warm_share"] == pytest.approx(4 / 7)
    assert s["warm_mean_depth"] == pytest.approx(48 / 7)
    with pytest.raises(ValueError):
        WarmCarry(1.0, 56, 8)


def test_a_small_age_prior_is_still_fifo_at_init_and_the_spread_is_logged():
    small = PrefixGatedBufferV2(D, n_vec=W, n_slots=K, carries_latent=True,
                                age_prior_init=0.1)
    ref = _v2()
    sa = sb = None
    for e, z in _writes(9, seed=11):
        sa, sb = small.merge(sa, e, z), ref.merge(sb, e, z)
    assert torch.equal(sa, sb)
    assert float(small.age_prior[0]) == pytest.approx(0.1)
    assert [float(x) for x in small.last_learned_over_age] == [0.0]   # zero head
    big = _v2(random_heads=True)
    st = None
    for e, z in _writes(9, seed=11):
        st = big.merge(st, e, z)
    assert float(big.last_learned_over_age[0]) > 0
    with pytest.raises(ValueError, match="v2_age_prior"):
        PrefixGatedBufferV2(D, n_vec=W, n_slots=K, age_prior_init=0.0)
