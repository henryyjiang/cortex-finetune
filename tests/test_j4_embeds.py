"""
J4, the carried Z as its own `input_embeds` columns (cortex_memory/latent_embed.py;
findings doc "Attempt 2", design J4; j4_prereg.md): the properties the design
rests on, pinned before any of it trains.

  1. THE LAYOUT IS [E | Z | tokens | summary], and `n_prefix` (returned) is the
     WHOLE prepended width while `_n_pre` stays E's block alone.  Those two
     differ only under J4, and every consumer that needs the total gets the
     total -- prefix_unpack (via the modeling file's round-trip), the EOS mask
     lift, and `_latent_state_write`'s real-token count.  Getting this wrong is
     silent: the write would pool 64 columns of the Z block as "real tokens".
  2. Z ENTERS AT E's ROW NORM.  The s0 site's measured failure was a 0.39-norm
     row against 171; spliced as carried, Z_end would be 17x under E.  So the
     rows are rescaled first and the entering ratio is ~1.0 at init.
  3. THE NULLS ARE ZEROS AT THE SAME COLUMNS, NEVER OMISSION -- for the no-read
     LIMB, the 2x2's z_null='off' CELL, and unwritten ring rows alike.  Omitting
     them would change the sequence length, the position ids and the memory
     between limbs and between cells, and chunk 1 (which carries nothing either
     way) cannot catch it.
  4. E-DROPOUT BLANKS E AND LEAVES Z.  The E-off cell is the only cell that can
     return NO, and it needs exactly this asymmetry.
  5. THE READ IS LIVE UNDER A NO-GRAD PREFIX.  Z's columns sit in
     `input_embeds`, which `core_block_forward` re-feeds on every iteration, so
     J4 has E's gradient property and not s0's (a single no-grad step cut s0's
     read gradient to exactly zero).
  6. THE DESIGNED INIT IS AN IDENTITY and survives post_init's kaiming reset.
     Kaiming here is not a no-op but a random rotation of Z at the right scale --
     a control arm wearing the treatment's name.
  7. THE SWITCHES REFUSE THE NEIGHBOURING DESIGNS.  'embeds' beside a dead write
     encoding, beside the s0 read, or without the rescale are each a different
     experiment, so each raises.

Run: python -m pytest tests/test_j4_embeds.py -q
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_cortex_eval import VOCAB, _build_raven  # noqa: E402

from cortex_graft import CortexMemory, reset_cortex_graft_init  # noqa: E402
from cortex_memory.health import latent_runtime  # noqa: E402
from cortex_memory.latent_embed import LatentEmbedRead  # noqa: E402

W, K, CL, EOS, D = 4, 16, 16, VOCAB - 1, 64
B, S, T = 2, 12, 4
#: E's MEASURED carried row norm on the B2 checkpoint (cortex_graft.prefix_pack's
#: `_e_carried_norm`, and the number latent_read.py quotes as ||E|| = 171.0).
#: The tests use it as the rescale target because the real runs do.
E_NORM = 171.0


def _flags(limb="real", **kw):
    f = dict(use_memory=True, memory_slots=0, accum_vecs=W,
             eos_token_id=EOS,
             prefix_memory="gated", gate_slots=K, gate_route="ring",
             gate_init="zero", gate_fill="grow",
             latent_carry=True, latent_encoding="endpoint",
             latent_read="embeds", latent_s0_read=False,
             latent_read_znorm="rms", latent_read_znorm_target=E_NORM,
             latent_read_scramble=(limb == "donor"),
             latent_carry_read=(limb != "noread"))
    f.update(kw)
    return f


def _cortex(limb="real", **kw):
    """The graft alone -- NOT through _build_raven, which turns a ValueError
    into a SKIP, so a must-raise test would go green checking nothing."""
    base = dict(n_embd=D, summary_init_token=EOS)
    base.update(_flags(limb, **kw))
    return CortexMemory(SimpleNamespace(**base))


def _model(limb="real", **kw):
    torch.manual_seed(1234)
    return _build_raven(**_flags(limb, **kw)).train()


def _carry(e_norm=E_NORM, z_norm=10.0, unwritten=0, seed=0):
    """A [B, K, 2D] dual-channel carry at REALISTIC scales.

    z_norm 10 is ||s_T|| at the summary columns (28.1x the trunc_normal_ noise
    at ||s0|| = 0.39), i.e. 17x under E -- the gap the rescale exists to close.
    `unwritten` trailing rows are exactly zero, as a gated ring's unreached rows
    are (PrefixGatedBuffer._slot_init_block).
    """
    torch.manual_seed(seed)
    e, z = torch.randn(B, K, D), torch.randn(B, K, D)
    e = e / e.norm(dim=-1, keepdim=True) * e_norm
    z = z / z.norm(dim=-1, keepdim=True) * z_norm
    st = torch.cat([e, z], dim=-1)
    if unwritten:
        st[:, -unwritten:] = 0.0
    return st


def _pack(g, state, eos=None, emb=None):
    emb = torch.randn(B, S, D) if emb is None else emb
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)
    g.begin(state, eos, S, torch.device("cpu"), torch.float32)
    packed, ppos, n_prefix, n_sum = g.prefix_pack(emb, pos)
    return packed, ppos, n_prefix, n_sum, emb


def _e_block(p):
    return p[:, :K]


def _z_block(p):
    return p[:, K:2 * K]


# ─── 1. the layout and the column accounting ─────────────────────────────────

class TestTheLayout:
    def test_the_three_limbs_build_with_one_projection_and_no_reader(self):
        for limb in ("real", "donor", "noread"):
            g = _cortex(limb)
            assert isinstance(g.latent_embed, LatentEmbedRead)
            # NOT latent_reader: J4 has no in-loop read and no gate, and the
            # diag's read_gate/own_share fields hang off that attribute.
            assert g.latent_reader is None and g.latent_scratch is None
            assert g.latent_carry_read == (limb != "noread")

    def test_j3_and_j1_configs_are_untouched(self):
        j3 = _cortex(latent_read="scratch", latent_encoding="scratch",
                     latent_read_znorm_target=3.0, latent_read_heads=4)
        j1 = _cortex(latent_read="xattn", latent_encoding="tokens",
                     latent_read_znorm="none", latent_read_heads=4)
        assert j3.latent_embed is None and j1.latent_embed is None

    def test_the_packed_layout_is_e_then_z_then_tokens_then_summary(self):
        g, st = _cortex(), _carry()
        p, ppos, n_prefix, n_sum, emb = _pack(g, st)
        assert p.shape[1] == K + K + S + W
        assert torch.allclose(_e_block(p), st[..., :D])       # E unchanged
        assert torch.allclose(p[:, 2 * K:2 * K + S], emb)     # tokens after both
        assert n_sum == W

    def test_n_prefix_is_the_total_and_n_pre_is_e_alone(self):
        """The one accounting split J4 introduces, and the reason it is safe:
        the modeling file round-trips `n_prefix` straight into prefix_unpack as
        an opaque count, so returning the TOTAL is what keeps that file
        unchanged -- while the substitution sites keep addressing E's columns."""
        g, st = _cortex(), _carry()
        _, _, n_prefix, _, _ = _pack(g, st)
        assert (n_prefix, g._n_pre, g._n_zpre, g._n_carry_cols) == (2 * K, K, K, 2 * K)

    def test_every_consumer_of_the_total_gets_the_total(self):
        g, st = _cortex(), _carry()
        p, _, _, _, _ = _pack(g, st)
        # the write's real-token count must exclude BOTH prepended blocks, or
        # 'tokens' would pool the Z block as if it were the chunk's last tokens
        assert p.shape[1] - g._n_carry_cols - g._n_sum == S
        # prefix_unpack strips both blocks and returns the real tokens.  It
        # takes the count the modeling file round-tripped, i.e. the total.
        x = torch.randn(B, p.shape[1], D)
        g.latent_init(x, 0, T)
        for _ in range(2):
            g.iter_write(x)                     # the merge needs a live tape
        real, new_state = g.prefix_unpack(x, 2 * K, W)
        assert real.shape[1] == S and torch.allclose(real, x[:, 2 * K:2 * K + S])
        assert new_state.shape == (B, K, 2 * D)

    def test_positions_keep_the_tail_layout(self):
        """Both carried blocks at 0, tokens 1..S, summary S+1..S+W -- so every
        token->carry offset stays POSITIVE and no index leaves the trained
        window.  E's 64 columns already share one position; Z's share it too."""
        g, st = _cortex(), _carry()
        _, ppos, _, _, _ = _pack(g, st)
        want = torch.cat([torch.zeros(2 * K, dtype=ppos.dtype),
                          torch.arange(1, S + 1),
                          torch.arange(S + 1, S + 1 + W)])
        assert torch.equal(ppos[0], want)

    def test_a_non_j4_arm_splices_no_z_columns(self):
        g = _cortex(latent_read="xattn", latent_encoding="tokens",
                    latent_read_znorm="none", latent_read_heads=4,
                    latent_tok_pool=2)
        p, _, n_prefix, _, _ = _pack(g, _carry())
        assert p.shape[1] == K + S + W and g._n_zpre == 0 and n_prefix == K

    def test_the_eos_mask_lift_spans_both_blocks(self):
        g, st = _cortex(), _carry()
        eos = torch.zeros(B, S, dtype=torch.bool)
        eos[:, 5] = True
        p, _, _, _, _ = _pack(g, st, eos=eos)
        m = g._packed_read_mask(p)          # raises if the widths disagree
        assert m.shape[1] == p.shape[1]
        assert float(m[:, :2 * K].min()) == 1.0   # carried blocks are live


# ─── 2. the scale ───────────────────────────────────────────────────────────

class TestZEntersAtEsScale:
    def test_every_row_enters_at_the_target_norm(self):
        g, st = _cortex(), _carry()
        p, _, _, _, _ = _pack(g, st)
        rn = _z_block(p).detach().float().norm(dim=-1)
        assert torch.allclose(rn, torch.full_like(rn, E_NORM), atol=1e-2)

    def test_at_init_the_block_is_the_rescaled_carry(self):
        """Identity projection, so step 0 splices a correctly-scaled copy: the
        read is LIVE from the first batch rather than a no-op to be discovered
        (the x0.90 shape this project has paid for twice)."""
        g, st = _cortex(), _carry()
        p, _, _, _, _ = _pack(g, st)
        z = st[..., D:]
        assert torch.allclose(_z_block(p),
                              z / z.norm(dim=-1, keepdim=True) * E_NORM, atol=1e-3)

    def test_the_diag_reports_the_read_strength_at_about_one(self):
        """z_embed_ratio is J4's only read-strength number -- it stands where
        J1's read_gate and J3's read_ratio stand, and it is ~1.0 at init by
        construction, so the TRAJECTORY is the reading."""
        g, st = _cortex(), _carry()
        _pack(g, st)
        rt = latent_runtime(g)          # unprefixed here; z_-prefixed in the diag
        assert rt["n_zpre"] == K
        assert abs(rt["embed_ratio"] - 1.0) < 0.01
        assert abs(rt["e_carried_norm"] - E_NORM) < 1.0
        assert rt["znorm_target"] == E_NORM

    def test_the_diag_key_is_the_one_the_scorer_reads(self):
        """training_diag rewrites every latent_runtime key as z_<key>, so a
        field named "z_embed_ratio" there lands as `z_z_embed_ratio` and
        evals/score_j1.embed_trajectory finds nothing -- a read-strength column
        that is silently always absent, which reads as "this run never recorded
        it" rather than as a bug.  Pin the spelling from both ends."""
        import importlib

        g, st = _cortex(), _carry()
        _pack(g, st)
        lat = latent_runtime(g)
        row = {f"z_{k}": v for k, v in lat.items() if k != "latent_carry"}
        sys.path.insert(0, os.path.join(REPO, "evals"))
        S = importlib.import_module("score_j1")
        assert "z_embed_ratio" in row and "z_z_embed_ratio" not in row
        assert "z_znorm_target" in row          # the read-out's rescale guard
        import json as _json
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cortex_diag.jsonl")
            with open(path, "w") as fh:
                for step, ratio in ((0, 1.0), (1, 1.0), (2, 0.05), (3, 0.05)):
                    r = dict(row)
                    r["step"], r["z_embed_ratio"] = step, ratio
                    fh.write(_json.dumps(r) + "\n")
            et = S.embed_trajectory(path)
        assert et is not None and et["collapsed"]

    def test_a_shrinking_projection_shows_up_as_a_falling_ratio(self):
        g, st = _cortex(), _carry()
        with torch.no_grad():
            g.latent_embed.proj.weight.mul_(0.1)
        _pack(g, st)
        assert abs(latent_runtime(g)["embed_ratio"] - 0.1) < 0.01


# ─── 3. the nulls ───────────────────────────────────────────────────────────

class TestTheNullsAreZerosNotOmission:
    def _geom(self, p, ppos, n_prefix):
        return (tuple(p.shape), ppos.tolist(), n_prefix)

    def test_the_no_read_limb_keeps_the_columns_and_zeroes_them(self):
        g, st = _cortex("noread"), _carry()
        p, _, _, _, _ = _pack(g, st)
        assert p.shape[1] == K + K + S + W
        assert float(_z_block(p).abs().sum()) == 0.0

    def test_the_z_off_cell_is_geometrically_identical_to_the_real_cell(self):
        """The 2x2's z_null='off'.  `_latent_z_rows` returns None there, and
        omitting the columns would make the cells differ in sequence length
        rather than in the carry -- which chunk 1 cannot catch, because it
        carries nothing and splices no columns either way."""
        st = _carry()
        emb = torch.randn(B, S, D)
        g_on = _cortex()
        on = _pack(g_on, st, emb=emb)
        g_off = _cortex()
        g_off.latent_read_null = ("off",)
        off = _pack(g_off, st, emb=emb)
        assert self._geom(on[0], on[1], on[2]) == self._geom(off[0], off[1], off[2])
        assert float(_z_block(off[0]).abs().sum()) == 0.0
        assert float(_z_block(on[0]).abs().sum()) > 0.0
        # and E is untouched by Z's null
        assert torch.allclose(_e_block(on[0]), _e_block(off[0]))

    def test_unwritten_ring_rows_stay_exactly_zero_as_es_do(self):
        g, st = _cortex(), _carry(unwritten=5)
        p, _, _, _, _ = _pack(g, st)
        assert float(_z_block(p)[:, -5:].abs().sum()) == 0.0
        assert float(_e_block(p)[:, -5:].abs().sum()) == 0.0   # unchanged for E

    def test_the_projection_has_no_bias_so_zeros_give_zeros(self):
        """Load-bearing twice: a bias would make the null a learned constant
        vector (an E0Z0 cell quietly reading something, which
        score_j1.NOREAD_ZMAIN_TOL exists to veto) and would fill unwritten ring
        rows with it."""
        m = LatentEmbedRead(D)
        assert m.proj.bias is None
        assert float(m(torch.zeros(1, 3, D)).abs().sum()) == 0.0

    def test_chunk_one_splices_neither_block(self):
        g = _cortex()
        p, _, n_prefix, _, _ = _pack(g, None)
        assert n_prefix == 0 and p.shape[1] == S + W

    def test_the_donor_null_swaps_content_at_the_same_columns(self):
        st = _carry()
        g = _cortex()
        g.latent_read_null = ("donor", st[..., D:].roll(1, dims=0))
        p, _, _, _, _ = _pack(g, st)
        assert p.shape[1] == K + K + S + W
        z = st[..., D:].roll(1, dims=0)
        assert torch.allclose(_z_block(p),
                              z / z.norm(dim=-1, keepdim=True) * E_NORM, atol=1e-3)


# ─── 4. the limbs differ in the carry alone ─────────────────────────────────

class TestTheLimbs:
    def test_the_donor_limb_rolls_the_carry_and_only_the_carry(self):
        g, st = _cortex("donor"), _carry()
        p, _, _, _, _ = _pack(g, st)
        z = st[..., D:].roll(1, dims=0)
        assert torch.allclose(_z_block(p),
                              z / z.norm(dim=-1, keepdim=True) * E_NORM, atol=1e-3)
        assert torch.allclose(_e_block(p), st[..., :D])       # E never rolled

    def test_e_dropout_blanks_e_and_leaves_z(self):
        """The E-off cell is the only cell that can return NO, and it needs E
        blanked with Z intact.  E-dropout addresses parts[-1], so the Z block is
        appended after it precisely so the two can never be confused."""
        g = _cortex(e_dropout=0.95)
        g.train()
        torch.manual_seed(0)
        p, _, _, _, _ = _pack(g, _carry())
        blanked = [b for b in range(B) if float(_e_block(p)[b].abs().sum()) == 0.0]
        assert blanked, "e_dropout 0.95 blanked no row of a 2-row batch"
        for b in blanked:
            assert float(_z_block(p)[b].abs().sum()) > 0.0

    def test_the_reported_e_norm_is_the_row_a_read_competes_with(self):
        """Measured BEFORE e_dropout, so z_embed_ratio's denominator stays the
        real E row even on the rows whose E was blanked."""
        g = _cortex(e_dropout=0.95)
        g.train()
        torch.manual_seed(0)
        _pack(g, _carry())
        assert abs(g._e_carried_norm - E_NORM) < 1.0


# ─── 5. the gradient property ───────────────────────────────────────────────

class TestTheGradient:
    def test_the_projection_gets_gradient_from_the_splice(self):
        g, st = _cortex(), _carry()
        p, _, _, _, _ = _pack(g, st)
        p.sum().backward()
        gr = g.latent_embed.proj.weight.grad
        assert gr is not None and float(gr.abs().sum()) > 0

    @pytest.mark.parametrize("n_no_grad", [0, 1, 3])
    def test_the_read_is_live_however_long_the_no_grad_prefix_is(self, n_no_grad):
        """s0's read gradient is EXACTLY zero at any split with n >= 1, because
        the no-grad iterations run first and Z entered once.  J4's columns live
        in `input_embeds`, which is re-fed on every iteration -- E's property."""
        g, st = _cortex(), _carry()
        _pack(g, st)
        g.latent_init(torch.randn(B, 2 * K + S + W, D), n_no_grad, T - n_no_grad)
        assert g.latent_read_grad_frac == 1.0

    def test_the_endpoint_write_still_reads_the_summary_columns(self):
        g, st = _cortex(), _carry()
        p, _, _, _, _ = _pack(g, st)
        x = torch.randn(B, p.shape[1], D)
        g.latent_init(x, 0, T)
        for _ in range(2):
            g.iter_write(x)
        w = g.latent_write()
        assert w.shape == (B, W, D) and torch.allclose(w, x[:, -W:])


# ─── 6. the designed init ───────────────────────────────────────────────────

class TestTheDesignedInit:
    def test_it_is_an_identity(self):
        assert torch.allclose(LatentEmbedRead(D).proj.weight, torch.eye(D))

    def test_it_survives_a_kaiming_reset(self):
        """post_init calls reset_parameters() on every cortex submodule.  Left
        alone, that runs a RANDOM ROTATION of Z at the right scale -- a control
        arm wearing the treatment's name, invisible in the loss."""
        m = LatentEmbedRead(D)
        torch.nn.init.kaiming_uniform_(m.proj.weight)
        assert not torch.allclose(m.proj.weight, torch.eye(D))
        tags = m.apply_designed_init()
        assert torch.allclose(m.proj.weight, torch.eye(D))
        assert any("proj=I" in t for t in tags)

    def test_reset_cortex_graft_init_reaches_it(self):
        """The hook that matters: the module's own method is not enough unless
        reset_cortex_graft_init calls it (LatentScratchpad's lesson)."""
        model = _model()
        cortex = getattr(model, "cortex", None)
        if cortex is None or cortex.latent_embed is None:
            pytest.skip("no graft on the built model")
        torch.nn.init.kaiming_uniform_(cortex.latent_embed.proj.weight)
        reset_cortex_graft_init(model)
        assert torch.allclose(cortex.latent_embed.proj.weight, torch.eye(D))

    def test_the_projection_is_exempt_from_weight_decay(self):
        assert getattr(LatentEmbedRead(D).proj.weight, "_no_weight_decay", False)


# ─── 7. the refusals ───────────────────────────────────────────────────────

class TestTheSwitchesRefuseNeighbouringDesigns:
    @pytest.mark.parametrize("kw, match", [
        # D3 measured the last-tokens writes holding ~1% of E's register
        # margin; reading one of those through a better reader re-runs attempt
        # 2's NO with a new read and the same empty channel.
        (dict(latent_encoding="tokens"), "latent_encoding endpoint"),
        (dict(latent_encoding="delta"), "latent_encoding endpoint"),
        # both read sites on = the same Z entering twice, unattributable
        (dict(latent_s0_read=True), "latent_s0_read"),
        # J1's no-read flag would omit the columns, not blank them
        (dict(latent_write_only=True), "no-read limb"),
        # spliced as carried, Z_end is 17x under E -- the s0 site's failure
        (dict(latent_read_znorm="none"), "znorm rms"),
    ])
    def test_bad_combinations_raise(self, kw, match):
        with pytest.raises(ValueError, match=match):
            _cortex("real", **kw)

    def test_the_no_read_limb_needs_a_read_that_has_columns_to_blank(self):
        with pytest.raises(ValueError, match="no-read limb of J3"):
            _cortex(latent_read="xattn", latent_encoding="tokens",
                    latent_read_znorm="none", latent_read_heads=4,
                    latent_carry_read=False)

    def test_embeds_is_an_accepted_read_site(self):
        with pytest.raises(ValueError, match="latent_read must be"):
            _cortex(latent_read="embed")          # the plausible typo


# ─── 8. end to end on the real model (cluster; skips on transformers skew) ──

class TestOnTheRealModel:
    def test_a_forward_runs_and_the_carry_round_trips(self):
        m = _model()
        ids = torch.randint(0, VOCAB - 1, (B, CL))
        out = m(ids, num_steps=torch.tensor([0, T]), return_m_cross=True)
        st = out["m_cross"] if isinstance(out, dict) else out.m_cross
        assert st.shape == (B, K, 2 * D)          # dual channel, unchanged
        assert out["logits"].shape[1] == CL       # the carry never reaches the head
        # second chunk: now there IS a carry, so the Z columns are spliced
        out2 = m(ids, num_steps=torch.tensor([0, T]), m_cross_in=st,
                 return_m_cross=True)
        assert out2["logits"].shape[1] == CL
        assert m.cortex._n_zpre == K

    def test_the_limbs_differ_in_the_carry_alone(self):
        ids = torch.randint(0, VOCAB - 1, (B, CL))
        outs = {}
        for limb in ("real", "noread"):
            m = _model(limb)
            torch.manual_seed(7)
            o1 = m(ids, num_steps=torch.tensor([0, T]), return_m_cross=True)
            st = o1["m_cross"] if isinstance(o1, dict) else o1.m_cross
            torch.manual_seed(7)
            o2 = m(ids, num_steps=torch.tensor([0, T]), m_cross_in=st,
                   return_m_cross=True)
            outs[limb] = o2["logits"]
        # chunk 2 differs (one limb reads the carry, the other reads zeros)
        assert not torch.allclose(outs["real"], outs["noread"])

    def test_the_z_columns_reach_the_loss(self):
        m = _model()
        ids = torch.randint(0, VOCAB - 1, (B, CL))
        o1 = m(ids, num_steps=torch.tensor([0, T]), return_m_cross=True)
        st = (o1["m_cross"] if isinstance(o1, dict) else o1.m_cross).detach()
        out = m(ids, num_steps=torch.tensor([0, T]), m_cross_in=st,
                labels=ids, return_m_cross=True)
        (out["loss"] if isinstance(out, dict) else out.loss).backward()
        gr = m.cortex.latent_embed.proj.weight.grad
        assert gr is not None and float(gr.abs().sum()) > 0


# ─── 9. the launcher and the read-out agree with this file ──────────────────

def _src(rel: str) -> str:
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        return fh.read()


class TestTheLaunchers:
    def test_the_joint_launcher_fixes_the_registered_design(self):
        s = _src("pace/j4_joint.sbatch")
        assert "ENCODING=endpoint" in s                 # D3's write, not a variable
        assert "--cortex.latent_read embeds" in s
        assert "--cortex.latent_s0_read false" in s
        assert "ZNORM_TARGET=${ZNORM_TARGET:-171.0}" in s
        # the no-read limb must blank the columns, never omit them
        assert "--cortex.latent_carry_read false" in s
        assert "--cortex.latent_write_only" not in s.replace(
            "NOT --cortex.latent_write_only", "")
        # all three pairs at J1's and J3's budget, so the designs compare
        assert "CELL_STEPS=${CELL_STEPS:-2000}" in s
        # micro >= 2 or the donor roll returns the same row and that limb IS
        # the real limb -- refused by the launcher, not left to the operator
        assert 'if [ "$MICRO_BS" -lt 2 ]' in s and "donor limb IS the real limb" in s

    def test_the_readout_reproduces_the_trained_read(self):
        s = _src("pace/j1_readout.sbatch")
        assert "j4) ENCODING=${ENCODING:-endpoint}" in s
        assert "--set latent_read=embeds" in s
        assert "ZNORM_TARGET=${ZNORM_TARGET:-171.0}" in s
        # the guard: a read-out rebuilt at another rescale target scores a model
        # nobody trained (171 vs the 3.0 default is a factor of 57)
        assert "z_znorm_target" in s
        assert "EXPERIMENT must be j1|j3|j4" in s

    def test_the_verdict_launcher_cannot_overwrite_the_first_pair(self):
        """Two E-dropout values are scored against the same main limbs, so the
        0.95 pass has to redirect its output or it replaces the 0.25 verdict."""
        s = _src("pace/j1_verdict.sbatch")
        assert "j4) ENCODING=${ENCODING:-endpoint}" in s
        assert "VERDICT_OUT" in s and "--out $VERDICT_OUT" in s

    def test_the_scorer_knows_j4_and_defaults_to_endpoint(self):
        import importlib
        sys.path.insert(0, os.path.join(REPO, "evals"))
        S = importlib.import_module("score_j1")
        assert "j4" in S.EXPERIMENTS
        assert S.run_name("real", "endpoint", "", "j4") == "j4-a3z-endpoint-real"
        assert (S.run_name("noread", "endpoint", "0.95", "j4")
                == "j4-a3z-endpoint-edrop0.95-noread")
        # J4's own registered numbers exist before any J4 number does
        assert S.E_CARRY_MAX == 1.20 and S.E_CARRY_BAND == (0.37, 0.85)

    def test_the_rescale_target_survives_a_set_override(self):
        """`--set` values are typed by evals/model_utils.parse_config_overrides,
        which parses bools and ints but NOT floats -- so 171.0 arrives as the
        STRING "171.0".  The graft coerces it; pin that, because a target that
        silently fell back to the 3.0 default would put Z in 57x too small."""
        import importlib
        sys.path.insert(0, os.path.join(REPO, "evals"))
        mu = importlib.import_module("model_utils")
        over = mu.parse_config_overrides(["latent_read_znorm_target=171.0"])
        g = _cortex("real", **{k: v for k, v in over.items()})
        assert g.latent_read_znorm_target == 171.0
