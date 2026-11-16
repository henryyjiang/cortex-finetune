"""diag_fact_probe.py's pure parts: the CV probe, provenance against the real
v2 buffer, the per-channel splice and the donor rule.  CPU, no model."""
from __future__ import annotations

import os
import random
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evals"))

from cortex_memory.buffers import PrefixGatedBuffer, PrefixGatedBufferV2  # noqa: E402
from diag_babilong_dissect import Fact  # noqa: E402
from diag_fact_probe import (  # noqa: E402
    cv_probe, dist_band, donor_answer, pick_swap_donor, provenance, slot_probe,
    splice,
)

D, W, K = 8, 4, 16


def _blobs(n, d, n_cls, sep, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % n_cls
    centers = torch.randn(n_cls, d, generator=g) * sep
    return centers[y] + torch.randn(n, d, generator=g), y


def test_probe_finds_separable_classes_and_not_noise():
    X, y = _blobs(240, 32, 6, sep=3.0)
    assert cv_probe(X, y, 6)["acc"] > 0.9
    noise = torch.randn(240, 32, generator=torch.Generator().manual_seed(5))
    assert cv_probe(noise, y, 6)["acc"] < 0.35          # chance 1/6


def test_slot_probe_finds_a_fact_held_by_one_slot_that_the_mean_dilutes():
    g = torch.Generator().manual_seed(2)
    n, r, d = 240, 16, 32
    y = torch.arange(n) % 6
    rows = torch.randn(n, r, d, generator=g) * 3.0
    centers = torch.randn(6, d, generator=g) * 4.0
    rows[:, 5] += centers[y]                            # only slot 5 knows
    res = slot_probe(rows, y, 6)
    assert res["best_slot"] == 5 and res["acc"] > 0.8
    assert res["acc"] > cv_probe(rows.mean(1), y, 6)["acc"]
    shuf = y[torch.randperm(n, generator=torch.Generator().manual_seed(9))]
    assert slot_probe(rows, shuf, 6)["acc"] < 0.4       # best-of-16 null


@pytest.mark.parametrize("pairing", ["paired", "independent"])
def test_provenance_recovers_every_kept_row_of_the_real_v2_buffer(pairing):
    torch.manual_seed(0)
    b = PrefixGatedBufferV2(D, n_vec=W, n_slots=K, carries_latent=True,
                            pairing=pairing)
    for h in b.score_heads:          # learned eviction, not FIFO
        torch.nn.init.normal_(h.weight, std=1.0)
    g = torch.Generator().manual_seed(1)
    state, e_w, z_w = None, [], []
    for _ in range(9):
        e, z = torch.randn(1, W, D, generator=g), torch.randn(1, W, D, generator=g)
        e_w.append(e[0]); z_w.append(z[0])
        state = b.merge(state, e, z)
    # the E store cap rescales rows: provenance must survive it
    e_fin = state[0, :, :D] * 3.7
    for rows, writes in ((e_fin, e_w), (state[0, :, D:], z_w)):
        pv = provenance(rows, writes)
        assert pv["valid"]
        for r in range(K):
            src = writes[pv["chunk"][r]][pv["slot"][r]]
            assert torch.allclose(torch.nn.functional.normalize(rows[r], dim=0),
                                  torch.nn.functional.normalize(src, dim=0))
            assert pv["slot"][r] == r % W               # the class constraint


def test_provenance_refuses_a_blending_buffer():
    torch.manual_seed(0)
    ring = PrefixGatedBuffer(D, n_vec=W, n_slots=K, route="ring", fill="grow",
                             carries_latent=True)
    g = torch.Generator().manual_seed(1)
    state, e_w = None, []
    for _ in range(9):
        e, z = torch.randn(1, W, D, generator=g), torch.randn(1, W, D, generator=g)
        e_w.append(e[0])
        state = ring.merge(state, e, z)
    assert not provenance(state[0, :, :D], e_w)["valid"]


def test_splice_replaces_only_the_named_channel():
    own, don = torch.zeros(1, K, 2 * D), torch.ones(1, K, 2 * D)
    e = splice(own, don, D, "e")
    assert e[..., :D].eq(1).all() and e[..., D:].eq(0).all()
    z = splice(own, don, D, "z")
    assert z[..., :D].eq(0).all() and z[..., D:].eq(1).all()
    assert splice(own, don, D, "both").eq(1).all()
    with pytest.raises(ValueError):
        splice(own, torch.ones(1, K - W, 2 * D), D, "e")


def _item(id_, gold, moves, n_prime=6, rows=K):
    facts, chunks = [], []
    for i, (person, room, chunk) in enumerate(moves):
        facts.append(Fact("move", person, room, i, i + 1, f"{person} went to the {room}."))
        chunks.append((chunk, chunk))
    return {"id": id_, "task": "qa1", "gold": gold, "entity": "Mary",
            "n_prime_chunks": n_prime, "rows": rows, "facts": facts,
            "fact_chunks": chunks}


def test_donor_must_put_the_person_in_a_different_known_room():
    item = _item("a", "kitchen", [("Mary", "kitchen", 2)])
    same = _item("b", "kitchen", [("Mary", "kitchen", 1)])
    unknown = _item("c", "garden", [("John", "garden", 1)])
    other_rows = _item("d", "office", [("Mary", "office", 1)], rows=K - W)
    good = _item("e", "office", [("Mary", "hallway", 0), ("Mary", "office", 3)])
    assert donor_answer("qa1", good, "Mary") == ("office", 3)
    pick = pick_swap_donor(item, [item, same, unknown, other_rows, good],
                           random.Random(0))
    assert pick is not None and pick[0]["id"] == "e" and pick[1] == "office"
    assert pick_swap_donor(item, [item, same, unknown], random.Random(0)) is None


# ---------------------------------------------------------------------------
# main() end to end: a stub model around the REAL v2 buffer, the dissect
# tests' fake tokenizer, synthetic qa1 stories.  Zero score heads = FIFO, so
# survival has a known answer: all W fact rows iff distance <= K/W.
# ---------------------------------------------------------------------------

class _StubModel(torch.nn.Module):
    V, DIM, WV, KS = 5000, 32, 4, 16

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.wte = torch.nn.Embedding(self.V, self.DIM)
        self.register_buffer("zmap", torch.randn(self.DIM, self.DIM))
        self.buf = PrefixGatedBufferV2(self.DIM, n_vec=self.WV, n_slots=self.KS,
                                       carries_latent=True, pairing="independent")
        from types import SimpleNamespace
        self.cortex = SimpleNamespace(prefix=self.buf, has_cross_state=True,
                                      latent_carry=False)

    def get_input_embeddings(self):
        return self.wte

    def forward(self, input_ids, num_steps=None, m_cross_in=None,
                return_m_cross=False, position_ids=None, prefix_write=True,
                prefix_read=True):
        x = self.wte(input_ids % self.V)
        logits = x @ self.wte.weight.T
        if m_cross_in is not None and prefix_read:
            logits = logits + (m_cross_in[..., :self.DIM].mean(1)
                               @ self.wte.weight.T)[:, None]
        m = None
        if return_m_cross:
            g = torch.Generator().manual_seed(int(input_ids.sum()))
            e = x.mean(1, keepdim=True) + torch.randn(
                1, self.WV, self.DIM, generator=g).to(x.device)
            m = self.buf.merge(m_cross_in, e, e @ self.zmap)
        return {"logits": logits, "m_cross": m}


def _stories(n, seed):
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from test_babilong_dissect import make_context
    rng = random.Random(seed)
    rooms = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]
    rows = []
    for _ in range(n):
        at = sorted(rng.sample(range(50, 1500), 3))
        moves = [(rng.choice(["Mary", "John"]), rng.choice(rooms)) for _ in at]
        moves[-1] = ("Mary", moves[-1][1])
        ctx = make_context(2000, {k: f"{p} went to the {r}." for k, (p, r)
                                  in zip(at, moves)}, seed=rng.randrange(10**6))
        rows.append({"input": ctx, "question": "Where is Mary?",
                     "target": moves[-1][1]})
    return rows


def test_main_end_to_end_on_a_stub_v2_model(tmp_path, monkeypatch, capsys):
    import json
    import diag_fact_probe as F
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from test_babilong_dissect import FakeTok
    model, tok = _StubModel(), FakeTok()
    data = {"1k": _stories(30, 1), "2k": _stories(30, 2)}

    def fake_load(ckpt, name, slots, dtype, device, **k):
        print(f"[load] overlaid {ckpt}: 0 missing / 0 unexpected keys")
        return model.to(device), object()
    monkeypatch.setattr(F, "load_checkpoint", fake_load)
    monkeypatch.setattr(F, "load_bucket", lambda task, L, p, r: list(data[L]))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                        lambda *a, **k: tok)
    out = tmp_path / "fp"
    assert F.main(["--model_name", "stub", "--checkpoint", "stub/chkpt.pt",
                   "--lengths", "1k", "2k", "--max_items", "15",
                   "--folds", "3", "--boot", "50", "--out_dir", str(out)]) == 0
    res = json.loads((out / "results.json").read_text())
    items = [json.loads(l) for l in (out / "items.jsonl").read_text().splitlines()]
    assert res["n_items"] == len(items) >= 10
    lap = _StubModel.KS // _StubModel.WV
    for ch in ("e", "z"):
        assert res["survival"][ch]["provenance_valid"]
        for it in items:
            want = _StubModel.WV if it["distance"] <= lap else 0
            assert it["survival"][ch]["n_fact_rows"] == want, (ch, it["id"])
    assert any(it["distance"] > lap for it in items)
    assert any(it["distance"] <= lap for it in items)
    for k in ("text_bag", "e_write", "z_write", "e_slot", "z_slot",
              "final_e", "final_z", "shuffled_e_write"):
        assert res["probes"][k]["acc"] is not None, k
    assert res["read"]["n_swapped"] > 0
    for which in ("e", "z", "both"):
        assert res["read"][which]["all"][0] is not None


def test_distance_bands():
    assert [dist_band(d) for d in (1, 2, 4, 5, 8, 9, 30)] == \
        ["1", "2-4", "2-4", "5-8", "5-8", "9+", "9+"]
