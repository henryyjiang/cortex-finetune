"""
Memory interpretability for cortex-final -- WHERE the carry is read, WHAT
causally depends on the read, and WHAT the rows hold.

FORWARD-ONLY.  Every number here comes from no-grad forwards of the trained
model on PG-19 validation, chained over `--n_chunks` chunks per row exactly as
the 2x2 and the influence horizon chain them (fresh carry at chunk 1, the
model's own write carried forward), at T=8.

THE PACKED LAYOUT THIS RELIES ON -- VERIFIED, NOT ASSUMED.  `prefix_pack`
builds, for chunk q,

    [ E rows (nE) | Z rows (nZ = nE) | L real tokens | 16 summary slots ]
      pos 0 ...     pos 0 ...          pos 1..L        pos L+1..L+16

with nE = 16 * min(q-1, 4) under gate_fill=grow (chunk 1 carries nothing).
Every carried column sits at RoPE position 0, so among carried columns the
attention is purely content-addressed.  The run re-derives this on row 0 at
chunk 2 and chunk n from the tensors prefix_pack actually returned (E block ==
the carried E, Z block == latent_embed(rescale(Z)), tokens == wte(ids), slots
== summary_emb, positions as above) and REFUSES if any of it disagrees.

THE ATTENTION IS SDPA, WHICH RETURNS NO WEIGHTS.  So every attention module's
forward is shadowed by a REPLICA of CausalSelfAttention.forward (same calls,
same order: Wqkv -> q/k RMSNorm -> heads -> RoPE -> SDPA(is_causal) -> proj)
that also recomputes softmax(QK^T/sqrt(d) + causal) in fp32 on the side.  The
replica is the model's path; the fp32 probabilities are the instrument.  Three
checks make that safe, all on row 0 and all VETOES:

  transparency  the full chain with the replica installed is BIT-IDENTICAL
                (per-token CE and a logits checksum) to the chain without it.
  recompute     on chunk 2 and chunk n, for EVERY attention call, proj(P @ V)
                from the fp32 probabilities matches the ORIGINAL class forward
                run on the same input: relative Frobenius error < 1e-3 (fp32).
                A NEGATIVE CONTROL (the same recompute with the causal mask
                dropped) must miss by >= 10x the recompute's own error at
                every call, or the check is refused as non-discriminating
                (measured: 9e-7 vs >= 0.37 in fp32).
  shares        every real-token query's mass over (E, Z, own tokens, summary)
                sums to 1, and the summary slots receive EXACTLY zero from
                earlier tokens (causal mask).

A. READ ATTENTION SHARES.  For real-token queries in chunks 2..n, per block
   (prelude layer i | core layer i at loop iteration t | coda layer i) and per
   head: mean probability mass on carried E columns, carried Z columns, the
   chunk's own real tokens, and summary slots (asserted 0).  Also:
     * per-column normalisation (mass / number of columns of that kind; own
       tokens are divided by the number VISIBLE to that query, i+1), because
       64 carried rows and up to 512 tokens differ in count;
     * by ring-row AGE (chunks since the row was last written: 1, 2, 3, 4+),
       derived from the ring rule by replaying
       cortex_memory.health.expected_ring_rows, cross-checked against a closed
       form, and checked against the rows the real merge actually changed;
     * by query position within the chunk (8 bins);
     * `first_tok` -- mass on the chunk's FIRST real token, in chunk 1 (no
       carry) and in chunks 2..n.  This is the ATTENTION-SINK CONTROL.  The
       carried columns sit at the front of the sequence, exactly where a
       causal LM dumps its no-op attention, so a large E/Z share can be a SINK
       rather than a read.  If E+Z mass in chunks >= 2 roughly equals the drop
       in first_tok mass relative to chunk 1, read the share as sink
       displacement.  Part B, not part A, says whether the mass is USED.
   The top-20 "memory heads" (highest E+Z mass) are listed with their block.
   Attention mass is NOT influence.  Treat A as a map of where to look.

B. CAUSAL LOCALISATION BY ATTENTION BLOCKING.  Mean NLL over chunks 2..n (fp32)
   intact vs with real-token -> carried-column attention set to -inf before the
   softmax, at ONE site at a time: each prelude layer; each core layer across
   ALL T iterations; each loop iteration t (all core layers at t); each coda
   layer; plus everywhere, and everywhere-E-only / everywhere-Z-only.
     * TEACHER-FORCED CARRY.  Chunk q always reads the INTACT chain's carry
       (state after chunk q-1), so a site's delta is the value of the READ at
       that site on that chunk -- never contaminated by a damaged write
       compounding down the chain.  The blocked forward's own write is
       discarded (merge is a no-op during B), and the ring's MUTABLE write
       pointer (buf._chunk/_z_chunk/_z_pending) is snapshotted and restored
       around every side forward -- and the run raises if it moved -- so no
       side forward can shift where a later chain writes.
     * Only REAL-TOKEN queries are blocked.  Carried columns still attend to
       each other and the summary slots still read the carry.
     * Blocking is not a null VALUE (zeros still score a mid-range logit and
       absorb softmax mass; noise is off-distribution): blocked columns get
       exactly zero probability.  Block-everywhere is therefore the clean total
       value of the read through attention -- the only way a carried column can
       reach a real token at all, since the adapter, the MLPs and the norms are
       per-position.
     * Single-site deltas are NOT additive.  A small delta at one layer can
       mean the read is redundant across layers, not that the layer is idle.
       Compare against block-everywhere.
     * Paired per row (same seeds, same intact carry), bootstrap CI over rows.
   Vetoes, every row: block-NOTHING (the wrapper installed, no site hit) must
   reproduce the intact per-token CE BIT-FOR-BIT.  Row 0: the explicit masked
   path with nothing blocked must reproduce intact NLL to < 1e-3 nats/chunk
   (the floor of the instrument; printed), and block-everywhere must differ
   from intact.
   RNG: initialize_state draws s0 noise on every forward, so every forward is
   seeded from (seed, row, chunk) -- NOT once per row as the 2x2 does.  Intact
   numbers are therefore distributionally, not bitwise, comparable to the 2x2.

C. WRITE SIDE.  The 16 summary slots' attention, per block (reported at the
   last core iteration and in the coda; every block is in results.json):
   mass on carried E / Z / real tokens / summary slots; then, renormalised over
   the chunk's real tokens: entropy (and entropy / ln L), mass by position bin,
   and mass by token class against that class's base rate (lift = share /
   base).  Token classes, a SIMPLE DOCUMENTED HEURISTIC on decoded text:
     content       word-initial, alphabetic, and capitalised or > 3 letters
     function      word-initial, alphabetic, lowercase, <= 3 letters
     continuation  alphabetic but not word-initial (a word's later sub-pieces)
     other         punctuation, digits, whitespace, mixed
   word-initial = the token's text starts with whitespace, or the previous
   token's text ends in whitespace / is not alphanumeric.  "Capitalised" also
   catches sentence-initial function words ("The"); that is a known flaw of the
   heuristic, not a finding.  Z rows are written at intermediate loop depths
   (slot j at depth 1 + j mod 7 at T=8), so the last-iteration attention is
   what forms E, not what forms Z -- every iteration is in results.json.

D. LOGIT LENS ON MEMORY ROWS.  E rows are post-ln_f states: lm_head applied
   directly (exact -- it is the model's own head on its own final states; on
   ring lap >= 2 a row is a gate mixture fg*old + ig*new, and lm_head of a
   mixture is the mixture of logits).  Z rows are pre-coda raw loop states:
   ln_f + lm_head, a CRUDE LENS that skips the coda -- labelled as such.  For
   each row whose writer chunk is known, precision@k (k=50) of the top-k
   decoded tokens against the bag of words of (i) the chunk that wrote it,
   (ii) the NEXT chunk, (iii) the DONOR -- the same chunk index of another row
   (the control; same register, different content).  Bag of words = lowercased
   stripped forms of alphabetic tokens; precision denominator is always k, so a
   lens that emits punctuation scores low on all three references alike.
   `p_long` repeats it counting only forms of >= 4 letters (less dominated by
   "the"/"of", which every chunk contains).  Buckets: `fresh` (the pre-merge
   write, writer exact, no gate mixing) and ring rows by age 1..4 as READ;
   `gated` marks ring rows whose last writer is on lap >= 2, and for those a
   fourth reference `prev` (the chunk the gate mixed the new write INTO, one
   lap earlier) asks whether the old content survives the gate.  READ THE AGE
   BUCKETS WITH CARE: a `pure` row is bit-identical from its write until it is
   overwritten, so its lens precision CANNOT change with age -- `pure` equals
   the mean over writers 1..lap by construction, and any age trend in D comes
   only from gate mixing and from which writer chunks populate each bucket.
   The content signal is own - donor (paired per row).  Three example rows (selected by
   highest own precision -- illustrative, not representative) are printed with
   their source chunk's first 40 words.

E. RING GEOMETRY.  Participation-ratio effective rank and mean centred cosine
   (cortex_memory.health.rank_stats -- the repo's one implementation) of the E
   and Z halves of the carry after each chunk.

ROWS.  The strided PG-19 pack is consecutive non-overlapping windows, ~21 per
book, with each book's last window EOS-marked and EOS-padded.  The first 24
rows would be ~1-2 books, and the next row -- the obvious donor -- the same
book's continuation (shared names, shared topic: a contaminated control).  So
books are detected from the tokens (BOS == EOS at position 0 opens one, an EOS
past position 0 marks the padded tail), padded windows are skipped, the
default `--row_select spread` takes one MIDDLE window per book with books
evenly spaced, and part D's donor is the middle window of the book half the
pack away -- always a different book.  results.json records every row's book
and donor; a run whose rows share books says so, because its row CIs are then
optimistic.

REFUSES (exit 2) rather than reporting a default: no cortex / no prefix buffer;
not a dual-channel PrefixGatedBuffer on route=ring, fill=grow, latent_stride=1;
latent_read != embeds; E or Z read switched off or nulled; any other read path
live (m_cross, accum, ccot, m_iter, latent_reader, scratch) -- those would
bypass attention and this instrument would not see them; summary_emb never
seeded or the gate projections all-zero (the checkpoint's cortex tensors did
not load); latent_read_znorm_target not given (the code default 3.0 is not a
trained value -- 78.9 at step 38146, 98.0 at 11444); no --checkpoint; a KV
cache or flex mask reaching the attention; any veto above.

fp32 by default.  B is a difference of two nearly equal losses -- the quantity
bf16 got wrong by 6x in P0.1.  --dtype bfloat16 is for smoke tests only; it
loosens the recompute tolerance, cannot pass the fidelity veto's fp32 bound
(that one only warns in bf16), and stamps results.json `smoke_only`.

USAGE (cluster, repo root)
  python evals/diag_memory_interp.py \
      --model_name ckpts/olmo-retrofit-cortex \
      --checkpoint cortex-5b/cortex-final/checkpoint_38146 \
      --data data/pg19_olmo_validation_len4096_strided \
      --set use_memory=true ... --set latent_read_znorm_target=78.9 \
      --n_chunks 8 --T 8 --max_rows 24 --parts A,B,C,D,E \
      --out_dir eval_results/memory_interp-38146
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortex_memory.health import (  # noqa: E402
    chance_margin, expected_ring_rows, rank_stats, refuse_if_scrambled,
)
from model_utils import (  # noqa: E402
    _unwrap, explain_missing_cortex, load_checkpoint, parse_config_overrides,
    to_num_steps,
)
from eval_influence_horizon import _patched_merge, paired_ci  # noqa: E402

PARTS = ("A", "B", "C", "D", "E")
READ_KINDS = ("E", "Z", "own", "summary", "first_tok")
WRITE_KINDS = ("E", "Z", "real", "summary")
TOKEN_CLASSES = ("content", "function", "continuation", "other")
#: ||proj(P @ V) - original module output|| / ||output|| (Frobenius).  A wrong
#: P (mask, scale, head layout) misses by O(0.1-1); the negative control
#: (non-causal P) is checked to miss by >= 10x this.
RECOMPUTE_TOL = {"float32": 1e-3, "bfloat16": 5e-2}
#: |mean NLL(explicit masked path, nothing blocked) - intact| per chunk.
FIDELITY_TOL = 1e-3
SHARE_SUM_TOL = 1e-4


# ---------------------------------------------------------------------------
# pure helpers -- no model, no GPU (tests/test_memory_interp.py)
# ---------------------------------------------------------------------------

def lap_chunks(W: int, K: int) -> int:
    """Chunks per ring lap.  Refuses a ragged ring (the buffer refuses it too)."""
    if W <= 0 or K <= 0 or K % W:
        raise ValueError(f"ring needs K ({K}) to be a positive multiple of W ({W})")
    return K // W


def carried_rows(q: int, W: int, K: int) -> int:
    """Rows carried INTO chunk q (1-based) under gate_fill=grow.

    Chunk 1 carries nothing; the block grows by W per chunk until it is K.
    """
    if q < 1:
        raise ValueError(f"chunk index is 1-based; got {q}")
    return W * min(q - 1, lap_chunks(W, K))


def ring_writers(q: int, W: int, K: int) -> list:
    """Row -> the chunk (1-based) that LAST wrote it, for the carry READ by chunk q.

    Derived by replaying the ring rule from cortex_memory.health.
    expected_ring_rows (chunk c writes rows ((c-1)W + j) mod K), which under a
    grow fill is also where the append lands.  Not asked of the buffer.
    """
    lap_chunks(W, K)
    writer = [None] * K
    for c in range(1, q):
        for r in expected_ring_rows(c - 1, W, K):
            writer[r] = c
    out = writer[:carried_rows(q, W, K)]
    if any(w is None for w in out):
        raise RuntimeError(f"ring replay left an unwritten row inside the "
                           f"carried block at q={q}: {out}")
    return out


def ring_writer_closed(q: int, b: int, W: int, K: int) -> Optional[int]:
    """Closed form for block b's last writer as read by chunk q (cross-check)."""
    L = lap_chunks(W, K)
    if q - 1 < b + 1:
        return None
    return b + 1 + L * ((q - 1 - (b + 1)) // L)


def block_ages(q: int, W: int, K: int) -> list:
    """Age (q - writer) of each carried W-row block, in block order.

    Raises if a block's rows disagree on their writer -- the ring writes whole
    blocks, so a split block means the replay is wrong.
    """
    w = ring_writers(q, W, K)
    out = []
    for b in range(len(w) // W):
        blk = set(w[b * W:(b + 1) * W])
        if len(blk) != 1:
            raise RuntimeError(f"block {b} at q={q} has writers {sorted(blk)}")
        out.append(q - blk.pop())
    return out


def age_bucket(age: int, n_age: int) -> int:
    """0-based bucket: ages 1..n_age-1 map to themselves, >= n_age to the last."""
    if age < 1:
        raise ValueError(f"a carried row has age >= 1; got {age}")
    return min(age, n_age) - 1


def column_layout(q: int, W: int, K: int, L: int, n_sum: int,
                  dual: bool = True) -> dict:
    """Column ranges [lo, hi) of each kind in chunk q's packed sequence."""
    nE = carried_rows(q, W, K)
    nZ = nE if dual else 0
    r0 = nE + nZ
    return {"nE": nE, "nZ": nZ, "L": L, "nS": n_sum,
            "E": (0, nE), "Z": (nE, nE + nZ), "real": (r0, r0 + L),
            "summary": (r0 + L, r0 + L + n_sum), "total": r0 + L + n_sum}


def position_bins(L: int, n_bins: int) -> torch.Tensor:
    """Bin index (0..n_bins-1) of each query position 0..L-1, near-equal bins."""
    if L < n_bins:
        raise ValueError(f"{L} positions cannot fill {n_bins} bins")
    return (torch.arange(L) * n_bins) // L


def onehot(idx: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(idx.long(), n).to(torch.float32)


def normalise_form(text: str) -> Optional[str]:
    """Lowercased stripped form of an alphabetic token, else None."""
    s = text.strip()
    return s.lower() if s and s.isalpha() else None


def build_form_table(vocab_text: list, min_len: int = 1):
    """vocab id -> form id (or -1).  Returns (LongTensor[V], list of forms)."""
    forms, index, out = [], {}, []
    for t in vocab_text:
        f = normalise_form(t)
        if f is None or len(f) < min_len:
            out.append(-1)
            continue
        if f not in index:
            index[f] = len(forms)
            forms.append(f)
        out.append(index[f])
    return torch.tensor(out, dtype=torch.long), forms


def bag_of_forms(token_ids: torch.Tensor, form_of: torch.Tensor,
                 n_forms: int) -> torch.Tensor:
    """Bool mask over form ids: which forms occur in `token_ids`."""
    f = form_of[token_ids.to(form_of.device).long()]
    bow = torch.zeros(n_forms, dtype=torch.bool, device=form_of.device)
    f = f[f >= 0]
    if f.numel():
        bow[f] = True
    return bow


def precision_at_k(topk_ids: torch.Tensor, form_of: torch.Tensor,
                   bow: torch.Tensor) -> torch.Tensor:
    """[N, k] token ids -> [N] fraction of the k whose form is in `bow`.

    Denominator is always k: a non-alphabetic prediction is a miss, never
    dropped, so the three references are scored on the same footing.
    """
    f = form_of[topk_ids.to(form_of.device).long()]
    hit = torch.zeros_like(f, dtype=torch.bool)
    ok = f >= 0
    hit[ok] = bow[f[ok]]
    return hit.float().mean(dim=-1)


def classify_tokens(texts: list) -> list:
    """Per-position class (index into TOKEN_CLASSES) -- see the header, part C."""
    out, prev = [], None
    for t in texts:
        s = t.strip()
        alpha = bool(s) and s.isalpha()
        word_initial = (t[:1].isspace() or prev is None or prev == ""
                        or prev[-1:].isspace() or not prev[-1:].isalnum())
        if not alpha:
            c = 3
        elif not word_initial:
            c = 2
        elif s[0].isupper() or len(s) > 3:
            c = 0
        else:
            c = 1
        out.append(c)
        prev = t
    return out


def read_masses(Pr: torch.Tensor, lay: dict, W: int, ages: list, n_age: int,
                bins_oh: torch.Tensor) -> dict:
    """Aggregate the L real-token queries' attention.  Pr: [H, L, S] (rows sum to 1).

    Sums over queries (callers divide by query counts).  Returns
      kind [H, 5]   mass on E, Z, own tokens, summary slots, first real token
      pcol [H, 3]   E/nE, Z/nZ, own/(visible own tokens = i+1)
      age  [H, 2, n_age]  E and Z mass per age bucket (one W-block per age)
      pos  [H, 3, n_bins] E, Z, own mass per query-position bin
      sum_err, summary_max  0-dim tensors (no host sync)
    """
    H, L, S = Pr.shape
    e0, e1 = lay["E"]
    z0, z1 = lay["Z"]
    r0, r1 = lay["real"]
    s0, s1 = lay["summary"]
    if r1 - r0 != L or s1 != S:
        raise ValueError(f"layout {lay} does not match a [{H},{L},{S}] block")
    mE = Pr[..., e0:e1].sum(-1)
    mZ = Pr[..., z0:z1].sum(-1)
    mO = Pr[..., r0:r1].sum(-1)
    mS = Pr[..., s0:s1].sum(-1)
    mF = Pr[..., r0]
    vis = torch.arange(1, L + 1, device=Pr.device, dtype=Pr.dtype)
    kind = torch.stack([mE, mZ, mO, mS, mF], -1).sum(1)
    pcol = torch.stack([mE / max(e1 - e0, 1), mZ / max(z1 - z0, 1), mO / vis],
                       -1).sum(1)
    age = Pr.new_zeros(H, 2, n_age)
    if len(ages) * W != e1 - e0:
        raise ValueError(f"{len(ages)} age blocks of {W} rows vs nE={e1 - e0}")
    for b, a in enumerate(ages):
        ab = age_bucket(a, n_age)
        age[:, 0, ab] += Pr[..., e0 + b * W:e0 + (b + 1) * W].sum((-1, -2))
        if z1 > z0:
            age[:, 1, ab] += Pr[..., z0 + b * W:z0 + (b + 1) * W].sum((-1, -2))
    pos = torch.stack([mE, mZ, mO], 1) @ bins_oh.to(Pr.dtype)
    return {"kind": kind, "pcol": pcol, "age": age, "pos": pos,
            "sum_err": (mE + mZ + mO + mS - 1).abs().max(),
            "summary_max": mS.max()}


def write_masses(Ps: torch.Tensor, lay: dict, bins_oh: torch.Tensor,
                 cls_oh: torch.Tensor) -> dict:
    """Aggregate the summary slots' attention.  Ps: [H, nS, S].

    Sums over slots.  kind [H, 4] (E, Z, real, summary); over the real tokens,
    renormalised: ent [H] (nats), ent_slot [nS] (head-mean), pos [H, n_bins],
    cls [H, n_cls], first [H] (share on the chunk's FIRST real token -- the
    sink), cls_ex [H, n_cls] (class shares with the first token removed and
    the rest renormalised, so a sink token's class cannot drive the lift).
    """
    e0, e1 = lay["E"]
    z0, z1 = lay["Z"]
    r0, r1 = lay["real"]
    s0, s1 = lay["summary"]
    mE = Ps[..., e0:e1].sum(-1)
    mZ = Ps[..., z0:z1].sum(-1)
    mR = Ps[..., r0:r1].sum(-1)
    mS = Ps[..., s0:s1].sum(-1)
    p = Ps[..., r0:r1] / mR.clamp_min(1e-30).unsqueeze(-1)
    ent = -(p * p.clamp_min(1e-30).log()).sum(-1)
    coh = cls_oh.to(p.dtype)
    cls = p @ coh
    p0 = p[..., 0:1]
    cls_ex = (cls - p0 * coh[0]) / (1 - p0).clamp_min(1e-12)
    return {"kind": torch.stack([mE, mZ, mR, mS], -1).sum(1),
            "ent": ent.sum(1), "ent_slot": ent.mean(0),
            "pos": (p @ bins_oh.to(p.dtype)).sum(1),
            "cls": cls.sum(1), "first": p0[..., 0].sum(1),
            "cls_ex": cls_ex.sum(1),
            "sum_err": (mE + mZ + mR + mS - 1).abs().max()}


def split_row(ids: torch.Tensor, n_chunks: int, L: int):
    """n_chunks input chunks of L tokens and their NEXT-token labels.

    Labels are shifted over the whole row (train.py's convention), so the last
    position of chunk c predicts chunk c+1's first token; the final label of
    the row is -100 when the row has no token past n_chunks * L.
    """
    need = n_chunks * L
    if ids.numel() < need:
        raise ValueError(f"row has {ids.numel()} tokens, needs {need}")
    x = ids[:need]
    y = torch.full((need,), -100, dtype=torch.long)
    tail = ids[1:need + 1]
    y[:tail.numel()] = tail
    return list(x.view(n_chunks, L)), list(y.view(n_chunks, L))


def detect_books(starts: list, tail_eos: list) -> list:
    """Row -> book index for a strided pack (consecutive windows of one document).

    `starts[i]`: row i opens a document (BOS at position 0 -- on OLMo-2 BOS ==
    EOS).  `tail_eos[i]`: row i holds an EOS past position 0, i.e. it is a
    document's padded LAST window.  A boundary falls before row i when row i
    starts one or row i-1 ended one.  A document that ends exactly on a window
    boundary and is followed by one with no BOS is undetectable; that only
    MERGES two books, which makes the spread selection more conservative.
    """
    out, b = [], 0
    for i in range(len(starts)):
        if i > 0 and (starts[i] or tail_eos[i - 1]):
            b += 1
        out.append(b)
    return out


def _middle_out(rows: list) -> list:
    """Rows ordered middle first, then alternating outward (deterministic)."""
    m = (len(rows) - 1) // 2
    order = sorted(range(len(rows)), key=lambda j: (abs(j - m), j))
    return [rows[j] for j in order]


def select_rows(usable: list, book: dict, k: int, mode: str) -> list:
    """Which dataset rows to evaluate.

    'first'   the first k usable rows -- the 2x2 / horizon convention.  On a
              strided PG-19 pack that is ~1-2 BOOKS (~21 windows each), so the
              row bootstrap would treat windows of one book as independent.
    'spread'  (default) one MIDDLE window per book, books evenly spaced over
              the pack; only when k exceeds the number of books does it take a
              second window per book (middle-out, round-robin).
    `book` maps row -> book id.  k <= 0 means every usable row.
    """
    if mode == "first" or k <= 0:
        return list(usable) if k <= 0 else list(usable[:k])
    if mode != "spread":
        raise ValueError(f"row_select must be 'first' or 'spread'; got {mode!r}")
    by_book: dict = {}
    for r in usable:
        by_book.setdefault(book[r], []).append(r)
    books = sorted(by_book)
    if k < len(books):
        books = [books[(i * len(books)) // k] for i in range(k)]
    queues = {b: _middle_out(by_book[b]) for b in books}
    out, rnd = [], 0
    while len(out) < k and any(len(q) > rnd for q in queues.values()):
        for b in books:
            if rnd < len(queues[b]) and len(out) < k:
                out.append(queues[b][rnd])
        rnd += 1
    return out


def pick_donor(row: int, usable: list, book: dict) -> Optional[int]:
    """A donor row from a DIFFERENT book: the middle window of the book half
    the pack away.  None if the pack has one detectable book."""
    books = sorted({book[r] for r in usable})
    if len(books) < 2:
        return None
    by_book: dict = {}
    for r in usable:
        by_book.setdefault(book[r], []).append(r)
    target = books[(books.index(book[row]) + len(books) // 2) % len(books)]
    return _middle_out(by_book[target])[0]


def chunk_seed(seed: int, row: int, chunk: int) -> int:
    return int(seed) + 7919 * int(row) + int(chunk)


def boot(vals, n_boot: int, seed: int) -> dict:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m, lo, hi = paired_ci(vals, n_boot, seed)
    return {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}


def r6(x):
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [r6(v) for v in x]
    if isinstance(x, float):
        return round(x, 6) if math.isfinite(x) else None
    return x


# ---------------------------------------------------------------------------
# the attention replica + instrument
# ---------------------------------------------------------------------------

class Acc:
    """Running sums for parts A and C, on the model's device, float64."""

    def __init__(self, n_blk, H, n_chunks, n_age, n_bins, n_sum, device):
        f = dict(dtype=torch.float64, device=device)
        self.n_age, self.n_bins = n_age, n_bins
        self.A_kind = torch.zeros(n_blk, H, n_chunks, len(READ_KINDS), **f)
        self.A_pcol = torch.zeros(n_blk, H, n_chunks, 3, **f)
        self.A_age = torch.zeros(n_blk, H, 2, n_age, **f)
        self.A_pos = torch.zeros(n_blk, H, 3, n_bins, **f)
        self.nq = torch.zeros(n_chunks, **f)
        self.n_age_q = torch.zeros(n_age, **f)
        self.n_bin_q = torch.zeros(n_bins, **f)
        self.row = torch.zeros(n_blk, 2, **f)
        self.row_nq = 0.0
        self.rows_EZ: list = []
        self.C_kind = torch.zeros(n_blk, H, len(WRITE_KINDS), **f)
        self.C_ent = torch.zeros(n_blk, H, **f)
        self.C_ent_slot = torch.zeros(n_blk, n_sum, **f)
        self.C_pos = torch.zeros(n_blk, H, n_bins, **f)
        self.C_cls = torch.zeros(n_blk, H, len(TOKEN_CLASSES), **f)
        self.C_cls_ex = torch.zeros(n_blk, H, len(TOKEN_CLASSES), **f)
        self.C_first = torch.zeros(n_blk, H, **f)
        self.nC = 0.0          # summary queries (slots x chunks)
        self.nC_chunks = 0.0
        self.logL = 0.0
        self.base_cls = torch.zeros(len(TOKEN_CLASSES), **f)
        self.base_cls_ex = torch.zeros(len(TOKEN_CLASSES), **f)
        self.max_sum_err = torch.zeros((), **f)
        self.max_summary = torch.zeros((), **f)
        self.max_wsum_err = torch.zeros((), **f)

    def begin_chunk(self, ctx, A: bool, C: bool):
        q, L = ctx["q"], ctx["L"]
        if A:
            self.nq[q - 1] += L
            if q >= 2:
                for a in ctx["ages"]:
                    self.n_age_q[age_bucket(a, self.n_age)] += L
                self.n_bin_q += ctx["bins_oh"].sum(0).double()
                self.row_nq += L
        if C:
            self.nC += ctx["nS"]
            self.nC_chunks += 1
            self.logL += math.log(L)
            self.base_cls += ctx["cls_oh"].mean(0).double()
            self.base_cls_ex += ctx["cls_oh"][1:].mean(0).double()

    def add_read(self, blk, P, lay, ctx):
        r0, r1 = lay["real"]
        res = read_masses(P[:, r0:r1, :], lay, ctx["W"], ctx["ages"],
                          self.n_age, ctx["bins_oh"])
        qi = ctx["q"] - 1
        self.A_kind[blk, :, qi] += res["kind"].double()
        self.A_pcol[blk, :, qi] += res["pcol"].double()
        if ctx["q"] >= 2:
            self.A_age[blk] += res["age"].double()
            self.A_pos[blk] += res["pos"].double()
            self.row[blk] += res["kind"][:, :2].mean(0).double()
        self.max_sum_err = torch.maximum(self.max_sum_err, res["sum_err"].double())
        self.max_summary = torch.maximum(self.max_summary,
                                         res["summary_max"].double())

    def add_write(self, blk, P, lay, ctx):
        s0, s1 = lay["summary"]
        res = write_masses(P[:, s0:s1, :], lay, ctx["bins_oh"], ctx["cls_oh"])
        self.C_kind[blk] += res["kind"].double()
        self.C_ent[blk] += res["ent"].double()
        self.C_ent_slot[blk] += res["ent_slot"].double()
        self.C_pos[blk] += res["pos"].double()
        self.C_cls[blk] += res["cls"].double()
        self.C_cls_ex[blk] += res["cls_ex"].double()
        self.C_first[blk] += res["first"].double()
        self.max_wsum_err = torch.maximum(self.max_wsum_err,
                                          res["sum_err"].double())

    def end_row(self):
        if self.row_nq:
            self.rows_EZ.append((self.row / self.row_nq).cpu())
        self.row.zero_()
        self.row_nq = 0.0


class AttnProbe:
    """Shadows every CausalSelfAttention.forward with an instrumented replica.

    Modes (attributes, set by the caller between forwards):
      capture_A / capture_C  accumulate parts A / C from fp32 causal probs
      verify                 compare against the ORIGINAL class forward
      block                  (scope, idx, channels) -> -inf real->carried
      force_explicit         run every call through P @ V (fidelity check)
    With no block hit and force_explicit off, the OUTPUT is computed by the
    module's own SDPA call, so the model's numbers are untouched.
    """

    def __init__(self, inner, T: int):
        tr = inner.transformer
        self.inner, self.cortex, self.T = inner, inner.cortex, int(T)
        self.n_pre, self.n_core, self.n_coda = (len(tr.prelude),
                                                len(tr.core_block), len(tr.coda))
        self.n_blk = self.n_pre + self.T * self.n_core + self.n_coda
        self.mods = []
        for kind, lst in (("prelude", tr.prelude), ("core", tr.core_block),
                          ("coda", tr.coda)):
            for i, b in enumerate(lst):
                self.mods.append((kind, i, b.attn))
        a0 = self.mods[0][2]
        self.H, self.Hkv, self.hd = a0.n_head, a0.n_kv_heads, a0.head_dim
        if self.H % self.Hkv:
            raise RuntimeError(f"n_head {self.H} not a multiple of n_kv {self.Hkv}")
        modfile = sys.modules[type(a0).__module__]
        if not hasattr(modfile, "apply_rotary_pos_emb"):
            raise RuntimeError(f"{modfile.__file__} has no apply_rotary_pos_emb; "
                               "the replica cannot match this modeling file")
        self.rope = modfile.apply_rotary_pos_emb
        self.modeling_file = getattr(modfile, "__file__", "?")
        self.capture_A = self.capture_C = self.verify = False
        self.force_explicit = False
        self.block = None
        self.ctx: Optional[dict] = None
        self.acc: Optional[Acc] = None
        self.lay: Optional[dict] = None
        self.core_calls = [0] * self.n_core
        self.verify_log: list = []
        self._bias = {}

    # -- labels -------------------------------------------------------------
    def blk_index(self, kind, i, t=None):
        if kind == "prelude":
            return i
        if kind == "core":
            return self.n_pre + self.n_core * t + i
        return self.n_pre + self.n_core * self.T + i

    def label(self, blk):
        if blk < self.n_pre:
            return f"prelude.{blk}"
        b = blk - self.n_pre
        if b < self.n_core * self.T:
            return f"core.{b % self.n_core}@t{b // self.n_core}"
        return f"coda.{b - self.n_core * self.T}"

    # -- install ------------------------------------------------------------
    def __enter__(self):
        for kind, i, attn in self.mods:
            if "forward" in vars(attn):
                raise RuntimeError(f"{kind}.{i} attention forward is already "
                                   "shadowed; refusing to stack instruments")
            attn.forward = self._make(kind, i, attn)
        return self

    def __exit__(self, *exc):
        for _, _, attn in self.mods:
            vars(attn).pop("forward", None)
        return False

    def _make(self, kind, i, attn):
        def fwd(x, freqs_cis, block_idx, mask=None, past_key_values=None):
            return self._forward(kind, i, attn, x, freqs_cis, block_idx, mask,
                                 past_key_values)
        return fwd

    # -- per-forward layout -------------------------------------------------
    def _on_forward_start(self, x):
        c = self.cortex
        nE, nZ, nS = int(c._n_pre), int(c._n_zpre), int(c._n_sum)
        S = int(x.shape[1])
        L = S - nE - nZ - nS
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("AttnProbe: forward with no chunk context set")
        if (L != ctx["L"] or nE != ctx["nE"] or nZ != nE or nS != ctx["nS"]):
            raise RuntimeError(
                f"packed layout at chunk {ctx['q']} is nE={nE} nZ={nZ} L={L} "
                f"nS={nS}; the ring rule says nE=nZ={ctx['nE']}, L={ctx['L']}, "
                f"nS={ctx['nS']}.  Refusing: every column range below would be "
                "attributed to the wrong kind.")
        r0 = nE + nZ
        self.lay = {"nE": nE, "nZ": nZ, "L": L, "nS": nS, "E": (0, nE),
                    "Z": (nE, nE + nZ), "real": (r0, r0 + L),
                    "summary": (r0 + L, S), "total": S}

    def _causal(self, S, device):
        key = (S, str(device))
        if key not in self._bias:
            self._bias[key] = torch.full((S, S), float("-inf"),
                                         device=device).triu(1)
        return self._bias[key]

    def _blocked(self, kind, i, t):
        if self.block is None:
            return None
        scope, idx, ch = self.block
        hit = (scope == "all"
               or (scope in ("prelude", "coda") and kind == scope and idx == i)
               or (scope == "core_layer" and kind == "core" and idx == i)
               or (scope == "iter" and kind == "core" and idx == t))
        if not hit:
            return None
        lay, ranges = self.lay, []
        if "E" in ch and lay["nE"]:
            ranges.append(lay["E"])
        if "Z" in ch and lay["nZ"]:
            ranges.append(lay["Z"])
        return ranges or None

    def _probs(self, q, k, bias):
        kf = k.float()
        if self.H != self.Hkv:
            kf = kf.repeat_interleave(self.H // self.Hkv, dim=1)
        s = torch.matmul(q.float(), kf.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        return torch.softmax(s + bias, dim=-1)

    def _pv(self, P, v):
        vf = v.float()
        if self.H != self.Hkv:
            vf = vf.repeat_interleave(self.H // self.Hkv, dim=1)
        return torch.matmul(P, vf)

    # -- the replica --------------------------------------------------------
    def _forward(self, kind, i, attn, x, freqs_cis, block_idx, mask, pkv):
        if mask is not None or pkv is not None:
            raise RuntimeError(
                "memory_interp: a block mask or KV cache reached the attention. "
                "The replica reproduces only the uncached causal SDPA path the "
                "chained eval forward takes; anything else would be "
                "instrumented wrongly.")
        bi = int(block_idx)
        t = None
        if kind == "prelude":
            if i == 0:
                self.core_calls = [0] * self.n_core
                self._on_forward_start(x)
            ok = bi == i
        elif kind == "core":
            t = self.core_calls[i]
            self.core_calls[i] += 1
            ok = t < self.T and bi == self.n_pre + self.n_core * t + i
        else:
            ok = bi == -(i + 1)
        if not ok:
            raise RuntimeError(f"block_idx {bi} does not match {kind}.{i} at "
                               f"t={t} (T={self.T}); the loop ran a depth this "
                               "instrument was not told about")
        blk = self.blk_index(kind, i, t)
        B, S, _ = x.shape
        q, k, v = attn.Wqkv(x).split(attn.chunks, dim=2)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        q = q.view(B, S, attn.n_head, attn.head_dim)
        k = k.view(B, S, attn.n_kv_heads, attn.head_dim)
        v = v.view(B, S, attn.n_kv_heads, attn.head_dim)
        if attn.config.qk_bias:
            q_bias, k_bias = attn.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = freqs_cis
        q, k = self.rope(q, k, cos, sin)
        if B != 1 or q.shape[2] != k.shape[2]:
            raise RuntimeError("memory_interp runs batch 1, uncached")
        ranges = self._blocked(kind, i, t)
        explicit = ranges is not None or self.force_explicit
        P = None
        if self.capture_A or self.capture_C or self.verify or explicit:
            bias = self._causal(S, x.device)
            if ranges:
                bias = bias.clone()
                r0, r1 = self.lay["real"]
                for lo, hi in ranges:
                    bias[r0:r1, lo:hi] = float("-inf")
            P = self._probs(q, k, bias)
        if explicit:
            y = self._pv(P, v).to(q.dtype)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True, enable_gqa=True)
        y = y.transpose(1, 2).reshape(B, S, attn.n_head * attn.head_dim).contiguous()
        out = attn.proj(y)
        if not explicit and P is not None:
            if self.capture_A:
                self.acc.add_read(blk, P[0], self.lay, self.ctx)
            if self.capture_C:
                self.acc.add_write(blk, P[0], self.lay, self.ctx)
            if self.verify:
                ref = type(attn).forward(attn, x, freqs_cis, block_idx, None, None)

                def project(PP):
                    yy = self._pv(PP, v).to(q.dtype).transpose(1, 2).reshape(
                        B, S, attn.n_head * attn.head_dim)
                    return attn.proj(yy.contiguous()).float()

                rf = ref.float()
                nrm = rf.norm().clamp_min(1e-30)
                d = project(P) - rf
                # NEGATIVE CONTROL: the same recompute WITHOUT the causal
                # mask.  A check that cannot tell the right P from a wrong one
                # proves nothing; this one must miss by orders of magnitude.
                neg = project(self._probs(q, k, 0.0)) - rf
                self.verify_log.append({
                    "block": self.label(blk), "chunk": self.ctx["q"],
                    "replica_bit_identical": bool(torch.equal(out, ref)),
                    "recompute_rel_err": float(d.norm() / nrm),
                    "recompute_maxabs_over_rms": float(
                        d.abs().max() / rf.pow(2).mean().sqrt().clamp_min(1e-30)),
                    "neg_control_rel_err": float(neg.norm() / nrm),
                    "prob_row_sum_err": float((P.sum(-1) - 1).abs().max())})
        return out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def refuse(msg: str) -> None:
    print(f"\nREFUSING: {msg}", flush=True)
    raise SystemExit(2)


class ring_pointer_guard:
    """Snapshot/restore the gated ring's MUTABLE write pointer around a side forward.

    `buf._chunk`, `buf._z_chunk` and `buf._z_pending` live on the buffer, a
    forward with m_cross_in=None resets them, and every real merge advances
    them -- so a side forward interleaved into a chain would make later writes
    land in the wrong ring rows, silently.  Every side forward here also
    patches merge to a no-op, so the pointer SHOULD NOT move; this restores it
    regardless and raises if it did move, because that would mean the no-op
    patch did not take.
    """

    KEYS = ("_chunk", "_z_chunk", "_z_pending")

    def __init__(self, buf):
        self.buf = buf

    def __enter__(self):
        self.snap = {}
        for k in self.KEYS:
            v = getattr(self.buf, k, None)
            self.snap[k] = list(v) if isinstance(v, list) else v
        return self

    def __exit__(self, exc_type, *exc):
        moved = []
        for k in self.KEYS:
            now, was = getattr(self.buf, k, None), self.snap[k]
            if isinstance(was, list):
                same = (isinstance(now, list) and len(now) == len(was)
                        and all(a is b for a, b in zip(now, was)))
                setattr(self.buf, k, list(was))
            else:
                same = now == was
                setattr(self.buf, k, was)
            if not same:
                moved.append(k)
        if moved and exc_type is None:
            raise RuntimeError(f"a side forward moved the ring pointer {moved}; "
                               "the no-op merge patch did not take")
        return False


def validate(inner, cfg, overrides, args) -> dict:
    cortex = getattr(inner, "cortex", None)
    if cortex is None:
        refuse("no cortex graft. " + explain_missing_cortex(cfg, overrides))
    buf = getattr(cortex, "prefix", None)
    if buf is None:
        refuse("no prefix buffer, so there is no carried column to attend to. "
               + explain_missing_cortex(cfg, overrides))
    if type(buf).__name__ != "PrefixGatedBuffer":
        refuse(f"buffer is {type(buf).__name__}; the age map is the gated "
               "ring's and nothing else's")
    if getattr(buf, "route", None) != "ring" or getattr(buf, "fill", None) != "grow":
        refuse(f"route={getattr(buf, 'route', None)!r} fill="
               f"{getattr(buf, 'fill', None)!r}; ages are derived for ring+grow")
    if int(getattr(buf, "latent_stride", 1)) != 1:
        refuse("latent_stride > 1 moves Z on its own pointer; the Z age map "
               "below would be E's and wrong")
    if not bool(getattr(buf, "carries_latent", False)):
        refuse("E-only carry: there is no Z channel and parts A/B/D would "
               "report an empty column range under the Z label")
    if cortex.latent_read != "embeds" or cortex.latent_embed is None:
        refuse(f"latent_read={cortex.latent_read!r}: Z is not spliced as "
               "columns, so 'Z columns' do not exist")
    if not cortex.e_carry_read or not cortex.latent_carry_read:
        refuse("e_carry_read / latent_carry_read is off: the spliced columns "
               "are zeros and every share would describe a null")
    if getattr(cortex, "latent_read_null", None) is not None:
        refuse("latent_read_null is set: the Z read is nulled")
    for name in ("m_cross", "accum", "ccot_direct", "m_iter", "latent_reader",
                 "latent_scratch"):
        if getattr(cortex, name, None) is not None:
            refuse(f"cortex.{name} is live: a read path that bypasses attention, "
                   "which this instrument cannot see")
    if not bool(buf.summary_seeded):
        refuse("summary_emb was never seeded -- the checkpoint's write path did "
               "not load, and the first forward would re-seed it from wte")
    if all(float(p.detach().abs().max()) == 0.0 for p in
           (buf.gate_proj_in.weight, buf.gate_proj_mem.weight)):
        refuse("both E gate projections are exactly zero (gate_init=zero, "
               "untrained): the checkpoint's cortex tensors did not load")
    if "latent_read_znorm_target" not in overrides and \
            not hasattr(cfg, "latent_read_znorm_target"):
        refuse("latent_read_znorm_target was not given; the code default (3.0) "
               "is not a trained value.  --set latent_read_znorm_target=78.9 "
               "at step 38146, 98.0 at step 11444")
    W, K = int(buf.n_vec), int(buf.n_slots)
    lap_chunks(W, K)
    return {"W": W, "K": K, "D": int(buf.hidden_size),
            "znorm_target": float(cortex.latent_read_znorm_target)}


def verify_layout(cortex, buf, model, ids, state, num_steps, q, geo, device):
    """Run chunk q once with prefix_pack tapped; check every block of the pack."""
    from cortex_graft import rescale_rows
    W, K, D = geo["W"], geo["K"], geo["D"]
    tap = {}
    real = cortex.prefix_pack

    def pack(input_embeds, position_ids, emb_scale=1.0, write=True, read=True):
        out = real(input_embeds, position_ids, emb_scale, write=write, read=read)
        tap["in"] = input_embeds.detach().clone()
        tap["out"] = out
        return out

    cortex.prefix_pack = pack
    try:
        with ring_pointer_guard(buf), _patched_merge(buf, lambda s, n, z=None: s), \
                torch.no_grad():
            model(input_ids=ids[None].to(device), num_steps=num_steps,
                  m_cross_in=state, return_m_cross=False)
    finally:
        del cortex.prefix_pack
    packed, pos, n_prefix, n_sum = tap["out"]
    L = ids.numel()
    nE = carried_rows(q, W, K)
    got = {"chunk": q, "n_prefix": int(n_prefix), "n_sum": int(n_sum),
           "packed_cols": int(packed.shape[1])}
    fails = []
    if int(n_prefix) != 2 * nE or int(n_sum) != W or packed.shape[1] != 2 * nE + L + W:
        fails.append(f"counts n_prefix={n_prefix} n_sum={n_sum} cols="
                     f"{packed.shape[1]} vs expected 2x{nE}, {W}, {2 * nE + L + W}")
    p = packed[0]
    E_carry, Z_carry = state[0, :, :D], state[0, :, D:]
    if not torch.equal(p[:nE], E_carry.to(p.dtype)):
        fails.append("columns [0, nE) are not the carried E rows")
    z_exp = cortex.latent_embed(rescale_rows(Z_carry[None].to(p.dtype),
                                             geo["znorm_target"]))[0].to(p.dtype)
    if not torch.allclose(p[nE:2 * nE].float(), z_exp.float(), rtol=1e-4, atol=1e-4):
        fails.append("columns [nE, 2nE) are not latent_embed(rescale(Z))")
    if not torch.equal(p[2 * nE:2 * nE + L], tap["in"][0]):
        fails.append("columns [2nE, 2nE+L) are not the chunk's token embeddings")
    slots = buf.summary_slots(1, p.dtype, p.device)[0]
    if not torch.equal(p[2 * nE + L:], slots):
        fails.append("the last W columns are not summary_emb")
    exp_pos = torch.cat([torch.zeros(2 * nE), torch.arange(1, L + 1),
                         torch.arange(L + 1, L + W + 1)]).to(pos.device, pos.dtype)
    if not torch.equal(pos[0], exp_pos):
        fails.append("position ids are not [0]*(nE+nZ), 1..L, L+1..L+W")
    got.update({
        "e_block_row_norm": float(p[:nE].float().norm(dim=-1).mean()),
        "z_block_row_norm": float(p[nE:2 * nE].float().norm(dim=-1).mean()),
        "token_row_norm": float(p[2 * nE:2 * nE + L].float().norm(dim=-1).mean()),
        "layout": f"[E 0..{nE - 1} | Z {nE}..{2 * nE - 1} | tokens "
                  f"{2 * nE}..{2 * nE + L - 1} | summary {2 * nE + L}.."
                  f"{2 * nE + L + W - 1}]; positions 0 (carried), 1..{L}, "
                  f"{L + 1}..{L + W}",
        "fails": fails})
    return got


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model_name", required=True,
                   help="the BASE dir (graft-prepared modeling file + tokenizer)")
    p.add_argument("--checkpoint", required=True,
                   help="the arm's checkpoint dir or chkpt.pt.  Required: the "
                        "base dir carries no trained cortex weights.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="graft flags; mirror pace/midrun_check.sbatch's cortex "
                        "SETS exactly, including latent_read_znorm_target")
    p.add_argument("--data", required=True,
                   help="save_to_disk dataset with an input_ids column")
    p.add_argument("--n_chunks", type=int, default=8)
    p.add_argument("--chunk_len", type=int, default=512)
    p.add_argument("--T", type=int, default=8,
                   help="loop depth; 8 is the trained depth (configs say 32)")
    p.add_argument("--max_rows", type=int, default=24)
    p.add_argument("--row_select", default="spread", choices=["spread", "first"],
                   help="'spread' (default): one middle window per book, books "
                        "evenly spaced -- the strided PG-19 pack is ~21 "
                        "windows per book, so the first 24 rows are ~1-2 books. "
                        "'first': the first max_rows usable rows (2x2 "
                        "convention).")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--parts", default="A,B,C,D,E")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--n_bins", type=int, default=8)
    p.add_argument("--out_dir", default="eval_results/memory_interp")
    p.add_argument("--allow_scrambled", action="store_true")
    return p.parse_args()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=os.path.dirname(os.path.abspath(
                                  __file__)), timeout=10).stdout.strip() or None
    except Exception:                                   # noqa: BLE001
        return None


def build_sites(probe) -> list:
    s = [("prelude", i, "EZ") for i in range(probe.n_pre)]
    s += [("core_layer", i, "EZ") for i in range(probe.n_core)]
    s += [("iter", t, "EZ") for t in range(probe.T)]
    s += [("coda", i, "EZ") for i in range(probe.n_coda)]
    s += [("all", None, "EZ"), ("all", None, "E"), ("all", None, "Z")]
    return s


def site_name(site) -> str:
    scope, idx, ch = site
    if scope == "all":
        return {"EZ": "all", "E": "all_E_only", "Z": "all_Z_only"}[ch]
    if scope == "core_layer":
        return f"core.{idx}@all_t"
    if scope == "iter":
        return f"iter.t{idx}@all_core"
    return f"{scope}.{idx}"


def main() -> int:
    args = parse_args()
    parts = {p.strip().upper() for p in args.parts.split(",") if p.strip()}
    if not parts or parts - set(PARTS):
        refuse(f"--parts must be a subset of {','.join(PARTS)}; got {args.parts}")
    if args.n_chunks < 2:
        refuse("--n_chunks < 2 leaves no chunk that reads a carry")
    smoke = args.dtype != "float32"
    if smoke:
        print("WARNING: --dtype bfloat16.  SMOKE ONLY.  Part B is a difference of "
              "two nearly equal losses (bf16 got that 6x wrong in P0.1); the "
              "recompute tolerance is loosened.  Do not quote these numbers.",
              flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device; this will be very slow.", flush=True)
    dtype = getattr(torch, args.dtype)
    overrides = parse_config_overrides(args.set)
    t_load = time.perf_counter()
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None, dtype,
                                 device, config_overrides=overrides or None)
    t_load = time.perf_counter() - t_load
    inner = _unwrap(model)
    geo = validate(inner, cfg, overrides, args)
    cortex, buf = inner.cortex, inner.cortex.prefix
    refuse_if_scrambled(cortex, "memory_interp", args.allow_scrambled)
    W, K, D = geo["W"], geo["K"], geo["D"]
    n_lap = lap_chunks(W, K)
    n, L, T = args.n_chunks, args.chunk_len, args.T
    num_steps = to_num_steps(T)
    for q in range(2, n + 1):              # the two derivations must agree
        w = ring_writers(q, W, K)
        for b in range(len(w) // W):
            if ring_writer_closed(q, b, W, K) != w[b * W]:
                refuse(f"ring replay and closed form disagree at q={q} b={b}")
    depth_map = [int(d) for d in cortex.latent_depth_map(T, W)]
    if device.type == "cuda":
        mem_after_load = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()

    from transformers import AutoTokenizer
    from datasets import load_from_disk
    tok = AutoTokenizer.from_pretrained(args.model_name)
    V = int(inner.lm_head.weight.shape[0])
    vocab_text = tok.batch_decode([[i] for i in range(V)],
                                  clean_up_tokenization_spaces=False)
    form_all, forms_all = build_form_table(vocab_text, 1)
    form_long, forms_long = build_form_table(vocab_text, 4)
    form_all, form_long = form_all.to(device), form_long.to(device)
    ds = load_from_disk(args.data)
    if "input_ids" not in ds.column_names:
        refuse(f"{args.data} has no input_ids column")
    # BOOK STRUCTURE.  The strided PG-19 pack is ~21 consecutive windows per
    # book; a document's last window is EOS-marked and EOS-padded (mask 0).
    # A usable row is a full window with no EOS past position 0 over the span
    # scored (BOS == EOS at position 0 opens a book and is fine).
    eos = tok.eos_token_id
    npds = ds.with_format("numpy")
    ids_col = npds["input_ids"]
    mask_col = npds["attention_mask"] if "attention_mask" in ds.column_names \
        else None
    starts, tails, usable = [], [], []
    for i in range(len(ds)):
        r = ids_col[i]
        starts.append(bool(len(r) and int(r[0]) == eos))
        tails.append(bool((r[1:] == eos).any()))
        ok = len(r) >= n * L and not bool((r[1:n * L + 1] == eos).any())
        if ok and mask_col is not None:
            ok = bool(mask_col[i][:n * L].all())
        if ok:
            usable.append(i)
    book_of = dict(enumerate(detect_books(starts, tails)))
    if not usable:
        refuse(f"no row of {args.data} holds {n} x {L} unpadded tokens")
    rows = select_rows(usable, book_of, args.max_rows, args.row_select)
    skipped = len(ds) - len(usable)
    n_books_rows = len({book_of[r] for r in rows})
    print(f"[rows] {len(rows)} rows from {n_books_rows} distinct books "
          f"({len({book_of[r] for r in usable})} books, {len(usable)} usable "
          f"rows, {skipped} skipped as short/padded); row_select="
          f"{args.row_select}", flush=True)
    if n_books_rows < len(rows):
        print(f"WARNING: {len(rows)} rows share {n_books_rows} books -- the row "
              f"bootstrap treats windows of one book as independent, so every "
              f"CI below is optimistic.", flush=True)

    probe = AttnProbe(inner, T)
    n_age = n_lap
    acc = Acc(probe.n_blk, probe.H, n, n_age, args.n_bins, W, device)
    probe.acc = acc
    sites = build_sites(probe) if "B" in parts else []
    bins_cache = {}

    def make_ctx(q, cls_ids):
        if L not in bins_cache:
            bins_cache[L] = onehot(position_bins(L, args.n_bins),
                                   args.n_bins).to(device)
        return {"q": q, "L": L, "W": W, "nS": W, "nE": carried_rows(q, W, K),
                "ages": block_ages(q, W, K) if q >= 2 else [],
                "bins_oh": bins_cache[L],
                "cls_oh": onehot(cls_ids, len(TOKEN_CLASSES)).to(device)}

    def ce_of(out, y):
        lg = out.logits[0].float()
        ce = F.cross_entropy(lg, y.to(device), reduction="none", ignore_index=-100)
        return ce, lg

    checks = {"transparency": None, "verify_calls": 0,
              "recompute_rel_err_max": None, "replica_bit_identical_all": None,
              "block_nothing_bit_identical_rows": 0, "fidelity_max_abs_nats": None,
              "block_all_differs": None, "ring_rows_checked": 0}
    layout_checks = []
    B_res = {site_name(s): [] for s in sites}
    B_chunk = {site_name(s): [[] for _ in range(n - 1)] for s in sites}
    intact_rows, intact_c1 = [], []
    D_rows: dict = {}
    E_rows: dict = {}
    examples = []
    donors: list = []
    timing = {"chain_s": [], "B_s": [], "D_s": [], "E_s": [], "bare_chain_s": None}
    t_start = time.perf_counter()

    def chain(si, xs, ys, cls, capture: bool, verify_chunks=()):
        """The intact chain.  Returns (per-token CE, states, writes, sums)."""
        states, writes, ces, sums = [None], [], [], []
        for c in range(1, n + 1):
            seen = {}
            real_merge = type(buf).merge.__get__(buf, type(buf))

            def spy(st, new_vecs, new_latent=None, _c=c):
                seen["e"] = new_vecs.detach().clone()
                seen["z"] = None if new_latent is None else new_latent.detach().clone()
                out = real_merge(st, new_vecs, new_latent)
                # THE RING RULE, CHECKED ON THE REAL MERGE.  Rows outside the
                # expected block must be bit-identical; a growing block must
                # keep its prefix and add exactly W rows.
                if st is None:
                    ok = out.shape[1] == W
                elif st.shape[1] < K:
                    ok = (out.shape[1] == st.shape[1] + W
                          and torch.equal(out[:, :st.shape[1]], st))
                else:
                    exp = set(expected_ring_rows(_c - 1, W, K))
                    keep = [r for r in range(K) if r not in exp]
                    ok = (out.shape[1] == K and torch.equal(out[:, keep], st[:, keep])
                          and set(expected_ring_rows(_c - 1, W, K))
                          == set(range(((_c - 1) * W) % K, ((_c - 1) * W) % K + W)))
                if not ok:
                    raise RuntimeError(f"the merge at chunk {_c} did not follow "
                                       "the ring rule; ages would be wrong")
                checks["ring_rows_checked"] += 1
                return out

            if capture:
                probe.ctx = make_ctx(c, cls[c - 1])
                acc.begin_chunk(probe.ctx, probe.capture_A, probe.capture_C)
                probe.verify = c in verify_chunks
            torch.manual_seed(chunk_seed(args.seed, si, c))
            with _patched_merge(buf, spy), torch.no_grad():
                out = model(input_ids=xs[c - 1][None].to(device),
                            num_steps=num_steps, m_cross_in=states[-1],
                            return_m_cross=True)
            ce, lg = ce_of(out, ys[c - 1])
            ces.append(ce)
            sums.append((float(lg.double().sum()), float(lg.double().pow(2).sum())))
            states.append(out.m_cross)
            writes.append((seen["e"], seen["z"]))
        probe.verify = False
        return ces, states, writes, sums

    def run_site(si, xs, ys, cls, states, site, explicit=False):
        """Chunks 2..n on the INTACT carry, one site blocked.  Per-token CE.

        Teacher-forced: chunk c reads the intact chain's state after c-1 and
        its own write is discarded (no-op merge), so nothing is re-chained and
        the ring pointer is restored around every forward.
        """
        out_ce = []
        probe.block, probe.force_explicit = site, explicit
        try:
            for c in range(2, n + 1):
                probe.ctx = make_ctx(c, cls[c - 1])
                torch.manual_seed(chunk_seed(args.seed, si, c))
                with ring_pointer_guard(buf), \
                        _patched_merge(buf, lambda s, nv, nl=None: s), \
                        torch.no_grad():
                    out = model(input_ids=xs[c - 1][None].to(device),
                                num_steps=num_steps, m_cross_in=states[c - 1],
                                return_m_cross=False)
                out_ce.append(ce_of(out, ys[c - 1])[0])
        finally:
            probe.block, probe.force_explicit = None, False
        return out_ce

    def mean_nll(ces, valid):
        tot = sum(float(ce[m].double().sum()) for ce, m in zip(ces, valid))
        cnt = sum(int(m.sum()) for m in valid)
        return tot / cnt

    for ri, si in enumerate(rows):
        ids = torch.tensor(ds[si]["input_ids"], dtype=torch.long)
        xs, ys = split_row(ids, n, L)
        cls = [torch.tensor(classify_tokens([vocab_text[t] for t in x.tolist()]))
               for x in xs]
        donor_si = pick_donor(si, usable, book_of)
        if donor_si is None and ri == 0:
            print("WARNING: one detectable book -- part D has no donor control "
                  "and its own-donor contrasts are omitted.", flush=True)
        donors.append(donor_si)
        dxs = split_row(torch.tensor(ds[donor_si]["input_ids"], dtype=torch.long),
                        n, L)[0] if donor_si is not None else None
        valid = [y.to(device) != -100 for y in ys]

        # ── row 0: layout + transparency (the replica must be invisible) ──
        if ri == 0:
            _sync(device)
            t0 = time.perf_counter()
            bare_ce, bare_states, _, bare_sums = chain(si, xs, ys, cls,
                                                       capture=False)
            _sync(device)
            timing["bare_chain_s"] = time.perf_counter() - t0
            for q in sorted({2, n}):
                lc = verify_layout(cortex, buf, model, xs[q - 1],
                                   bare_states[q - 1], num_steps, q, geo, device)
                layout_checks.append(lc)
                if lc["fails"]:
                    refuse(f"packed layout at chunk {q}: {lc['fails']}")

        with probe:
            probe.capture_A, probe.capture_C = "A" in parts, "C" in parts
            _sync(device)
            t0 = time.perf_counter()
            ces, states, writes, sums = chain(
                si, xs, ys, cls, capture=True,
                verify_chunks=(sorted({2, n}) if ri == 0 else ()))
            _sync(device)
            timing["chain_s"].append(time.perf_counter() - t0)
            probe.capture_A = probe.capture_C = False
            acc.end_row()

            if ri == 0:
                same = all(torch.equal(a, b) for a, b in zip(ces, bare_ce)) \
                    and sums == bare_sums
                checks["transparency"] = bool(same)
                if not same:
                    refuse("the instrumented chain is NOT bit-identical to the "
                           "bare chain -- the replica changed the model")
                vl = probe.verify_log
                checks["verify_calls"] = len(vl)
                checks["replica_bit_identical_all"] = all(
                    v["replica_bit_identical"] for v in vl)
                checks["recompute_rel_err_max"] = max(
                    v["recompute_rel_err"] for v in vl)
                checks["recompute_maxabs_over_rms_max"] = max(
                    v["recompute_maxabs_over_rms"] for v in vl)
                checks["neg_control_rel_err_min"] = min(
                    v["neg_control_rel_err"] for v in vl)
                checks["prob_row_sum_err_max"] = max(
                    v["prob_row_sum_err"] for v in vl)
                # DISCRIMINATION, call by call: the wrong P must sit >= 10x
                # further from the module output than the right P does.
                ratio = min(v["neg_control_rel_err"]
                            / max(v["recompute_rel_err"], 1e-12) for v in vl)
                checks["neg_control_discrimination_min"] = ratio
                if ratio < 10:
                    refuse(f"the recompute check cannot tell a causal P from a "
                           f"non-causal one at some call (negative control only "
                           f"{ratio:.1f}x the recompute error); it proves "
                           "nothing there")
                exp_calls = len(sorted({2, n})) * probe.n_blk
                if len(vl) != exp_calls:
                    refuse(f"verified {len(vl)} attention calls, expected "
                           f"{exp_calls}: some block never ran through the probe")
                if not checks["replica_bit_identical_all"]:
                    refuse("the replica's output differs from the original "
                           "CausalSelfAttention.forward on some call")
                if checks["recompute_rel_err_max"] > RECOMPUTE_TOL[args.dtype]:
                    worst = max(vl, key=lambda v: v["recompute_rel_err"])
                    refuse(f"proj(P @ V) misses the module output by "
                           f"{worst['recompute_rel_err']:.2e} of its RMS at "
                           f"{worst['block']} chunk {worst['chunk']} (tol "
                           f"{RECOMPUTE_TOL[args.dtype]}): the probabilities "
                           "are wrong")

            # A/C share invariants (cheap host sync once per row)
            if parts & {"A", "C"}:
                if float(acc.max_sum_err) > SHARE_SUM_TOL:
                    refuse(f"read shares sum to 1 +/- {float(acc.max_sum_err):.2e}")
                if float(acc.max_summary) != 0.0:
                    refuse(f"real tokens put {float(acc.max_summary):.3e} mass on "
                           "summary slots; the causal mask is not what the "
                           "instrument assumes")
                if float(acc.max_wsum_err) > SHARE_SUM_TOL:
                    refuse(f"write shares sum to 1 +/- {float(acc.max_wsum_err):.2e}")
                checks["read_share_sum_err_max"] = float(acc.max_sum_err)
                checks["summary_mass_from_real_tokens_max"] = float(acc.max_summary)
                checks["write_share_sum_err_max"] = float(acc.max_wsum_err)

            nll_intact = mean_nll(ces[1:], valid[1:])
            intact_rows.append(nll_intact)
            intact_c1.append(mean_nll(ces[:1], valid[:1]))

            # ── B ────────────────────────────────────────────────────────
            if "B" in parts:
                _sync(device)
                t0 = time.perf_counter()
                nothing = run_site(si, xs, ys, cls, states, ("none", None, "EZ"))
                if not all(torch.equal(a, b) for a, b in zip(nothing, ces[1:])):
                    refuse(f"row {si}: block-NOTHING is not bit-identical to "
                           "the intact chain; the pairing is broken")
                checks["block_nothing_bit_identical_rows"] += 1
                if ri == 0:
                    fid = run_site(si, xs, ys, cls, states, None, explicit=True)
                    gaps = [abs(mean_nll([a], [m]) - mean_nll([b], [m]))
                            for a, b, m in zip(fid, ces[1:], valid[1:])]
                    checks["fidelity_max_abs_nats"] = max(gaps)
                    if max(gaps) > FIDELITY_TOL:
                        msg = (f"the explicit masked path with NOTHING blocked "
                               f"misses intact by {max(gaps):.2e} nats/chunk "
                               f"(tol {FIDELITY_TOL})")
                        if smoke:
                            print(f"WARNING (bf16 smoke): {msg}", flush=True)
                        else:
                            refuse(msg)
                for site in sites:
                    sce = run_site(si, xs, ys, cls, states, site)
                    name = site_name(site)
                    B_res[name].append(mean_nll(sce, valid[1:]) - nll_intact)
                    for j, (a, b, m) in enumerate(zip(sce, ces[1:], valid[1:])):
                        B_chunk[name][j].append(mean_nll([a], [m])
                                                - mean_nll([b], [m]))
                    if ri == 0 and site == ("all", None, "EZ"):
                        checks["block_all_differs"] = not all(
                            torch.equal(a, b) for a, b in zip(sce, ces[1:]))
                        if not checks["block_all_differs"]:
                            refuse("blocking every real->carried edge changed "
                                   "nothing: the block is not reaching the "
                                   "attention")
                _sync(device)
                timing["B_s"].append(time.perf_counter() - t0)

        # ── D ────────────────────────────────────────────────────────────
        if "D" in parts:
            t0 = time.perf_counter()
            bow = {v: [bag_of_forms(x, fo, len(fl)) for x in xs]
                   for v, fo, fl in (("all", form_all, forms_all),
                                     ("long", form_long, forms_long))}
            dbow = ({v: [bag_of_forms(x, fo, len(fl)) for x in dxs]
                     for v, fo, fl in (("all", form_all, forms_all),
                                       ("long", form_long, forms_long))}
                    if dxs is not None else None)
            fo_of = {"all": form_all, "long": form_long}
            row_acc: dict = {}

            def add(key, vals):
                row_acc.setdefault(key, []).extend(vals.tolist())

            def lens(rows_, ch):
                h = rows_.to(device=inner.lm_head.weight.device,
                             dtype=inner.lm_head.weight.dtype)
                with torch.no_grad():
                    if ch == "Z":
                        h = inner.transformer.ln_f(h)
                    return inner.lm_head(h).float().topk(args.topk, dim=-1).indices

            def score(top, ch, bucket, w, extra=None, prev=None):
                for v in ("all", "long"):
                    refs = {"own": bow[v][w - 1],
                            "next": bow[v][w] if w < n else None,
                            "donor": dbow[v][w - 1] if dbow is not None else None,
                            "prev": bow[v][prev - 1] if prev else None}
                    for rname, bw in refs.items():
                        if bw is None:
                            continue
                        pk = precision_at_k(top, fo_of[v], bw)
                        add((ch, bucket, v, rname), pk)
                        if extra is not None:
                            for j, d in enumerate(extra):
                                add((ch, f"depth{d}", v, rname), pk[j:j + 1])
                alpha = (form_all[top] >= 0).float().mean(-1)
                add((ch, bucket, "alpha_frac", "-"), alpha)
                return

            for c in range(1, n + 1):                         # fresh writes
                for ch, t in (("E", writes[c - 1][0]), ("Z", writes[c - 1][1])):
                    top = lens(t[0], ch)
                    score(top, ch, "fresh", c, depth_map if ch == "Z" else None)
                    if ri == 0 and ch == "E" and c in (2, 4, 6) and c <= n:
                        own = precision_at_k(top, form_all, bow["all"][c - 1])
                        j = int(own.argmax())
                        ztop = lens(writes[c - 1][1][0, j:j + 1], "Z")[0]
                        words = tok.decode(xs[c - 1].tolist()).split()[:40]
                        examples.append({
                            "chunk": c, "slot": j, "z_depth": depth_map[j],
                            "selected_by": "max own precision@k among 16 E slots",
                            "E_top15": [vocab_text[t] for t in top[j, :15].tolist()],
                            "E_own_p": float(own[j]),
                            "E_donor_p": (float(precision_at_k(
                                top[j:j + 1], form_all, dbow["all"][c - 1])[0])
                                if dbow is not None else None),
                            "Z_top15_crude": [vocab_text[t] for t in ztop[:15].tolist()],
                            "source_first40": " ".join(words)})
            for q in range(2, n + 1):                         # ring rows as read
                st = states[q - 1][0]
                wr = ring_writers(q, W, K)
                for ch, half in (("E", st[:, :D]), ("Z", st[:, D:])):
                    top = lens(half, ch)
                    for b in range(len(wr) // W):
                        w = wr[b * W]
                        a = q - w
                        tb = top[b * W:(b + 1) * W]
                        score(tb, ch, f"age{min(a, n_age)}", w)
                        if w > n_lap:   # gate mix: does it still hold w - lap?
                            score(tb, ch, "gated", w, prev=w - n_lap)
                        else:
                            score(tb, ch, "pure", w)
            for key, vals in row_acc.items():
                D_rows.setdefault("|".join(key), []).append(sum(vals) / len(vals))
            timing["D_s"].append(time.perf_counter() - t0)

        # ── E ────────────────────────────────────────────────────────────
        if "E" in parts:
            t0 = time.perf_counter()
            for c in range(1, n + 1):
                st = states[c]
                for ch, half in (("E", st[..., :D]), ("Z", st[..., D:])):
                    rs = rank_stats(half)
                    for k_ in ("eff_rank_pr", "centred_cosine", "eff_rank_entropy_sq"):
                        E_rows.setdefault(f"{ch}|{c}|{k_}", []).append(rs[k_])
            timing["E_s"].append(time.perf_counter() - t0)

        print(f"  row {ri + 1}/{len(rows)} (ds {si}): intact {nll_intact:.4f} "
              f"chain {timing['chain_s'][-1]:.1f}s"
              + (f"  B {timing['B_s'][-1]:.1f}s" if timing["B_s"] else "")
              + (f"  D {timing['D_s'][-1]:.1f}s" if timing["D_s"] else ""),
              flush=True)
        report = build_report(args, parts, probe, acc, geo, depth_map, checks,
                              layout_checks, B_res, B_chunk, sites, intact_rows,
                              intact_c1, D_rows, E_rows, examples, timing, V,
                              rows[:ri + 1], skipped, smoke, t_load, device,
                              mem_after_load if device.type == "cuda" else None,
                              n_lap)
        report["row_books"] = [book_of[r] for r in rows[:ri + 1]]
        report["n_distinct_books"] = len(set(report["row_books"]))
        report["donor_rows"] = donors
        report["donor_books"] = [None if d is None else book_of[d] for d in donors]
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "results.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)

    print_report(report, probe)
    print(f"\nwrote {os.path.join(args.out_dir, 'results.json')}  "
          f"({time.perf_counter() - t_start:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def build_report(args, parts, probe, acc, geo, depth_map, checks, layout_checks,
                 B_res, B_chunk, sites, intact_rows, intact_c1, D_rows, E_rows,
                 examples, timing, V, rows_done, skipped, smoke, t_load, device,
                 mem_after_load, n_lap) -> dict:
    n = args.n_chunks
    lab = [probe.label(b) for b in range(probe.n_blk)]
    rep = {
        "instrument": "memory interpretability (diag_memory_interp)",
        "when": datetime.now().isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "smoke_only": bool(smoke),
        "model_name": args.model_name, "checkpoint": args.checkpoint,
        "data": args.data, "rows": list(rows_done), "rows_skipped_short": skipped,
        "config": {"n_chunks": n, "chunk_len": args.chunk_len, "T": args.T,
                   "row_select": args.row_select,
                   "dtype": args.dtype, "seed": args.seed, "boot": args.boot,
                   "parts": sorted(parts), "topk": args.topk,
                   "n_bins": args.n_bins, "set": list(args.set),
                   "W": geo["W"], "K": geo["K"], "D": geo["D"],
                   "lap_chunks": n_lap, "znorm_target": geo["znorm_target"],
                   "z_slot_depths": depth_map,
                   "matmul_precision": torch.get_float32_matmul_precision(),
                   "modeling_file": probe.modeling_file,
                   "heads": probe.H, "kv_heads": probe.Hkv, "head_dim": probe.hd},
        "checks": checks, "layout": layout_checks,
        "intact": {"chunks_2_to_n": chance_margin(intact_rows, V),
                   "chunk_1": chance_margin(intact_c1, V),
                   "per_row_chunks_2_to_n": r6(intact_rows)},
        "timing": {
            "load_s": t_load,
            "bare_chain_s_row0": timing["bare_chain_s"],
            "chain_with_AC_capture_s_per_row": _mean(timing["chain_s"]),
            "chain_s_per_row_excl_row0": _mean(timing["chain_s"][1:]),
            "B_s_per_row": _mean(timing["B_s"]),
            "B_s_per_row_excl_row0": _mean(timing["B_s"][1:]),
            "row0_note": "row 0 also runs the bare chain, the layout check, "
                         "the per-call verify (chunks 2 and n) and the fidelity "
                         "pass; the excl_row0 numbers are the steady state",
            "D_s_per_row": _mean(timing["D_s"]),
            "E_s_per_row": _mean(timing["E_s"]),
            "peak_gpu_gib_load": mem_after_load,
            "peak_gpu_gib_run": (torch.cuda.max_memory_allocated() / 2**30
                                 if device.type == "cuda" else None)},
        "block_labels": lab,
    }
    if timing["bare_chain_s"] is not None and timing["chain_s"]:
        rep["timing"]["AC_capture_overhead_s_row0"] = (timing["chain_s"][0]
                                                      - timing["bare_chain_s"])

    if "A" in parts and float(acc.nq[1:].sum()) > 0:
        nq2 = float(acc.nq[1:].sum())
        kind2 = (acc.A_kind[:, :, 1:].sum(2) / nq2)            # [blk, H, 5]
        pcol2 = (acc.A_pcol[:, :, 1:].sum(2) / nq2)
        first_c1 = acc.A_kind[:, :, 0, 4] / max(float(acc.nq[0]), 1.0)
        per_chunk = [(acc.A_kind[:, :, qi, :2].mean(1) / float(acc.nq[qi])).tolist()
                     for qi in range(1, n)]
        age = acc.A_age / acc.n_age_q.clamp_min(1).view(1, 1, 1, -1)
        pos = acc.A_pos / acc.n_bin_q.clamp_min(1).view(1, 1, 1, -1)
        rowsEZ = torch.stack(acc.rows_EZ) if acc.rows_EZ else None
        blocks = []
        for b in range(probe.n_blk):
            km, pm = kind2[b].mean(0), pcol2[b].mean(0)
            d = {"block": lab[b],
                 **{k: float(km[j]) for j, k in enumerate(READ_KINDS)},
                 "first_tok_chunk1": float(first_c1[b].mean()),
                 "E_per_col": float(pm[0]), "Z_per_col": float(pm[1]),
                 "own_per_visible_col": float(pm[2]),
                 "E_by_chunk": [r[b][0] for r in per_chunk],
                 "Z_by_chunk": [r[b][1] for r in per_chunk],
                 "E_by_age": r6(age[b, :, 0].mean(0)),
                 "Z_by_age": r6(age[b, :, 1].mean(0)),
                 "E_by_pos": r6(pos[b, :, 0].mean(0)),
                 "Z_by_pos": r6(pos[b, :, 1].mean(0)),
                 "own_by_pos": r6(pos[b, :, 2].mean(0))}
            if rowsEZ is not None:
                d["E_ci"] = boot(rowsEZ[:, b, 0].tolist(), args.boot, args.seed)
                d["Z_ci"] = boot(rowsEZ[:, b, 1].tolist(), args.boot, args.seed)
            blocks.append(d)
        ez = (kind2[..., 0] + kind2[..., 1])
        top = torch.topk(ez.flatten(), min(20, ez.numel())).indices.tolist()
        heads = []
        for f in top:
            b, h = divmod(f, probe.H)
            heads.append({"block": lab[b], "head": h,
                          "E": float(kind2[b, h, 0]), "Z": float(kind2[b, h, 1]),
                          "own": float(kind2[b, h, 2]),
                          "first_tok": float(kind2[b, h, 4]),
                          "first_tok_chunk1": float(first_c1[b, h]),
                          "E_per_col": float(pcol2[b, h, 0]),
                          "Z_per_col": float(pcol2[b, h, 1]),
                          "E_by_age": r6(age[b, h, 0]), "Z_by_age": r6(age[b, h, 1]),
                          "E_by_pos": r6(pos[b, h, 0]), "Z_by_pos": r6(pos[b, h, 1])})
        rep["A"] = {
            "queries_chunks_2_to_n": nq2,
            "age_buckets": [f"{a}" for a in range(1, n_lap)] + [f"{n_lap}+"],
            "age_queries": r6(acc.n_age_q),
            "per_col_note": "age mass / W gives per-column mass for that age",
            "blocks": blocks, "top20_memory_heads": heads,
            "per_head_E": r6(kind2[..., 0]), "per_head_Z": r6(kind2[..., 1]),
            "per_head_own": r6(kind2[..., 2]),
            "per_head_first_tok": r6(kind2[..., 4]),
            "per_head_first_tok_chunk1": r6(first_c1)}

    if "C" in parts and acc.nC > 0:
        nC = acc.nC
        ck = acc.C_kind / nC
        ent = acc.C_ent / nC
        logL = acc.logL / max(acc.nC_chunks, 1)
        cpos = acc.C_pos / nC
        ccls = acc.C_cls / nC
        base = (acc.base_cls / max(acc.nC_chunks, 1))
        base_ex = (acc.base_cls_ex / max(acc.nC_chunks, 1))
        cex = acc.C_cls_ex / nC
        cfirst = acc.C_first / nC
        blocks = []
        for b in range(probe.n_blk):
            share = ccls[b].mean(0)
            share_ex = cex[b].mean(0)
            blocks.append({
                "block": lab[b],
                **{k: float(ck[b, :, j].mean()) for j, k in enumerate(WRITE_KINDS)},
                "first_tok_share": float(cfirst[b].mean()),
                "entropy_nats": float(ent[b].mean()),
                "entropy_over_lnL": float(ent[b].mean()) / logL,
                "entropy_by_slot": r6(acc.C_ent_slot[b] / max(acc.nC_chunks, 1)),
                "pos_share": r6(cpos[b].mean(0)),
                "class_share": r6(share),
                "class_lift": r6(share / base.clamp_min(1e-12)),
                "class_share_ex_first": r6(share_ex),
                "class_lift_ex_first": r6(share_ex / base_ex.clamp_min(1e-12)),
                "per_head_E": r6(ck[b, :, 0]), "per_head_Z": r6(ck[b, :, 1]),
                "per_head_entropy": r6(ent[b])})
        rep["C"] = {"token_classes": list(TOKEN_CLASSES),
                    "class_base_rate": r6(base),
                    "class_base_rate_ex_first": r6(base_ex), "ln_L": logL,
                    "blocks": blocks}

    if "B" in parts and any(B_res.values()):
        rep["B"] = {"nll_intact_mean": _mean(intact_rows),
                    "note": "delta = blocked - intact, nats/token over chunks "
                            "2..n, paired per row; teacher-forced intact carry",
                    "sites": []}
        for s in sites:
            nm = site_name(s)
            rep["B"]["sites"].append({
                "site": nm, **boot(B_res[nm], args.boot, args.seed),
                "by_chunk": [_mean(v) for v in B_chunk[nm]],
                "per_row": r6(B_res[nm])})

    if "D" in parts and D_rows:
        out = {"lens": {"E": "lm_head(E row) -- exact head on post-ln_f states",
                        "Z": "lm_head(ln_f(Z row)) -- CRUDE lens, skips the coda"},
               "k": args.topk, "cells": {}, "contrasts": {}, "examples": examples}
        for key, vals in sorted(D_rows.items()):
            out["cells"][key] = boot(vals, args.boot, args.seed)
        for key in list(D_rows):
            ch, bucket, v, rname = key.split("|")
            if rname != "own":
                continue
            don = D_rows.get(f"{ch}|{bucket}|{v}|donor")
            if don is None:
                continue
            for other in ("own", "next", "prev"):
                a = D_rows.get(f"{ch}|{bucket}|{v}|{other}")
                if a is None or len(a) != len(don):
                    continue
                out["contrasts"][f"{ch}|{bucket}|{v}|{other}-donor"] = boot(
                    [x - y for x, y in zip(a, don)], args.boot, args.seed)
        rep["D"] = out

    if "E" in parts and E_rows:
        rep["E"] = {"note": "carry AFTER chunk c (rows 16..64); rank_stats",
                    "cells": {k: _mean(v) for k, v in E_rows.items()}}
    return rep


def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def _f(x, spec=".4f"):
    return "    -   " if x is None else format(x, spec)


def print_report(rep, probe) -> None:
    bar = "=" * 92
    print(f"\n{bar}\nmemory interp -- {rep['checkpoint']}  rows={len(rep['rows'])}"
          f" books={rep.get('n_distinct_books')} donor_books="
          f"{rep.get('donor_books')}"
          f"  {'SMOKE ONLY (bf16)' if rep['smoke_only'] else 'fp32'}\n{bar}")
    ih = rep["intact"]["chunks_2_to_n"]
    if ih["mean_nll"] is not None:
        print(f"intact NLL chunks 2..n {ih['mean_nll']:.4f}  chunk 1 "
              f"{rep['intact']['chunk_1']['mean_nll']:.4f}  chance "
              f"{ih['chance']:.4f}")
        if ih["at_chance"]:
            print("*** THE MODEL IS AT CHANCE.  NOTHING BELOW IS READABLE. ***")
    c = rep["checks"]
    print(f"checks: transparency={c['transparency']} replica_bit_identical="
          f"{c['replica_bit_identical_all']} recompute_rel_err_max="
          f"{_f(c['recompute_rel_err_max'], '.2e')} (neg control min "
          f"{_f(c.get('neg_control_rel_err_min'), '.2e')}) over "
          f"{c['verify_calls']} calls; "
          f"block_nothing_bitwise_rows={c['block_nothing_bit_identical_rows']} "
          f"fidelity={_f(c['fidelity_max_abs_nats'], '.2e')} "
          f"block_all_differs={c['block_all_differs']} "
          f"ring_merges_checked={c['ring_rows_checked']}")
    for lc in rep["layout"]:
        print(f"layout chunk {lc['chunk']}: {lc['layout']}  |E|={lc['e_block_row_norm']:.1f}"
              f" |Z|={lc['z_block_row_norm']:.1f} |tok|={lc['token_row_norm']:.2f}")
    if "A" in rep:
        A = rep["A"]
        print(f"\nA. read shares, real-token queries, chunks 2..n (head-mean)")
        print(f"  {'block':<14}{'E':>8}{'Z':>8}{'own':>8}{'first':>8}{'first@c1':>9}"
              f"{'E/col':>9}{'Z/col':>9}{'own/col':>9}   E_by_age")
        for d in A["blocks"]:
            print(f"  {d['block']:<14}{d['E']:>8.4f}{d['Z']:>8.4f}{d['own']:>8.4f}"
                  f"{d['first_tok']:>8.4f}{d['first_tok_chunk1']:>9.4f}"
                  f"{d['E_per_col']:>9.5f}{d['Z_per_col']:>9.5f}"
                  f"{d['own_per_visible_col']:>9.5f}   "
                  + " ".join(f"{x:.3f}" for x in d["E_by_age"]))
        print("  top-20 memory heads (E+Z mass):")
        for h in A["top20_memory_heads"]:
            print(f"    {h['block']:<14} h{h['head']:<3} E {h['E']:.4f} Z {h['Z']:.4f}"
                  f" own {h['own']:.4f} first {h['first_tok']:.4f} "
                  f"(c1 {h['first_tok_chunk1']:.4f})  E_age "
                  + " ".join(f"{x:.3f}" for x in h["E_by_age"]))
    if "B" in rep:
        B = rep["B"]
        print(f"\nB. attention blocking, real->carried, delta nats/token "
              f"(intact {B['nll_intact_mean']:.4f})")
        print(f"  {'site':<20}{'delta':>10}{'95% CI':>24}{'n':>4}")
        for s in B["sites"]:
            sig = "" if (s["lo"] is None or s["n"] < 2
                         or s["lo"] <= 0 <= s["hi"]) else " *"
            print(f"  {s['site']:<20}{_f(s['mean'], '.5f'):>10}   "
                  f"[{_f(s['lo'], '.5f')}, {_f(s['hi'], '.5f')}]{s['n']:>4}{sig}")
    if "C" in rep:
        C = rep["C"]
        T = probe.T
        keep = {f"core.{i}@t{T - 1}" for i in range(probe.n_core)} | \
               {f"coda.{i}" for i in range(probe.n_coda)}
        print(f"\nC. summary-slot attention (write side); base rates "
              + " ".join(f"{k} {v:.3f}" for k, v in
                         zip(C["token_classes"], C["class_base_rate"])))
        print(f"  {'block':<14}{'E':>7}{'Z':>7}{'real':>7}{'slots':>7}{'H/lnL':>7}"
              f"{'first':>7}   lift ex-first(content,function,cont,other)   "
              f"pos_share")
        for d in C["blocks"]:
            if d["block"] not in keep:
                continue
            print(f"  {d['block']:<14}{d['E']:>7.3f}{d['Z']:>7.3f}{d['real']:>7.3f}"
                  f"{d['summary']:>7.3f}{d['entropy_over_lnL']:>7.3f}"
                  f"{d['first_tok_share']:>7.3f}   "
                  + " ".join(f"{x:5.2f}" for x in d["class_lift_ex_first"]) + "   "
                  + " ".join(f"{x:.2f}" for x in d["pos_share"]))
        print("  (E/Z/real/slots = mass of the 16 slot queries; H, first, lift and "
              "pos are over the real tokens, renormalised)")
    if "D" in rep:
        Dd = rep["D"]
        print(f"\nD. logit lens precision@{Dd['k']} (all alphabetic forms); "
              "Z is a CRUDE lens")
        print(f"  {'cell':<16}{'own':>8}{'next':>8}{'donor':>8}{'prev':>8}"
              f"{'own-donor':>12}{'95% CI':>22}")
        buckets = ["fresh", "age1", "age2", "age3", f"age{probe_age(rep)}",
                   "pure", "gated"]
        for ch in ("E", "Z"):
            for bk in dict.fromkeys(buckets):
                g = lambda r: Dd["cells"].get(f"{ch}|{bk}|all|{r}", {}).get("mean")
                cn = Dd["contrasts"].get(f"{ch}|{bk}|all|own-donor")
                if g("own") is None:
                    continue
                print(f"  {ch + ' ' + bk:<16}{_f(g('own')):>8}{_f(g('next')):>8}"
                      f"{_f(g('donor')):>8}{_f(g('prev')):>8}"
                      f"{_f(cn and cn['mean'], '+.4f'):>12}   "
                      f"[{_f(cn and cn['lo'], '+.4f')}, {_f(cn and cn['hi'], '+.4f')}]")
        for ex in Dd["examples"]:
            print(f"\n  example: chunk {ex['chunk']} slot {ex['slot']} "
                  f"({ex['selected_by']}) own p {ex['E_own_p']:.2f} donor p "
                  f"{_f(ex['E_donor_p'], '.2f')}")
            print(f"    source: {ex['source_first40']}")
            print(f"    E top15: {ex['E_top15']}")
            print(f"    Z top15 (crude, depth {ex['z_depth']}): {ex['Z_top15_crude']}")
    if "E" in rep:
        cells = rep["E"]["cells"]
        print("\nE. ring geometry after chunk c (PR rank / centred cos)")
        cs = sorted({int(k.split("|")[1]) for k in cells})
        for ch in ("E", "Z"):
            print(f"  {ch}: " + "  ".join(
                f"c{c} {cells[f'{ch}|{c}|eff_rank_pr']:.2f}/"
                f"{cells[f'{ch}|{c}|centred_cosine']:.2f}" for c in cs))
    t = rep["timing"]
    print(f"\ntiming s/row: chain(+A,C capture) {_f(t['chain_with_AC_capture_s_per_row'], '.1f')}"
          f" (excl row0 {_f(t['chain_s_per_row_excl_row0'], '.1f')})"
          f"  bare chain(row0) {_f(t['bare_chain_s_row0'], '.1f')}  B "
          f"{_f(t['B_s_per_row'], '.1f')} (excl row0 "
          f"{_f(t['B_s_per_row_excl_row0'], '.1f')})  D "
          f"{_f(t['D_s_per_row'], '.1f')}  E "
          f"{_f(t['E_s_per_row'], '.2f')}  load {t['load_s']:.0f}s  peak GPU "
          f"{_f(t['peak_gpu_gib_run'], '.2f')} GiB (load {_f(t['peak_gpu_gib_load'], '.2f')})")


def probe_age(rep) -> int:
    return int(rep["config"]["lap_chunks"])


if __name__ == "__main__":
    raise SystemExit(main())
