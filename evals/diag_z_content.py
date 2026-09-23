"""
Z attempt 2, STEP 0 -- the two forward-only diagnostics D1 and D2.
Findings doc, section "Attempt 2", Step 0.

WHY THESE COME BEFORE ANY TRAINING.  Attempt 1 never separated a WRITE problem
from a READ problem.  The tier 1.5 oracle trained a read on a frozen write and
found nothing (real vs donor -0.0001 nats), and the no-read check showed the
trained read does not beat no read at all (job 13450717).  That is equally
well explained by "the read cannot find the content" and by "the write holds
no content to find".  These two probes ask the write directly.

  D1  SEED vs DOCUMENT.  Run the same chunk, with the SAME incoming carry, under
      two s0 seeds.  The only thing that differs is the trunc_normal_ noise
      initialize_state draws.  Compare how far each encoding moves across seeds
      with how far it moves across documents (another book, same chunk index):

          ratio = seed_var / doc_var,   seed_var = E||X(seed a) - X(seed b)||^2 / 2
                                        doc_var  = E||X(doc i)  - X(doc j) ||^2 / 2

      doc_var contains the seed noise as well (two documents are two draws), so
      ratio is the share of the encoding's variance that is s0 noise:
      ~1 = content-free, ~0 = content-determined.  Motivation: d_1 is 23.8x s0,
      so the early deltas are mostly the loop walking AWAY from its random start,
      and the carried Z may be partly the previous chunk's noise draw.  Reported
      per encoding and per loop depth.

  D2  WHAT Z HOLDS BEYOND E.  Kernel ridge probes on [E] against [E, Z] against
      [E, donor Z], predicting three targets on held-out BOOKS:
          bow_self   this chunk's bag of words (what the write recorded)
          bow_next   the next chunk's bag of words (what a carry is for)
          nll_next   the next chunk's mean NLL
      Z HOLDS CONTENT BEYOND E only if BOTH gains clear zero:
          gain over E    R^2(E+Z) - R^2(E)
          content gain   R^2(E+Z) - R^2(E + another book's Z)
      Neither alone is enough.  Extra dimensions can move a ridge fit, which is
      what the donor controls for (same dimensions, scale and statistics,
      differing only in whether the content is this document's; never noise,
      the p30 S5 lesson).  But the donor block can also HURT the fit, and then
      real-minus-donor is positive for a Z that adds nothing: the test suite's
      planted copy of E scored a content gain of +0.225 with a gain over E of
      ~0.  So the donor is a second bar, not the baseline.

ENCODINGS, all [W, D] per chunk so every block has the same width:
  E        the E write: the summary slots' post-ln_f states (merge's new_vecs)
  Z_delta  the Z write as built: staggered deltas, band 2..9 (merge's new_latent)
  Z_end    the loop endpoint s_T at the summary columns (latent_states)
  Z_tok    s_T at the last `tok_rows` REAL token columns, mean-pooled by
           `tok_pool` -- the text's own working state, not the summary slots'

READING THE OUTPUT (cuts are CHOSEN, not measured -- say so if quoting them):
  D1 ratio > 0.5  NOISE-DOMINATED: most of the encoding is s0's random draw
       0.1..0.5   MIXED
       < 0.1      CONTENT-DETERMINED
  D2 both gains' CIs above zero on bow_next: Z holds usable content E lacks.
     R^2(E) on bow_self must be clearly positive, or the probe is not measuring
     anything and no other row is interpretable (the probe's own positive
     control, printed as PROBE HEALTH).

TRAPS THIS IS WRITTEN AROUND
  * FLOAT32.  Late deltas are ~1% of ||s_t||; bf16 invented a plateau at 6x the
    true value in P0.1.  The tool refuses bf16.
  * The seed has to reach s0 and ONLY s0.  initialize_state draws from the
    global RNG, so torch.manual_seed right before the forward controls it.  The
    tool proves it on the first sample: seed a run twice must agree (repeat
    error far below the seed effect), and seed b must differ.  Either failing
    aborts, because "seed variance" would then be measuring something else.
  * The incoming carry is IDENTICAL across the two seeds (the chain follows
    seed a; seed b re-runs the chunk from the same m_in, cloned).  Otherwise the
    seed effect would include the previous chunk's noise twice over.
  * PG-19's strided pack is consecutive windows of each book, in book order,
    with bos only on each book's first window.  A random row split would put
    windows of one book on both sides and inflate every R^2.  Books are
    recovered from the bos and the split and the bootstrap are BY BOOK.
  * Ragged rows (a book's last window, padded) are skipped, and counted.

USAGE
  python evals/diag_z_content.py --model_name ckpts/olmo-retrofit-cortex \\
      --checkpoint <probe checkpoint dir> --data data/pg19_olmo_validation_len4096_strided \\
      <--set flags for the arm> --T 8 --n_chunks 8 --out_dir eval_results/z2_content/a3z
  Launcher: pace/z2_diag_content.sbatch (array over a2, a3z).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import load_checkpoint, to_num_steps, _unwrap  # noqa: E402
from model_utils import parse_config_overrides  # noqa: E402

ENCODINGS = ("E", "Z_delta", "Z_end", "Z_tok")
Z_ENCODINGS = ENCODINGS[1:]
TARGETS = ("bow_self", "bow_next", "nll_next")

#: D1 reading cuts.  CHOSEN, not measured.
NOISE_CUT = 0.5
CONTENT_CUT = 0.1
#: The reseed check: seed a run twice must differ by less than this fraction of
#: the seed a vs seed b difference, or the RNG is not what controls s0.
REPEAT_TOL = 1e-3
#: D2 claims nothing below this many held-out books: a book-cluster bootstrap
#: over 2 books produced CIs reaching +600 in the local smoke.  Chosen.
MIN_TEST_BOOKS = 5
#: D2's own positive control: E must predict its own chunk's words at least
#: this well, or the probe is not measuring.  Chosen.
PROBE_HEALTH_R2 = 0.02


# ─── the capture ────────────────────────────────────────────────────────────

def capture(model, cortex, xc: torch.Tensor, m_in: Optional[torch.Tensor],
            num_steps, device, seed: int, tok_rows: int, tok_pool: int) -> dict:
    """One forward of one chunk with s0 drawn from `seed`.  Returns the four
    encodings (fp32, CPU), the per-depth tape, the logits and the new carry.

    The E and Z writes are taken from `prefix.merge`'s own arguments, i.e. what
    the buffer actually receives, rather than re-derived: a re-derivation is a
    second implementation that can drift from the first.
    """
    buf = cortex.prefix
    real_merge = buf.merge
    had_merge = "merge" in vars(buf)
    cap: dict = {}

    def wrapped_merge(state, new_vecs, new_latent=None):
        cap["E"] = new_vecs.detach()
        cap["Z"] = None if new_latent is None else new_latent.detach()
        return real_merge(state, new_vecs, new_latent)

    buf.merge = wrapped_merge
    torch.manual_seed(seed)
    try:
        with torch.no_grad():
            out = model(input_ids=xc.unsqueeze(0).to(device),
                        num_steps=num_steps,
                        m_cross_in=None if m_in is None else m_in.clone(),
                        return_m_cross=True,
                        output_details={"return_logits": True,
                                        "return_latents": True,
                                        "return_head": False,
                                        "return_stats": False})
    finally:
        # Delete the shadow rather than assigning the bound method back; see
        # diag_latent_scale.record_trajectory for why.
        if had_merge:
            buf.merge = real_merge
        else:
            del buf.merge

    if "E" not in cap:
        raise RuntimeError("prefix.merge never fired: this forward wrote no "
                           "carry, so there is no E or Z write to measure.")
    if cap["Z"] is None:
        raise RuntimeError("merge received no Z write (new_latent is None). "
                           "latent_carry is off on this build -- check the "
                           "--set flags (RED 12).")
    n_pre, n_sum = int(cortex._n_pre), int(cortex._n_sum)
    S = int(xc.numel())
    ls = getattr(out, "latent_states", None)
    if ls is None:
        raise RuntimeError("the forward returned no latent_states.")
    if ls.shape[1] != n_pre + S + n_sum:
        raise RuntimeError(
            f"latent_states has {ls.shape[1]} columns, expected n_pre + S + "
            f"n_sum = {n_pre} + {S} + {n_sum}.  The packed layout is not "
            "[carry | tokens | summary] and every column slice below is wrong.")
    z_end = ls[0, S + n_pre:].float()
    # SELF-CHECK: the tape's last state is s_T at the summary columns, so it
    # must equal latent_states' last n_sum columns.  If not, the columns this
    # tool calls "summary" are not the ones the write reads.
    if cortex._z_prev is None or not torch.allclose(
            cortex._z_prev[0].float(), z_end, atol=1e-4, rtol=1e-4):
        raise RuntimeError("self-check failed: the Z tape's final state is not "
                           "latent_states' summary columns.")
    if tok_rows > S or tok_rows % tok_pool:
        raise ValueError(f"tok_rows={tok_rows} must be <= S={S} and divisible "
                         f"by tok_pool={tok_pool}")
    tok = ls[0, n_pre + S - tok_rows:n_pre + S].float()
    tok = tok.reshape(tok_rows // tok_pool, tok_pool, -1).mean(dim=1)
    tape = torch.stack([d[0].float() for d in cortex._z_tape])   # [T, n_sum, D]
    return {
        "E": cap["E"][0].float().cpu(),
        "Z_delta": cap["Z"][0].float().cpu(),
        "Z_end": z_end.cpu(),
        "Z_tok": tok.cpu(),
        "tape": tape.cpu(),
        "logits": out.logits[0].float(),
        "state": getattr(out, "m_cross", None),
    }


def _row_sq(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean over rows of the squared L2 distance, fp64 for the reduction."""
    return float((a.double() - b.double()).pow(2).sum(dim=-1).mean())


def collect(model, cortex, rows, n_chunks: int, num_steps, device, seed: int,
            d1_rows: set, tok_rows: int, tok_pool: int,
            log_every: int = 25) -> tuple[list, dict]:
    """Chain every row through its chunks.  One record per (row, chunk).

    rows: list of (row_index, book_id, ids [L]).  d1_rows: row indices that
    also get the seed-b re-run (and keep their tape for D1's depth table).
    Returns (samples, reseed_check).
    """
    samples, check = [], None
    for n_done, (ri, book, ids) in enumerate(rows):
        x, y = ids[:-1], ids[1:]
        keep = (x.numel() // n_chunks) * n_chunks
        xs = list(torch.chunk(x[:keep], n_chunks))
        ys = list(torch.chunk(y[:keep], n_chunks))
        state = None
        for i, (xc, yc) in enumerate(zip(xs, ys)):
            seed_a = seed + 1_000_003 * ri + 101 * i
            a = capture(model, cortex, xc, state, num_steps, device, seed_a,
                        tok_rows, tok_pool)
            nll = float(F.cross_entropy(a["logits"], yc.to(a["logits"].device)))
            rec = {"row": ri, "book": book, "chunk": i, "ids": xc.clone(),
                   "nll": nll}
            rec.update({k: a[k] for k in ENCODINGS})
            if ri in d1_rows:
                b = capture(model, cortex, xc, state, num_steps, device,
                            seed_a + 7_777_777, tok_rows, tok_pool)
                rec["seed_sq"] = {k: _row_sq(a[k], b[k]) for k in ENCODINGS}
                rec["tape"] = a["tape"]
                rec["tape_seed_sq"] = [_row_sq(a["tape"][t], b["tape"][t])
                                       for t in range(a["tape"].shape[0])]
                if check is None:
                    again = capture(model, cortex, xc, state, num_steps,
                                    device, seed_a, tok_rows, tok_pool)
                    check = {k: {"repeat_sq": _row_sq(a[k], again[k]),
                                 "seed_sq": rec["seed_sq"][k]}
                             for k in ENCODINGS}
            samples.append(rec)
            state = a["state"]
        if log_every and (n_done + 1) % log_every == 0:
            print(f"  {n_done + 1}/{len(rows)} rows", flush=True)
    return samples, (check or {})


def reseed_verdict(check: dict) -> Optional[str]:
    """None if the seed controls s0 and nothing else; else the failure."""
    if not check:
        return "no D1 sample ran, so the reseed was never checked"
    for k, c in check.items():
        if c["seed_sq"] <= 0.0 and k != "E":
            return (f"seed b did not move {k} at all: torch.manual_seed is not "
                    "what draws s0 on this build")
        if c["seed_sq"] > 0 and c["repeat_sq"] > REPEAT_TOL * c["seed_sq"]:
            return (f"seed a run twice differs on {k} by {c['repeat_sq']:.3e} "
                    f"against a seed effect of {c['seed_sq']:.3e}: something "
                    "other than the seed is random in this forward")
    return None


# ─── books, donors, splits ──────────────────────────────────────────────────

def is_book_end(ids: list, mask: Optional[list], eos: Optional[int]) -> bool:
    """Does this row hold the end of its document?

    `stride_windows` marks a document's end with ONE eos after its last token
    and pads the rest of the row with mask 0.  Padding alone misses a tail of
    exactly row_len - 1 tokens: the eos marker fills the row and nothing is
    padded.  So a row ends a book if it is padded OR its last token is eos
    (PG-19 text never contains the special eos token, so a full mid-book
    window cannot end on one by accident).
    """
    if mask is not None and min(mask) == 0:
        return True
    return eos is not None and int(ids[-1]) == int(eos)


def pack_eos(rows, config_eos: Optional[int]) -> tuple[Optional[int], str]:
    """The eos the PACK was built with: the padding token of a ragged row.

    Not the model config's.  The B2-family configs carry Huginn's token ids
    (eos_token_id 65505) while the PG-19 packs were padded with the OLMo
    tokenizer's eos, so the config value would silently never match and every
    unpadded book end would be missed.
    """
    for r in rows:
        m = r.get("attention_mask")
        if m is not None and min(m) == 0:
            return int(r["input_ids"][list(m).index(0)]), "from the pack's padding"
    return config_eos, f"no ragged row; using the config's {config_eos}"


def book_ids(ragged: list, first_tokens: Optional[list] = None,
             bos_id: Optional[int] = None,
             fallback_block: int = 20) -> tuple[list, str]:
    """Book id per row.  Rows are consecutive windows in book order.

    PRIMARY: document ends.  `stride_windows` gives every book exactly one
    ragged last window (EOS-marked, mask 0 on the padding), and the PG-19 packer
    keeps those rows by default (MIN_TOKENS=0), so a new book starts on the row
    AFTER each ragged row.  (A book ending exactly on a window boundary merges
    with the next one: rare, and it only makes a "book" bigger.)
    SECONDARY: a bos opening each book, only when --bos_id is passed -- the
    packer's --prepend_bos is OFF by default, so there usually is none.
    Falls back to fixed contiguous blocks, LOUDLY, when neither marks two books.
    """
    n = len(ragged)
    if sum(bool(r) for r in ragged) >= 2:
        out, b = [], 0
        for r in ragged:
            out.append(b)
            if r:
                b += 1
        return out, f"document ends: {len(set(out))} books"
    if bos_id is not None and first_tokens is not None:
        starts = [int(t) == int(bos_id) for t in first_tokens]
        if sum(starts) >= 2:
            out, b = [], -1
            for i, s in enumerate(starts):
                if s or i == 0:
                    b += 1
                out.append(b)
            return out, f"bos {bos_id}: {b + 1} books"
    out = [i // fallback_block for i in range(n)]
    return out, (f"FALLBACK: neither document ends nor a bos mark 2 books, so "
                 f"books are contiguous blocks of {fallback_block} rows.  Splits "
                 "may leak a book across train and test.")


def donor_index(books: list, chunks: list) -> list:
    """For each sample, a sample from a DIFFERENT book at the SAME chunk index.

    Same chunk index because chunk 1 has no incoming carry and later chunks do,
    so the encodings' statistics depend on it.  A roll by half the group, then a
    walk forward past any same-book pick.
    """
    n = len(books)
    by_chunk: dict = {}
    for i in range(n):
        by_chunk.setdefault(chunks[i], []).append(i)
    donor = [-1] * n
    for c, idx in by_chunk.items():
        m = len(idx)
        if len({books[i] for i in idx}) < 2:
            raise ValueError(f"chunk {c}: every sample is from one book, so no "
                             "donor from another document exists")
        for p, i in enumerate(idx):
            q = (p + max(1, m // 2)) % m
            while books[idx[q]] == books[i]:
                q = (q + 1) % m
            donor[i] = idx[q]
    return donor


# ─── D1 ─────────────────────────────────────────────────────────────────────

def _book_boot_ratio(num: list, den: list, books: list, n_boot: int,
                     seed: int) -> tuple:
    """Ratio of means with a book-cluster bootstrap CI."""
    ub = sorted(set(books))
    pos = {b: i for i, b in enumerate(ub)}
    nb = torch.zeros(len(ub), dtype=torch.float64)
    db = torch.zeros(len(ub), dtype=torch.float64)
    for x, y, b in zip(num, den, books):
        nb[pos[b]] += x
        db[pos[b]] += y
    point = float(nb.sum() / db.sum()) if db.sum() > 0 else float("nan")
    g = torch.Generator().manual_seed(seed)
    w = torch.multinomial(torch.full((len(ub),), 1.0), len(ub) * n_boot,
                          replacement=True, generator=g).view(n_boot, len(ub))
    counts = torch.zeros(n_boot, len(ub), dtype=torch.float64)
    counts.scatter_add_(1, w, torch.ones_like(w, dtype=torch.float64))
    r = (counts @ nb) / (counts @ db).clamp_min(1e-300)
    lo, hi = torch.quantile(r, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return point, float(lo), float(hi)


def d1_reading(ratio: float) -> str:
    if ratio != ratio:
        return "n/a"
    if ratio > NOISE_CUT:
        return "NOISE-DOMINATED"
    if ratio < CONTENT_CUT:
        return "CONTENT-DETERMINED"
    return "MIXED"


def d1_report(samples: list, n_boot: int, seed: int) -> dict:
    sub = [s for s in samples if "seed_sq" in s]
    if len(sub) < 4:
        raise ValueError(f"D1 has {len(sub)} samples; need at least 4")
    books = [s["book"] for s in sub]
    donor = donor_index(books, [s["chunk"] for s in sub])
    out = {"n_samples": len(sub), "n_books": len(set(books)), "encodings": {},
           "depths": []}
    for k in ENCODINGS:
        seed_half = [s["seed_sq"][k] / 2 for s in sub]
        doc_half = [_row_sq(s[k], sub[donor[i]][k]) / 2
                    for i, s in enumerate(sub)]
        r, lo, hi = _book_boot_ratio(seed_half, doc_half, books, n_boot, seed)
        norm = float(torch.stack([s[k] for s in sub]).norm(dim=-1).mean())
        out["encodings"][k] = {
            "row_norm": norm,
            "seed_var": sum(seed_half) / len(sub),
            "doc_var": sum(doc_half) / len(sub),
            "ratio": r, "ci_lo": lo, "ci_hi": hi, "reading": d1_reading(r)}
    T = sub[0]["tape"].shape[0]
    for t in range(T):
        seed_half = [s["tape_seed_sq"][t] / 2 for s in sub]
        doc_half = [_row_sq(s["tape"][t], sub[donor[i]]["tape"][t]) / 2
                    for i, s in enumerate(sub)]
        r, lo, hi = _book_boot_ratio(seed_half, doc_half, books, n_boot, seed)
        norm = float(torch.stack([s["tape"][t] for s in sub])
                     .norm(dim=-1).mean())
        out["depths"].append({"t": t + 1, "delta_norm": norm, "ratio": r,
                              "ci_lo": lo, "ci_hi": hi})
    zs = {k: out["encodings"][k]["ratio"] for k in Z_ENCODINGS}
    out["lowest_ratio_z"] = min(zs, key=lambda k: zs[k])
    return out


# ─── D2 ─────────────────────────────────────────────────────────────────────

def build_vocab(chunk_ids: list, vocab_size: int, skip_top: int,
                size: int) -> torch.Tensor:
    """Token ids by document frequency over TRAIN chunks, dropping the
    `skip_top` most frequent (function words) and keeping the next `size`."""
    df = torch.zeros(vocab_size, dtype=torch.long)
    for ids in chunk_ids:
        df[torch.unique(ids)] += 1
    order = torch.argsort(df, descending=True)
    order = order[df[order] > 0]
    return order[skip_top:skip_top + size]


def bow(ids: torch.Tensor, vocab: torch.Tensor, vocab_size: int) -> torch.Tensor:
    counts = torch.bincount(ids, minlength=vocab_size).double()
    return torch.log1p(counts[vocab])


def standardized_kernel(X: torch.Tensor, train: torch.Tensor,
                        device) -> torch.Tensor:
    """Linear kernel on per-dim z-scored features (train statistics), divided
    by the width so blocks of any width contribute on the same footing."""
    X = X.to(device=device, dtype=torch.float32)
    mu = X[train].mean(dim=0)
    sd = X[train].std(dim=0).clamp_min(1e-6)
    X = (X - mu) / sd
    return (X @ X.T).double() / X.shape[1]


def _fold_ids(groups: torch.Tensor, n_folds: int) -> torch.Tensor:
    ug = torch.unique(groups)
    fold_of = {int(g): i % n_folds for i, g in enumerate(ug.tolist())}
    return torch.tensor([fold_of[int(g)] for g in groups.tolist()])


def ridge_eval(K: torch.Tensor, Ys: dict, tr: torch.Tensor, te: torch.Tensor,
               groups_tr: torch.Tensor, lam_rel: torch.Tensor,
               n_folds: int) -> dict:
    """Dual (kernel) ridge.  Lambda per target by book-grouped CV on train.

    Returns, per target: test R^2, the chosen lambda (relative to the mean of
    diag K_train), and per-test-sample SSE and SST for paired bootstraps.
    """
    dev = K.device
    Ktr = K[tr][:, tr]
    lams = lam_rel.to(dev, torch.float64) * Ktr.diagonal().mean()
    folds = _fold_ids(groups_tr, n_folds).to(dev)
    cv = {t: torch.zeros(len(lams), dtype=torch.float64, device=dev) for t in Ys}
    for f in range(n_folds):
        fit, val = (folds != f), (folds == f)
        if not bool(val.any()) or not bool(fit.any()):
            continue
        evals, U = torch.linalg.eigh(Ktr[fit][:, fit])
        KvU = Ktr[val][:, fit] @ U
        for t, Y in Ys.items():
            Yt = Y[tr].to(dev, torch.float64)
            mu = Yt[fit].mean(dim=0)
            UtY = U.T @ (Yt[fit] - mu)
            for li, lam in enumerate(lams):
                pred = KvU @ (UtY / (evals + lam).unsqueeze(1)) + mu
                cv[t][li] += (Yt[val] - pred).pow(2).sum()
    evals, U = torch.linalg.eigh(Ktr)
    KeU = K[te][:, tr] @ U
    out = {}
    for t, Y in Ys.items():
        Yt = Y[tr].to(dev, torch.float64)
        Ye = Y[te].to(dev, torch.float64)
        mu = Yt.mean(dim=0)
        li = int(torch.argmin(cv[t]))
        pred = KeU @ ((U.T @ (Yt - mu)) / (evals + lams[li]).unsqueeze(1)) + mu
        sse = (Ye - pred).pow(2).sum(dim=1)
        sst = (Ye - mu).pow(2).sum(dim=1)
        out[t] = {"r2": float(1 - sse.sum() / sst.sum()),
                  "lambda_rel": float(lam_rel[li]),
                  "lambda_at_edge": li in (0, len(lams) - 1),
                  "sse": sse.cpu(), "sst": sst.cpu()}
    return out


def book_boot_gain(sse_a: torch.Tensor, sse_b: torch.Tensor, sst: torch.Tensor,
                   books: list, n_boot: int, seed: int) -> tuple:
    """R^2_a - R^2_b = sum(sse_b - sse_a) / sum(sst), book-cluster bootstrap."""
    d = (sse_b - sse_a).double().tolist()
    return _book_boot_ratio(d, sst.double().tolist(), books, n_boot, seed)


def d2_report(samples: list, vocab_size: int, device,
              test_every: int, n_folds: int, bow_vocab: int, bow_skip_top: int,
              n_boot: int, seed: int, lam_rel=None) -> dict:
    by_key = {(s["row"], s["chunk"]): s for s in samples}
    use = [s for s in samples if (s["row"], s["chunk"] + 1) in by_key]
    books_all = sorted({s["book"] for s in use})
    test_books = {b for i, b in enumerate(books_all)
                  if i % test_every == test_every - 1}
    if len(test_books) < 2 or len(books_all) - len(test_books) < 2:
        # The donor has to come from ANOTHER book inside the same split.
        raise ValueError(
            f"D2 needs >= 2 books on each side of the split; {len(books_all)} "
            f"books with --test_every {test_every} gives {len(test_books)} "
            "test books")
    is_te = torch.tensor([s["book"] in test_books for s in use])
    tr = torch.nonzero(~is_te).flatten()
    te = torch.nonzero(is_te).flatten()
    books = [s["book"] for s in use]
    groups = torch.tensor(books)

    vocab = build_vocab([use[i]["ids"] for i in tr.tolist()], vocab_size,
                        bow_skip_top, bow_vocab)
    Ys = {
        "bow_self": torch.stack([bow(s["ids"], vocab, vocab_size) for s in use]),
        "bow_next": torch.stack([bow(by_key[(s["row"], s["chunk"] + 1)]["ids"],
                                     vocab, vocab_size) for s in use]),
        "nll_next": torch.tensor(
            [[by_key[(s["row"], s["chunk"] + 1)]["nll"]] for s in use],
            dtype=torch.float64),
    }

    # Donors within each split, so no test sample's donor features were seen
    # in training and vice versa.
    donor = [-1] * len(use)
    for idx in (tr.tolist(), te.tolist()):
        d = donor_index([books[i] for i in idx], [use[i]["chunk"] for i in idx])
        for p, i in enumerate(idx):
            donor[i] = idx[d[p]]
    perm = torch.tensor(donor)

    K = {}
    for k in ENCODINGS:
        X = torch.stack([s[k].flatten() for s in use])
        K[k] = standardized_kernel(X, tr, device)
        del X
    pd = perm.to(K["E"].device)
    configs = {"E": K["E"]}
    for z in Z_ENCODINGS:
        configs[z] = K[z]
        configs[f"E+{z}"] = K["E"] + K[z]
        configs[f"E+donor:{z}"] = K["E"] + K[z][pd][:, pd]
    if lam_rel is None:
        lam_rel = torch.logspace(-4, 2, 13, dtype=torch.float64)

    fits = {}
    for name, Kc in configs.items():
        fits[name] = ridge_eval(Kc, Ys, tr, te, groups[tr], lam_rel, n_folds)
        print(f"  fit {name}", flush=True)

    te_books = [books[i] for i in te.tolist()]
    out = {"n_train": int(tr.numel()), "n_test": int(te.numel()),
           "n_train_books": len(books_all) - len(test_books),
           "n_test_books": len(test_books), "vocab": int(vocab.numel()),
           "r2": {name: {t: f[t]["r2"] for t in TARGETS}
                  for name, f in fits.items()},
           "lambda_rel": {name: {t: f[t]["lambda_rel"] for t in TARGETS}
                          for name, f in fits.items()},
           "lambda_at_edge": sorted({f"{name}/{t}" for name, f in fits.items()
                                     for t in TARGETS
                                     if f[t]["lambda_at_edge"]}),
           "gains": {}}
    for z in Z_ENCODINGS:
        out["gains"][z] = {}
        for t in TARGETS:
            real, don, e = fits[f"E+{z}"][t], fits[f"E+donor:{z}"][t], fits["E"][t]
            c, clo, chi = book_boot_gain(real["sse"], don["sse"], real["sst"],
                                         te_books, n_boot, seed)
            g, glo, ghi = book_boot_gain(real["sse"], e["sse"], real["sst"],
                                         te_books, n_boot, seed)
            out["gains"][z][t] = {
                "content_gain": c, "content_ci": [clo, chi],
                "gain_over_E": g, "gain_over_E_ci": [glo, ghi],
                # BOTH bars.  Real-minus-donor alone is positive for a Z that
                # is a copy of E, because the donor block hurts the fit.
                "holds_content": bool(clo > 0 and glo > 0)}
    out["probe_health_ok"] = bool(out["r2"]["E"]["bow_self"] > PROBE_HEALTH_R2)
    out["enough_test_books"] = len(test_books) >= MIN_TEST_BOOKS
    # THE VETOES OVERRIDE THE FLAGS, not just the printout: results.json is
    # what gets read later, and a flag that says True under a failed veto is a
    # confident wrong number in the file even if the log warned.
    if not (out["probe_health_ok"] and out["enough_test_books"]):
        for z in out["gains"]:
            for t in out["gains"][z]:
                out["gains"][z][t]["holds_content"] = False
    return out


# ─── report ─────────────────────────────────────────────────────────────────

def print_report(rep: dict) -> None:
    d1 = rep.get("d1")
    if d1:
        print(f"\n{'=' * 78}")
        print(f"D1 -- seed vs document | {d1['n_samples']} samples, "
              f"{d1['n_books']} books | ratio = seed_var / doc_var")
        print("     ~1 = the encoding is s0 noise; ~0 = it is the document")
        print("=" * 78)
        print(f"  {'encoding':<10}{'||X||':>10}{'seed_var':>12}{'doc_var':>12}"
              f"{'ratio':>9}  {'95% CI':<20}reading")
        for k, e in d1["encodings"].items():
            print(f"  {k:<10}{e['row_norm']:>10.4f}{e['seed_var']:>12.4e}"
                  f"{e['doc_var']:>12.4e}{e['ratio']:>9.3f}  "
                  f"[{e['ci_lo']:.3f}, {e['ci_hi']:.3f}]      {e['reading']}")
        print("\n  per loop depth, summary columns (the tape Z_delta samples from):")
        print(f"  {'t':<4}{'||d_t||':>10}{'ratio':>9}  95% CI")
        for r in d1["depths"]:
            print(f"  {r['t']:<4}{r['delta_norm']:>10.4f}{r['ratio']:>9.3f}  "
                  f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
        print(f"\n  lowest-ratio Z encoding: {d1['lowest_ratio_z']}  "
              f"(cuts {CONTENT_CUT}/{NOISE_CUT} are chosen, not measured)")
    d2 = rep.get("d2")
    if d2:
        print(f"\n{'=' * 78}")
        print(f"D2 -- what Z holds beyond E | train {d2['n_train']} samples / "
              f"{d2['n_train_books']} books, test {d2['n_test']} / "
              f"{d2['n_test_books']} books, vocab {d2['vocab']}")
        print("=" * 78)
        print(f"  {'features':<18}" + "".join(f"{t:>12}" for t in TARGETS)
              + "   (held-out R^2)")
        for name, r in d2["r2"].items():
            print(f"  {name:<18}" + "".join(f"{r[t]:>12.4f}" for t in TARGETS))
        print("\n  content gain = R^2(E+Z) - R^2(E + another book's Z); "
              "over E = R^2(E+Z) - R^2(E)")
        print("  book-cluster 95% CIs.  HOLDS CONTENT needs BOTH above zero.")
        for z, g in d2["gains"].items():
            for t in TARGETS:
                x = g[t]
                sig = "  * HOLDS CONTENT BEYOND E" if x["holds_content"] else ""
                print(f"  {z:<9}{t:<10}{x['content_gain']:>+10.4f}  "
                      f"[{x['content_ci'][0]:+.4f}, {x['content_ci'][1]:+.4f}]"
                      f"   over E {x['gain_over_E']:+.4f} "
                      f"[{x['gain_over_E_ci'][0]:+.4f}, "
                      f"{x['gain_over_E_ci'][1]:+.4f}]{sig}")
        edge = d2["lambda_at_edge"]
        if edge:
            more = f" and {len(edge) - 6} more" if len(edge) > 6 else ""
            print(f"\n  note: lambda hit the grid edge for "
                  f"{', '.join(edge[:6])}{more}")
        if not d2["enough_test_books"]:
            print(f"\n  *** ONLY {d2['n_test_books']} TEST BOOKS (< "
                  f"{MIN_TEST_BOOKS}): the book-cluster CIs are not usable and")
            print("  *** no HOLDS CONTENT flag is set.")
        if not d2["probe_health_ok"]:
            print("\n  *** PROBE HEALTH FAILED: E cannot predict its own chunk's")
            print("  *** words (R^2 <= 0.02).  The probe is not measuring, and no")
            print("  *** D2 row above is interpretable.")
        else:
            print(f"\n  probe health: R^2(E -> bow_self) = "
                  f"{d2['r2']['E']['bow_self']:.4f} (the probe's positive control)")


# ─── main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="graft flags; REQUIRED on an overlay checkpoint (RED "
                        "12).  Mirror pace/eval_carry_2x2.sbatch's SETS.")
    p.add_argument("--n_chunks", type=int, default=8)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--max_examples", type=int, default=0,
                   help="rows to read; 0 = the whole pack")
    p.add_argument("--d1_rows", type=int, default=200,
                   help="rows that also get the seed-b re-run, spread evenly")
    p.add_argument("--tok_rows", type=int, default=64)
    p.add_argument("--tok_pool", type=int, default=4)
    p.add_argument("--bow_vocab", type=int, default=2048)
    p.add_argument("--bow_skip_top", type=int, default=100)
    p.add_argument("--test_every", type=int, default=5,
                   help="every Nth book is held out")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--bos_id", type=int, default=None,
                   help="the token that opens each book, used only if the "
                        "pack's ragged document ends do not mark the books "
                        "(the packer's --prepend_bos is off by default)")
    p.add_argument("--dtype", default="float32", choices=["float32"])
    p.add_argument("--out_dir", default="eval_results/z2_content")
    p.add_argument("--allow_scrambled", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from cortex_memory.health import refuse_if_scrambled
    from eval_carry_2x2 import has_latent_channel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = parse_config_overrides(args.set)
    model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                 torch.float32, device,
                                 config_overrides=overrides or None)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    refuse_if_scrambled(cortex, "z_content", args.allow_scrambled)
    if cortex is None or getattr(cortex, "prefix", None) is None:
        print("FAILED: this checkpoint has no prefix buffer (check --set).")
        return 2
    if not has_latent_channel(cortex):
        print("FAILED: no Z channel on this build -- D1 and D2 measure Z's "
              "write, so there is nothing to measure.  Check --set "
              "latent_carry=true.")
        return 3

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    n = len(ds) if args.max_examples == 0 else min(args.max_examples, len(ds))
    eos, eos_note = pack_eos([ds[i] for i in range(n)],
                             getattr(getattr(inner, "config", cfg),
                                     "eos_token_id", None))
    print(f"[z2] eos for book ends: {eos_note}", flush=True)
    kept, firsts, ends, ragged = [], [], [], 0
    for i in range(n):
        r = ds[i]
        firsts.append(int(r["input_ids"][0]))
        m = r.get("attention_mask")
        ends.append(is_book_end(r["input_ids"], m, eos))
        if m is not None and min(m) == 0:
            ragged += 1          # padded: skipped, the chain needs full rows
        else:
            kept.append(i)
    # Books are found on ALL rows read, ragged ones included: the ragged row is
    # the marker, and skipping it first would erase every boundary.
    books, book_note = book_ids(ends, firsts, args.bos_id)
    print(f"[z2] pack {args.data}: {n} rows read, {ragged} ragged skipped, "
          f"{len(kept)} used | books: {book_note}", flush=True)
    stride = max(1, len(kept) // max(1, args.d1_rows))
    d1_rows = set(kept[::stride][:args.d1_rows])
    rows = [(i, books[i], torch.tensor(ds[i]["input_ids"], dtype=torch.long))
            for i in kept]

    samples, check = collect(model, cortex, rows, args.n_chunks,
                             to_num_steps(args.T), device, args.seed, d1_rows,
                             args.tok_rows, args.tok_pool)
    bad = reseed_verdict(check)
    report = {
        "instrument": "Z attempt 2 step 0: D1 seed-vs-document, D2 probes",
        "when": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name, "checkpoint": args.checkpoint,
        "data": args.data, "sets": args.set,
        "config": {k: getattr(args, k) for k in (
            "n_chunks", "T", "max_examples", "d1_rows", "tok_rows", "tok_pool",
            "bow_vocab", "bow_skip_top", "test_every", "folds", "boot", "seed",
            "dtype")},
        "rows": {"read": n, "ragged_skipped": ragged, "used": len(kept),
                 "d1": len(d1_rows), "books": book_note},
        "reseed_check": check, "reseed_ok": bad is None,
    }
    if bad is not None:
        print(f"\n*** RESEED CHECK FAILED: {bad}.  D1 is not measuring s0 "
              "noise; nothing is reported.")
        _write(report, args.out_dir)
        return 4
    print("[z2] reseed check: same seed reproduces, a new seed moves every Z "
          "encoding -- OK", flush=True)
    report["d1"] = d1_report(samples, args.boot, args.seed)
    vocab_size = int(getattr(getattr(inner, "config", cfg), "vocab_size"))
    report["d2"] = d2_report(samples, vocab_size, device,
                             args.test_every, args.folds, args.bow_vocab,
                             args.bow_skip_top, args.boot, args.seed)
    print_report(report)
    _write(report, args.out_dir)
    return 0


def _write(report: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
