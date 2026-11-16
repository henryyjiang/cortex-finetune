"""
Fact probe -- is a BABILong fact WRITTEN into E / Z, does it SURVIVE to the
question, and is it READ?

diag_babilong_dissect.py measures all of this causally on the whole carry
(teleport / donor / fact_in_window).  It cannot say WHICH channel holds the
fact, or whether a null is a write failure or a read failure.  This tool asks
the representation directly, per channel, on qa1 (one supporting fact: the
queried person's last move).

Per item (fact parsing, chunking and the chain are the dissect's own code):

  WRITE      linear probes (6-way, the six rooms) on what the fact chunk WROTE:
             the 16 E rows and the 16 Z rows the buffer's merge was handed.
               text_bag   mean input embedding of the fact chunk's tokens --
                          the CEILING: what a linear readout of the chunk's
                          own words gets.  Same chunk, same other facts, so
                          the dilution confound is shared with the writes.
               e/z_write  mean of the chunk's 16 written rows
               e/z_slot   one probe per summary slot, best slot reported
                          (a fact held by one slot is not averaged away)
               e/z_prev   the chunk BEFORE the fact chunk: negative control
               shuffled_* the same probe on permuted labels: the null,
                          including the optimism of best-L2 / best-slot.
  SURVIVAL   provenance of every row in the final carry (the one the question
             chunk reads): each row is matched to the write it came from by
             cosine.  v2 keeps rows bit-identical (E is rescaled by the store
             cap, so cosine, not equality), so a valid match is cos > 0.999.
             A blending buffer (cortex-final's gated ring) fails that test and
             SURVIVAL is reported as not identifiable rather than guessed.
               n fact rows surviving, per channel, by fact distance
               final_e/z       probe on the mean of the whole final carry
               final_e/z_fact  probe on the surviving fact rows only
  READ       the final carry of a DONOR item -- same cell, same row count,
             whose story places this item's person in a DIFFERENT room --
             spliced in per channel, scored on this item's question window:
               shift = [NLL(gold) - NLL(donor's room)] under the swap
                       minus the same under the item's own carry
             > 0: the answer moved toward what the donor's memory says, i.e.
             the read USES that channel's content.  Splices: E only (own Z),
             Z only (own E), both.  Also own - nocarry on the gold margin.

HOW TO READ (top to bottom; stop at the first that fails)
  text_bag ~ chance            the chunk itself does not linearly hold the
                               room (dilution) -- the probe is uninformative
  e/z_write ~ e/z_prev, both << text_bag
                               WRITE failure for that channel
  write ok, final_*_fact ok but fact rows rarely survive past ~4 chunks
                               RETENTION / eviction
  rows survive and decode, shift ~ 0
                               READ failure: the fact is there, unused
  shift > 0                    the channel's content reaches the answer

USAGE
  python evals/diag_fact_probe.py --model_name ckpts/olmo-retrofit-cortex \
      --checkpoint cortex-pm/pm-v2capwarm-long/checkpoint_44558 $SETS \
      --dataset_path data/babilong-1k --lengths 2k 4k 8k 16k \
      --out_dir eval_results/pm/limbs/pm-v2capwarm-long/fact_probe
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from model_utils import (  # noqa: E402
    _unwrap, load_checkpoint, parse_config_overrides, to_num_steps,
)
from eval_babilong import collect_candidates, load_bucket  # noqa: E402
from diag_babilong_dissect import (  # noqa: E402
    Ambiguous, _Tee, boot_ci, chain_seed, facts_upto, parse_lengths,
    prepare_item, refuse, run_chain, score_condition, world_answer,
)

PROV_COS = 0.999          # a kept row matches its write to fp32 rounding


# ---------------------------------------------------------------------------
# Pure: probes
# ---------------------------------------------------------------------------

def _fit_lr(X: torch.Tensor, y: torch.Tensor, n_cls: int, l2: float,
            iters: int = 200) -> torch.Tensor:
    """Multinomial logistic regression (with bias) by full-batch LBFGS."""
    n, d = X.shape
    Xb = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)
    W = torch.zeros(d + 1, n_cls, dtype=X.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([W], lr=1.0, max_iter=iters,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xb @ W, y) \
            + l2 * W[:-1].pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach()


def _logp(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], dim=1)
    return torch.log_softmax(Xb @ W, dim=-1)


def cv_probe(X: torch.Tensor, y, n_cls: int, folds: int = 5,
             l2_grid=(1e-4, 1e-3, 1e-2, 1e-1), seed: int = 0) -> dict:
    """K-fold CV accuracy of a linear probe on X [n, d], best L2 on the grid.

    Features are standardized on each training fold.  Best-of-grid is
    optimistic; run the same call on permuted labels for the matching null.
    """
    y = torch.as_tensor(y, dtype=torch.long)
    n = int(y.numel())
    if n < folds * 2:
        return {"n": n, "acc": None, "note": "too few items"}
    X = X.detach().double()
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    fold_of = torch.empty(n, dtype=torch.long)
    fold_of[order] = torch.arange(n) % folds
    per_l2 = {}
    for l2 in l2_grid:
        correct, ll = 0, 0.0
        for k in range(folds):
            tr, te = fold_of != k, fold_of == k
            mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-8)
            W = _fit_lr((X[tr] - mu) / sd, y[tr], n_cls, l2)
            lp = _logp((X[te] - mu) / sd, W)
            correct += int((lp.argmax(-1) == y[te]).sum())
            ll += float(-lp.gather(1, y[te][:, None]).sum())
        per_l2[l2] = {"acc": correct / n, "logloss": ll / n}
    best = max(per_l2, key=lambda k: per_l2[k]["acc"])
    counts = torch.bincount(y, minlength=n_cls).double()
    return {"n": n, "acc": per_l2[best]["acc"],
            "logloss": per_l2[best]["logloss"], "l2": best,
            "majority": float(counts.max() / n), "per_l2": per_l2}


def slot_probe(X: torch.Tensor, y, n_cls: int, **kw) -> dict:
    """X [n, W, d]: one probe per summary slot (slot j is the same column on
    every item), best slot reported.  A fact held by one slot is not averaged
    away.  Best-of-W adds optimism: compare with the same call on permuted
    labels."""
    per = [cv_probe(X[:, j], y, n_cls, **kw) for j in range(X.shape[1])]
    accs = [p["acc"] for p in per]
    if any(a is None for a in accs):
        return {"n": per[0]["n"], "acc": None, "note": "too few items"}
    j = max(range(len(accs)), key=accs.__getitem__)
    return {**per[j], "best_slot": j, "per_slot_acc": accs}


# ---------------------------------------------------------------------------
# Pure: provenance, splices, donors
# ---------------------------------------------------------------------------

def provenance(final_rows: torch.Tensor, writes: list) -> dict:
    """Match each final carry row [R, D] to the write it came from.

    writes[c] is chunk c's [W, D] write for this channel.  Returns per-row
    chunk index, slot, best cosine, and whether every match is exact enough
    to call it provenance (a blended row matches nothing at PROV_COS).
    """
    W = writes[0].shape[0]
    bank = torch.nn.functional.normalize(torch.cat(writes, 0).float(), dim=-1)
    fr = torch.nn.functional.normalize(final_rows.float(), dim=-1)
    cos, idx = (fr @ bank.T).max(dim=1)
    return {"chunk": (idx // W).tolist(), "slot": (idx % W).tolist(),
            "cos": cos.tolist(), "valid": bool((cos > PROV_COS).all())}


def splice(own: torch.Tensor, donor: torch.Tensor, hidden: int,
           which: str) -> torch.Tensor:
    """own/donor [B, R, D*] carries; replace E ('e'), Z ('z') or both."""
    if own.shape != donor.shape:
        raise ValueError(f"donor carry {tuple(donor.shape)} != own "
                         f"{tuple(own.shape)}")
    if which == "both":
        return donor.clone()
    if own.shape[-1] != 2 * hidden:
        raise ValueError("per-channel splice needs a dual-channel carry")
    e = (donor if which == "e" else own)[..., :hidden]
    z = (donor if which == "z" else own)[..., hidden:]
    return torch.cat([e, z], dim=-1)


def donor_answer(task: str, donor: dict, entity: str):
    """Where the donor's priming chunks put `entity` -> (room, distance)."""
    facts = facts_upto(donor, donor["n_prime_chunks"] - 1)
    ans, support = world_answer(task, facts, entity)
    if ans is None or not support:
        return None, None
    last = facts[support[-1]]
    chunk = donor["fact_chunks"][donor["facts"].index(last)][0]
    return ans, donor["n_prime_chunks"] - chunk


def pick_swap_donor(item: dict, pool: list, rng: random.Random):
    """Same carry row count; the donor's story must put this item's person in
    a known room different from this item's gold."""
    ok = []
    for d in pool:
        if d["id"] == item["id"] or d["rows"] != item["rows"]:
            continue
        ans, dist = donor_answer(item["task"], d, item["entity"])
        if ans is None or ans.lower() == item["gold"].lower():
            continue
        ok.append((d, ans, dist))
    if not ok:
        return None
    ok.sort(key=lambda t: t[0]["id"])
    return rng.choice(ok)


def dist_band(d: int) -> str:
    return "1" if d <= 1 else "2-4" if d <= 4 else "5-8" if d <= 8 else "9+"


# ---------------------------------------------------------------------------
# Model side
# ---------------------------------------------------------------------------

class WriteRecorder:
    """Wraps buf.merge to keep each chunk's (E, Z) write, as handed to the
    buffer -- before keep/replace or the E store cap touch it."""

    def __init__(self, buf):
        self.buf, self.orig, self.writes = buf, buf.merge, []

        def merge(state, new_vecs, new_latent=None):
            self.writes.append((
                new_vecs[0].detach().float().cpu(),
                None if new_latent is None
                else new_latent[0].detach().float().cpu()))
            return self.orig(state, new_vecs, new_latent)
        buf.merge = merge

    def reset(self):
        self.writes = []


def margins(ranked, cands, gold: str, other: Optional[str] = None) -> dict:
    nll = {r["text"].lower(): r["nll_sum"] for r in ranked}
    g = gold.lower()
    out = {"gold_margin": nll[g] - min(v for k, v in nll.items() if k != g),
           "top1": min(nll, key=nll.get) == g}
    if other is not None:
        out["shift"] = nll[g] - nll[other.lower()]
        out["picks_other"] = min(nll, key=nll.get) == other.lower()
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser("BABILong qa1 fact probe: write / survival / read")
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dataset_path", default=None)
    p.add_argument("--dataset_repo", default="RMT-team/babilong-1k-samples")
    p.add_argument("--lengths", nargs="+", default=["2k", "4k", "8k", "16k"])
    p.add_argument("--max_items", type=int, default=200,
                   help="items with the fact OUTSIDE the question window, per "
                        "length (ambiguous / in-window rows do not count)")
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--num_chunks", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--allow_key_mismatch", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    task = "qa1"
    t0 = time.time()
    lengths = parse_lengths(args.lengths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps = to_num_steps(args.T)
    overrides = parse_config_overrides(args.set)
    cands_by_len = {}
    for L in lengths:
        c = collect_candidates(load_bucket(task, L, args.dataset_path,
                                           args.dataset_repo))
        if not c:
            refuse(f"{task}/{L}: empty candidate set")
        cands_by_len[L] = c
    rooms = sorted({c.lower() for cs in cands_by_len.values() for c in cs})
    cls = {r: i for i, r in enumerate(rooms)}

    cap = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, cap)):
        model, cfg = load_checkpoint(args.checkpoint, args.model_name, None,
                                     getattr(torch, args.dtype), device,
                                     config_overrides=overrides or None)
    inner = _unwrap(model)
    cortex = getattr(inner, "cortex", None)
    buf = getattr(cortex, "prefix", None) if cortex is not None else None
    if buf is None:
        refuse("no prefix buffer: this probe needs a memory arm")
    if not args.checkpoint:
        refuse("no --checkpoint overlay: the trained cortex would be wiped")
    m = re.search(r"\[load\] overlaid .*?: (\d+) missing / (\d+) unexpected",
                  cap.getvalue())
    if not m:
        refuse("could not confirm the overlay ('[load] overlaid' line missing)")
    if (int(m.group(1)) or int(m.group(2))) and not args.allow_key_mismatch:
        refuse(f"overlay: {m.group(1)} missing / {m.group(2)} unexpected keys "
               f"-- the graft was built without this arm's --set geometry")
    hidden = int(buf.hidden_size)
    dual = bool(getattr(buf, "carries_latent", False))
    wte = inner.get_input_embeddings().weight
    print(f"[arm] {type(buf).__name__} n_vec={buf.n_vec} "
          f"n_slots={getattr(buf, 'n_slots', '?')} dual={dual} hidden={hidden}",
          flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name)
    rec = WriteRecorder(buf)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items, skipped = [], {}
    feats = {k: [] for k in ("text_bag", "e_write", "z_write", "e_rows",
                             "z_rows", "final_e", "final_z")}
    prev = {"e_prev": [], "z_prev": [], "y": []}
    fact_rows = {"e": [], "z": [], "y_e": [], "y_z": []}

    # ---- pass 1: chains, writes, provenance, own / nocarry scores ----------
    for L in lengths:
        n_ok = 0
        for row_idx, ex in enumerate(load_bucket(task, L, args.dataset_path,
                                                 args.dataset_repo)):
            if n_ok >= args.max_items:
                break
            try:
                it = prepare_item(tok, task, L, row_idx, ex, args.seq_len,
                                  args.max_new_tokens, args.num_chunks, rooms)
            except Ambiguous as e:
                skipped[e.reason] = skipped.get(e.reason, 0) + 1
                continue
            if it["src_chunk"] is None:
                skipped["fact_in_window"] = skipped.get("fact_in_window", 0) + 1
                continue
            torch.manual_seed(chain_seed(args.seed, it["id"]))
            rec.reset()
            states = run_chain(model, it["prime_chunks"], num_steps, device,
                               buf=buf)
            if len(rec.writes) != it["n_prime_chunks"]:
                raise RuntimeError(f"{len(rec.writes)} merges recorded for "
                                   f"{it['n_prime_chunks']} priming chunks")
            final = states[-1]
            src, y = it["src_chunk"], cls[it["gold"].lower()]
            e_w = [w[0] for w in rec.writes]
            z_w = [w[1] for w in rec.writes] if dual else None
            chunk_ids = it["prime_chunks"][src][0]
            feats["text_bag"].append(wte[chunk_ids.to(wte.device)].detach()
                                     .float().mean(0).cpu())
            feats["e_write"].append(e_w[src].mean(0))
            feats["e_rows"].append(e_w[src])
            fin = final[0].detach().float().cpu()
            feats["final_e"].append(fin[:, :hidden].mean(0))
            if dual:
                feats["z_write"].append(z_w[src].mean(0))
                feats["z_rows"].append(z_w[src])
                feats["final_z"].append(fin[:, hidden:].mean(0))
            if src >= 1:
                prev["e_prev"].append(e_w[src - 1].mean(0))
                if dual:
                    prev["z_prev"].append(z_w[src - 1].mean(0))
                prev["y"].append(y)

            prov = {"e": provenance(fin[:, :hidden], e_w)}
            if dual:
                prov["z"] = provenance(fin[:, hidden:], z_w)
            surv = {}
            for ch, pv in prov.items():
                keep = [r for r, c in enumerate(pv["chunk"]) if c == src]
                surv[ch] = {"valid": pv["valid"], "n_fact_rows": len(keep),
                            "min_cos": min(pv["cos"])}
                if pv["valid"] and keep:
                    sl = slice(0, hidden) if ch == "e" else slice(hidden, None)
                    fact_rows[ch].append(fin[keep, sl].mean(0))
                    fact_rows[f"y_{ch}"].append(y)

            own = score_condition(model, tok, it["final_ids"], cands_by_len[L],
                                  num_steps, final, it["id"])
            none = score_condition(model, tok, it["final_ids"], cands_by_len[L],
                                   num_steps, None, it["id"])
            items.append({
                "id": it["id"], "task": task, "length": L, "gold": it["gold"],
                "entity": it["entity"], "y": y, "src": src,
                "distance": it["distance"], "n_prime_chunks": it["n_prime_chunks"],
                "facts": it["facts"], "fact_chunks": it["fact_chunks"],
                "rows": int(final.shape[1]), "survival": surv,
                "own": margins(own, cands_by_len[L], it["gold"]),
                "nocarry": margins(none, cands_by_len[L], it["gold"]),
                "_final": final.detach().cpu(), "_final_ids": it["final_ids"]})
            n_ok += 1
        print(f"[{L}] {n_ok} items  ({time.time() - t0:.0f}s)", flush=True)

    if len(items) < 2 * args.folds:
        refuse(f"only {len(items)} usable items (skipped: {skipped})")

    # ---- pass 2: donor swaps ----------------------------------------------
    rng = random.Random(args.seed)
    for it in items:
        pool = [d for d in items if d["length"] == it["length"]]
        pick = pick_swap_donor(it, pool, rng)
        if pick is None:
            it["swap"] = None
            continue
        d, ans, ddist = pick
        cands = cands_by_len[it["length"]]
        own = score_condition(model, tok, it["_final_ids"], cands, num_steps,
                              it["_final"].to(device), it["id"])
        base = margins(own, cands, it["gold"], ans)
        sw = {"donor": d["id"], "donor_room": ans, "donor_dist": ddist,
              "picks_donor_own": base["picks_other"]}
        for which in (("e", "z", "both") if dual else ("both",)):
            m_sw = splice(it["_final"], d["_final"], hidden, which).to(device)
            r = margins(score_condition(model, tok, it["_final_ids"], cands,
                                        num_steps, m_sw, it["id"]),
                        cands, it["gold"], ans)
            sw[which] = {"shift": r["shift"] - base["shift"],
                         "picks_donor": r["picks_other"]}
        it["swap"] = sw

    # ---- probes ------------------------------------------------------------
    y_all = [it["y"] for it in items]
    n_cls = len(rooms)
    P = lambda X, y: cv_probe(X, y, n_cls, args.folds, seed=args.seed)  # noqa: E731
    S = lambda X, y: slot_probe(X, y, n_cls, folds=args.folds, seed=args.seed)  # noqa: E731
    perm = torch.randperm(len(y_all), generator=torch.Generator()
                          .manual_seed(args.seed + 1))
    y_shuf = [y_all[i] for i in perm]
    probes = {"text_bag": P(torch.stack(feats["text_bag"]), y_all),
              "e_write": P(torch.stack(feats["e_write"]), y_all),
              "shuffled_e_write": P(torch.stack(feats["e_write"]), y_shuf),
              "e_slot": S(torch.stack(feats["e_rows"]), y_all),
              "shuffled_e_slot": S(torch.stack(feats["e_rows"]), y_shuf),
              "final_e": P(torch.stack(feats["final_e"]), y_all)}
    if prev["y"]:
        probes["e_prev"] = P(torch.stack(prev["e_prev"]), prev["y"])
    # a blending ring matches only on short, unwrapped chains: that subset is
    # biased, so the surviving-rows probe needs provenance on EVERY item
    prov_ok = {ch: all(it["survival"][ch]["valid"] for it in items)
               for ch in (("e", "z") if dual else ("e",))}
    if fact_rows["y_e"] and prov_ok["e"]:
        probes["final_e_fact"] = P(torch.stack(fact_rows["e"]), fact_rows["y_e"])
    if dual:
        probes["z_write"] = P(torch.stack(feats["z_write"]), y_all)
        probes["z_slot"] = S(torch.stack(feats["z_rows"]), y_all)
        probes["shuffled_z_slot"] = S(torch.stack(feats["z_rows"]), y_shuf)
        probes["final_z"] = P(torch.stack(feats["final_z"]), y_all)
        if prev["y"]:
            probes["z_prev"] = P(torch.stack(prev["z_prev"]), prev["y"])
        if fact_rows["y_z"] and prov_ok["z"]:
            probes["final_z_fact"] = P(torch.stack(fact_rows["z"]),
                                       fact_rows["y_z"])

    # ---- survival and read aggregates --------------------------------------
    survival = {}
    for ch in (("e", "z") if dual else ("e",)):
        valid = all(it["survival"][ch]["valid"] for it in items)
        by = {}
        for it in items:
            b = by.setdefault(dist_band(it["distance"]), [])
            b.append(it["survival"][ch]["n_fact_rows"])
        survival[ch] = {"provenance_valid": valid,
                        "min_cos": min(it["survival"][ch]["min_cos"] for it in items),
                        "by_distance": {k: {"n": len(v),
                                            "mean_rows": sum(v) / len(v),
                                            "any": sum(x > 0 for x in v) / len(v)}
                                        for k, v in sorted(by.items())}}
    read = {"own_minus_nocarry_margin": boot_ci(
        [it["own"]["gold_margin"] - it["nocarry"]["gold_margin"] for it in items],
        args.boot, args.seed)}
    swapped = [it for it in items if it["swap"]]
    for which in (("e", "z", "both") if dual else ("both",)):
        read[which] = {"all": boot_ci([it["swap"][which]["shift"] for it in swapped],
                                      args.boot, args.seed)}
        for band in ("1", "2-4", "5-8", "9+"):
            v = [it["swap"][which]["shift"] for it in swapped
                 if dist_band(it["swap"]["donor_dist"]) == band]
            if v:
                read[which][f"donor_dist {band}"] = boot_ci(v, args.boot, args.seed)
        read[which]["picks_donor"] = sum(it["swap"][which]["picks_donor"]
                                         for it in swapped) / max(len(swapped), 1)
    read["picks_donor_own"] = sum(it["swap"]["picks_donor_own"]
                                  for it in swapped) / max(len(swapped), 1)
    read["n_swapped"], read["n_no_donor"] = len(swapped), len(items) - len(swapped)

    res = {"checkpoint": args.checkpoint, "task": task, "lengths": lengths,
           "rooms": rooms, "chance": 1 / n_cls, "n_items": len(items),
           "skipped": skipped, "probes": probes, "survival": survival,
           "read": read, "seconds": time.time() - t0}
    with open(out_dir / "results.json", "w") as f:
        json.dump(res, f, indent=2)
    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({k: v for k, v in it.items()
                                if not k.startswith("_") and k not in
                                ("facts", "fact_chunks")}) + "\n")

    print(f"\n== fact probe: {len(items)} qa1 items, chance {1 / n_cls:.3f} ==")
    for k, v in probes.items():
        if v.get("acc") is not None:
            print(f"  {k:18s} acc {v['acc']:.3f}  (n={v['n']}, majority "
                  f"{v['majority']:.3f}, l2={v['l2']})")
    for ch, s in survival.items():
        print(f"  survival {ch}: provenance valid={s['provenance_valid']} "
              f"(min cos {s['min_cos']:.4f})")
        for band, b in s["by_distance"].items():
            print(f"     dist {band:>4}: {b['mean_rows']:.1f}/{buf.n_vec} fact rows "
                  f"left, any {b['any']:.2f}  (n={b['n']})")
    ci = lambda t: "n/a" if t[0] is None else (  # noqa: E731
        f"{t[0]:+.3f}" + ("" if t[1] is None else f" [{t[1]:+.3f}, {t[2]:+.3f}]"))
    print(f"  own - nocarry gold margin {ci(read['own_minus_nocarry_margin'])} "
          f"(negative = the carry helps)")
    for which in (("e", "z", "both") if dual else ("both",)):
        print(f"  swap {which:4s} shift {ci(read[which]['all'])}  picks donor "
              f"{read[which]['picks_donor']:.3f} (own carry "
              f"{read['picks_donor_own']:.3f})")
    print(f"  skipped: {skipped}")
    print(f"-> {out_dir}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
