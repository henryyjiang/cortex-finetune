import json, sys, collections

rows = [json.loads(l) for l in open(sys.argv[1])]
norm = lambda s: " ".join(str(s).strip().lower().split())

cells = collections.Counter(f"{r['task']}/{r['bucket']}" for r in rows)
short = {k: v for k, v in cells.items() if v < max(cells.values())}
if short:
    print(f"incomplete cells: {short}\n")

def argmin(vals):
    return min(range(len(vals)), key=lambda i: vals[i])

for key in ("task", None):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"] if key else f"{r['task']}/{r['bucket']}"].append(r)
    print(f"{'cell':<12} {'n':>5} {'|C|':>4} {'chance':>7} {'contain':>8} "
          f"{'rank':>7} {'rank_ln':>8} {'pmi':>7}")
    print("-" * 62)
    for name in sorted(by):
        rs = by[name]
        n = len(rs)
        contain = sum(1 for r in rs if r["correct"]) / n
        ch = rk = rkln = pmi = m = 0
        csize = 0
        for r in rs:
            c = r.get("candidates")
            if not c:
                continue
            m += 1
            csize += len(c)
            g, nll, ntok = norm(r["gold"]), r["cand_nll"], r["cand_ntok"]
            ch += 1.0 / len(c)
            rk += norm(c[argmin(nll)]) == g
            rkln += norm(c[argmin([nll[i] / max(ntok[i], 1) for i in range(len(c))])]) == g
            nc = r.get("cand_nll_nocontext")
            if nc:
                pmi += norm(c[argmin([nll[i] - nc[i] for i in range(len(c))])]) == g
        if m:
            print(f"{name:<12} {n:>5} {csize/m:>4.1f} {ch/m*100:>6.1f}% "
                  f"{contain*100:>7.1f}% {rk/m*100:>6.1f}% {rkln/m*100:>7.1f}% {pmi/m*100:>6.1f}%")
    print()
