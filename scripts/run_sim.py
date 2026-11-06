"""Steps 2-4a: offline eviction simulation on the recorded traces.

Per prompt -> data/sim/<id>.json : {budget_frac: {"B":..., "full":[112], "ceiling":[112], policy:[112 masses]}}
plus data/sim/<id>_<frac>_oracle_w.pt (keep/evicted logs for live replay).
Heads are (layer, kv_head) flattened as layer*4 + kv_head.
"""
import argparse
import json
import zlib
import time
from pathlib import Path

import numpy as np
import torch

from kvo import policies as P
from kvo import sim

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [0.05, 0.10, 0.25, 0.50]
BELADY_TAUS = [0.001, 0.003, 0.01, 0.03]  # sensitivity of the Belady oracle only; main tau = P.TAU
GROUP = 28  # heads simulated together (memory)

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
torch.set_num_threads(8)

ids = sorted(p.name for p in (ROOT / "data/traces").iterdir() if (p / "meta.json").exists())
if args.only:
    ids = [i for i in ids if i in args.only.split(",")]
(ROOT / "data/sim").mkdir(exist_ok=True)

for pid in ids:
    out = ROOT / "data/sim" / f"{pid}.json"
    if out.exists() and not args.force:
        print("skip", pid); continue
    t0 = time.time()
    d = ROOT / "data/traces" / pid
    meta = json.load(open(d / "meta.json"))
    L = meta["L"]
    tr = np.load(d / "trace.npy", mmap_mode="r")
    T, nl, nkv, N = tr.shape
    H = nl * nkv
    st = np.load(d / "stats.npz")
    stats_all = {k: torch.from_numpy(st[k].reshape(H, L)) for k in st.files}
    res = {str(f): dict(B=max(1, round(f * L))) for f in BUDGETS}
    for f in BUDGETS:
        for k in ["full", "ceiling", "random", "lru", "streaming", "h2o", "snapkv", "oracle_w"] + [f"belady@{x}" for x in BELADY_TAUS]:
            res[str(f)][k] = [0.0] * H
        res[str(f)]["jaccard_belady_w"] = [0.0] * H
    logs = {str(f): dict(keep=torch.zeros(H, res[str(f)]["B"], dtype=torch.long), evicted=torch.zeros(T, H, dtype=torch.long)) for f in BUDGETS}
    for g0 in range(0, H, GROUP):
        hs = slice(g0, g0 + GROUP)
        A = torch.from_numpy(np.ascontiguousarray(tr.reshape(T, H, N)[:, hs])).float()
        stats = {k: v[hs] for k, v in stats_all.items()}
        futs = {x: P.Future(A, tau=x) for x in BELADY_TAUS}
        fut = futs[P.TAU]
        for f in BUDGETS:
            r, B = res[str(f)], res[str(f)]["B"]
            ref = sim.reference(A, L, B)
            for k in ["full", "ceiling"]:
                r[k][hs] = ref[k].tolist()
            for c in P.ONLINE:
                r[c.name][hs] = sim.simulate(A, stats, L, B, c, seed=zlib.crc32(f"{pid}|{c.name}".encode()))["mass"].tolist()
            w = sim.simulate(A, stats, L, B, P.WeightedGreedy, future=fut)
            r["oracle_w"][hs] = w["mass"].tolist()
            logs[str(f)]["keep"][hs], logs[str(f)]["evicted"][:, hs] = w["keep"], w["evicted"]
            for x in BELADY_TAUS:
                bel = sim.simulate(A, stats, L, B, P.Belady, future=futs[x], seed=1)
                r[f"belady@{x}"][hs] = bel["mass"].tolist()
                if x == P.TAU:  # overlap of the two oracles' prefill keep sets
                    kb = torch.zeros(len(range(H)[hs]), L, dtype=torch.bool).scatter_(1, bel["keep"], True)
                    kw = torch.zeros_like(kb).scatter_(1, w["keep"], True)
                    r["jaccard_belady_w"][hs] = ((kb & kw).sum(1) / (kb | kw).sum(1)).tolist()
        del A, futs, fut
    for f in BUDGETS:
        torch.save(logs[str(f)], ROOT / "data/sim" / f"{pid}_{f}_oracle_w.pt")
    res["meta"] = dict(L=L, T=T, task=meta["task"])
    json.dump(res, open(out, "w"))
    r = res["0.1"]
    s = lambda k: sum(r[k]) / sum(r["oracle_w"])
    print(f"{pid} T={T} {time.time()-t0:.0f}s @10%: " + " ".join(f"{k}={s(k):.3f}" for k in ["random", "lru", "streaming", "h2o", "snapkv", f"belady@{P.TAU}", "ceiling"]), flush=True)
