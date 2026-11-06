"""Step 2 check: how far is the weighted greedy oracle from the exact optimum?
On the first Tw<=32 decode steps of a sample of (prompt, head), compare greedy (future limited to the window)
with the LP relaxation of the exact problem (upper bound on the optimum). CPU only.
Output: data/sim/exact_check.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from kvo import policies as P
from kvo import sim

ROOT = Path(__file__).resolve().parents[1]
HEADS = [0 * 4 + 0, 3 * 4 + 1, 9 * 4 + 2, 14 * 4 + 3, 20 * 4 + 0, 27 * 4 + 1]   # spread over layers
FRACS = [0.05, 0.25]
rows = []
tasks = [json.loads(l) for l in open(ROOT / "data/tasks.jsonl")]
pick = [t["id"] for task in ["needle", "summ", "mdqa"] for t in [x for x in tasks if x["task"] == task][:2]]
for pid in pick:
    d = ROOT / "data/traces" / pid
    meta = json.load(open(d / "meta.json"))
    L = meta["L"]
    tr = np.load(d / "trace.npy", mmap_mode="r")
    T, nl, nkv, N = tr.shape
    Tw = min(T, 32)
    A = torch.from_numpy(np.ascontiguousarray(tr.reshape(T, nl * nkv, N)[:Tw, HEADS, : L + Tw])).float()
    fut = P.Future(A)
    for f in FRACS:
        B = round(f * L)
        g = sim.simulate(A, {}, L, B, P.WeightedGreedy, future=fut)["mass"]
        for i, h in enumerate(HEADS):
            t0 = time.time()
            lp, frac = sim.exact_window_lp(A[:, i].double().numpy(), L, B)
            rows.append(dict(id=pid, head=h, frac=f, B=B, Tw=Tw, greedy=g[i].item(), lp=lp, lp_frac=frac))
            print(f"{pid} h{h} {f} greedy={g[i].item():.4f} lp={lp:.4f} ratio={g[i].item()/lp:.4f} frac={frac:.2g} {time.time()-t0:.0f}s", flush=True)
json.dump(rows, open(ROOT / "data/sim/exact_check.json", "w"))
r = np.array([x["greedy"] / x["lp"] for x in rows])
print(f"greedy/LP: mean {r.mean():.4f} min {r.min():.4f}")
