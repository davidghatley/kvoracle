"""Step 4b: live generation under each policy and budget; prefill is shared across runs of a prompt.

Policies: the 5 online heuristics + oracle_replay (weighted-greedy oracle decisions from the offline sim,
replayed live; exact while the live output matches the full-cache output, approximate after divergence).
Output: data/live/<id>.json  {"full": {...}, "<frac>|<policy>": {"text", "score", "agree"}}  (resumable)
"""
import argparse
import json
import zlib
import time
from pathlib import Path

import torch

from kvo import metrics
from kvo import policies as P
from kvo.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [0.05, 0.10, 0.25, 0.50]

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--task", default="")
ap.add_argument("--policies", default="random,lru,streaming,h2o,snapkv,oracle_replay")
args = ap.parse_args()

tasks = [json.loads(l) for l in open(ROOT / "data/tasks.jsonl")]
if args.only:
    tasks = [t for t in tasks if t["id"] in args.only.split(",")]
if args.task:
    tasks = [t for t in tasks if t["task"] == args.task]
CLS = {c.name: c for c in P.ONLINE}
(ROOT / "data/live").mkdir(exist_ok=True)

eng = Engine()
for t in tasks:
    pid = t["id"]
    out = ROOT / "data/live" / f"{pid}.json"
    res = json.load(open(out)) if out.exists() else {}
    meta = json.load(open(ROOT / "data/traces" / pid / "meta.json"))
    full_text = meta["text"]
    res["full"] = dict(text=full_text, score=metrics.score(t["task"], full_text, t["answers"]), agree=1.0)
    todo = [(f, p) for f in BUDGETS for p in args.policies.split(",") if f"{f}|{p}" not in res]
    if not todo:
        print("skip", pid); continue
    L = eng.prefill(t["prompt"])
    assert L == meta["L"]
    for f, p in todo:
        B = max(1, round(f * L))
        t0 = time.time()
        if p == "oracle_replay":
            log = torch.load(ROOT / "data/sim" / f"{pid}_{f}_oracle_w.pt")
            nkv = eng.nkv
            fac = lambda li, st: P.make(P.Replay, nkv, B, L, st, device="cuda",
                                        keep=log["keep"][li * nkv:(li + 1) * nkv], evicted=log["evicted"][:, li * nkv:(li + 1) * nkv])
        else:
            fac = lambda li, st: P.make(CLS[p], eng.nkv, B, L, st, device="cuda", seed=zlib.crc32(f"{pid}|{p}|{li}".encode()))
        toks = eng.decode_live(fac, B, t["max_new"])
        txt = eng.text(toks)
        res[f"{f}|{p}"] = dict(text=txt, score=metrics.score(t["task"], txt, t["answers"]),
                               agree=metrics.rouge_l(txt, full_text), s=time.time() - t0)
        json.dump(res, open(out, "w"))
        print(f"{pid} {f} {p:14s} score={res[f'{f}|{p}']['score']:.3f} agree={res[f'{f}|{p}']['agree']:.2f} "
              f"{time.time()-t0:.0f}s {txt[:60]!r}", flush=True)
