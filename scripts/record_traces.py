"""Step 1: full-cache greedy generation with per-step attention traces.

Per prompt -> data/traces/<id>/
  trace.npy  fp16 [T, 28 layers, 4 kv heads, L+T]  group-mean attention of each decode step to every past key
  topv.npy / topi.npy  fp16 / int32 [T, 28, 28 query heads, 32]  per-query-head top-32 (values, key positions)
  stats.npz  prefill statistics [28, 4, L]: colsum, window, lastuse
  meta.json  L, T, tokens, text, timings
Usage: record_traces.py [--tasks data/tasks.jsonl] [--only id1,id2] [--task needle]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from kvo.engine import Engine

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default=str(ROOT / "data/tasks.jsonl"))
ap.add_argument("--only", default="")
ap.add_argument("--task", default="")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

tasks = [json.loads(l) for l in open(args.tasks)]
if args.only:
    tasks = [t for t in tasks if t["id"] in args.only.split(",")]
if args.task:
    tasks = [t for t in tasks if t["task"] == args.task]

eng = Engine()
for t in tasks:
    out = ROOT / "data/traces" / t["id"]
    if (out / "meta.json").exists() and not args.force:
        print("skip", t["id"]); continue
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize(); t0 = time.time()
    L = eng.prefill(t["prompt"])
    torch.cuda.synchronize(); t1 = time.time()
    toks, tr, tv, ti = eng.decode_trace(t["max_new"])
    torch.cuda.synchronize(); t2 = time.time()
    np.save(out / "trace.npy", tr)
    np.save(out / "topv.npy", tv)
    np.save(out / "topi.npy", ti)
    st = {k: torch.stack([s[k] for s in eng.stats]).cpu().numpy() for k in ["colsum", "window", "lastuse"]}
    np.savez(out / "stats.npz", **st)
    meta = dict(id=t["id"], task=t["task"], L=L, T=tr.shape[0], tokens=toks, text=eng.text(toks),
                prefill_s=t1 - t0, decode_s=t2 - t1, peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                trace_mb=tr.nbytes / 2**20, topk_mb=(tv.nbytes + ti.nbytes) / 2**20)
    json.dump(meta, open(out / "meta.json", "w"))
    print(f"{t['id']} L={L} T={tr.shape[0]} prefill {t1-t0:.0f}s decode {t2-t1:.0f}s "
          f"trace {meta['trace_mb']:.0f}MB peak {meta['peak_mem_gb']:.1f}GB | {meta['text'][:100]!r}", flush=True)
    del tr, tv, ti
