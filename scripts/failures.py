"""Step 5: which tokens did the oracle keep that a failing heuristic evicted? (needle task)

For each needle prompt and budget, re-simulate the oracle and each heuristic, take the cache contents at each
decode step, and attribute the attention mass of "oracle kept, heuristic missing" tokens to token types:
  sink (pos < 4) | needle (the inserted sentence) | question (prompt tail after the haystack) | template | haystack
Also: needle-token mass retained per policy, per layer; and which query heads look at the needle (top-32 traces).
Output: data/sim/failures.json and printed summary.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from kvo import policies as P
from kvo import sim

ROOT = Path(__file__).resolve().parents[1]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tasks = [json.loads(l) for l in open(ROOT / "data/tasks.jsonl") if json.loads(l)["task"] == "needle"]
POLS = [P.LRU, P.StreamingLLM, P.H2O, P.SnapKV]
TYPES = ["sink", "needle", "question", "template", "haystack"]


def token_types(t):
    enc = tok(t["prompt"], add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    L = len(offs)
    ty = np.array(["haystack"] * L, dtype=object)
    n0 = t["prompt"].index(t["needle"]); n1 = n0 + len(t["needle"])
    q0 = t["prompt"].rindex("\n\nWhat is the special magic number")
    h0 = t["prompt"].index("Below is a long document.")
    for i, (a, b) in enumerate(offs):
        if b <= h0:
            ty[i] = "template"
        elif a >= q0:
            ty[i] = "question"
        elif a < n1 and b > n0:
            ty[i] = "needle"
    ty[:4] = "sink"
    return ty


def held_masks(A, stats, L, B, cls, **kw):
    """Run a policy and return list over steps of [H, N] bool masks of past tokens held at that step."""
    T, H, N = A.shape
    pol, keep = P.make(cls, H, B, L, stats, seed=1, **kw)
    S = B + 1
    rows = torch.arange(H)
    pos = torch.full((H, S), -1, dtype=torch.long)
    pos[:, :B] = keep
    pol.init_state(pos)
    free = torch.full((H,), B, dtype=torch.long)
    out = []
    for t in range(T):
        cur = L + t
        pos[rows, free] = cur
        pol.on_insert(free, cur)
        m = torch.zeros(H, N, dtype=torch.bool)
        m.scatter_(1, pos, True)
        m[:, cur] = False
        out.append(m)
        a = A[t].gather(1, pos)
        pol.observe(t, a / a.sum(-1, keepdim=True).clamp_min(1e-12), pos, cur)
        free = pol.evict(t, pos, cur)
    return out


res = {}
for f in [0.05, 0.10]:
    agg = {c.name: dict(missing_by_type={k: 0.0 for k in TYPES}, missing_total=0.0, needle_mass=0.0,
                        needle_layer=np.zeros(28), needle_heads_lost=0) for c in POLS}
    agg["oracle_w"] = dict(needle_mass=0.0, needle_layer=np.zeros(28))
    full_needle = 0.0
    full_needle_layer = np.zeros(28)
    n_needle_heads = 0
    for t in tasks:
        d = ROOT / "data/traces" / t["id"]
        meta = json.load(open(d / "meta.json"))
        L = meta["L"]
        tr = np.load(d / "trace.npy")
        T, nl, nkv, N = tr.shape
        A = torch.from_numpy(tr.reshape(T, nl * nkv, N)).float()
        st = np.load(d / "stats.npz")
        stats = {k: torch.from_numpy(st[k].reshape(nl * nkv, L)) for k in st.files}
        ty = token_types(t)
        assert len(ty) == L
        is_needle = torch.zeros(N, dtype=torch.bool); is_needle[:L] = torch.from_numpy(ty == "needle")
        B = round(f * L)
        orc = held_masks(A, stats, L, B, P.WeightedGreedy, future=P.Future(A))
        nm_full = torch.stack([(A[s] * is_needle).sum(-1) for s in range(T)]).sum(0)  # [H]
        full_needle += nm_full.sum().item()
        full_needle_layer += nm_full.view(nl, nkv).sum(1).numpy()
        hot = nm_full > 0.05 * T  # heads paying > 5% of their attention to the needle on average
        n_needle_heads += int(hot.sum())
        agg["oracle_w"]["needle_mass"] += sum((A[s] * orc[s] * is_needle).sum().item() for s in range(T))
        agg["oracle_w"]["needle_layer"] += sum((A[s] * orc[s] * is_needle).sum(-1).view(nl, nkv).sum(1).numpy() for s in range(T))
        tyN = np.concatenate([ty, np.array(["generated"] * (N - L), dtype=object)])
        for c in POLS:
            hm = held_masks(A, stats, L, B, c)
            g = agg[c.name]
            for s in range(T):
                miss = A[s] * (orc[s] & ~hm[s])
                g["missing_total"] += miss.sum().item()
                ms = miss.sum(0).numpy()
                for k in TYPES:
                    g["missing_by_type"][k] += float(ms[tyN == k].sum())
                nmass = (A[s] * hm[s] * is_needle)
                g["needle_mass"] += nmass.sum().item()
                g["needle_layer"] += nmass.sum(-1).view(nl, nkv).sum(1).numpy()
            kept0 = (hm[0] & is_needle).any(-1)
            g["needle_heads_lost"] += int((hot & ~kept0).sum())
        del A
    out = {"n_needle_heads": n_needle_heads, "full_needle_mass": full_needle,
           "needle_share_of_all_mass": None, "full_needle_layer": full_needle_layer.tolist()}
    for k, g in agg.items():
        gg = {kk: (v.tolist() if isinstance(v, np.ndarray) else v) for kk, v in g.items()}
        out[k] = gg
    res[str(f)] = out
    print(f"\n== budget {f:.0%}: heads paying >5% to the needle: {n_needle_heads} (over 10 prompts x 112 heads)")
    print(f"needle mass retained / full-cache needle mass: oracle_w {agg['oracle_w']['needle_mass']/full_needle:.3f}")
    for c in POLS:
        g = agg[c.name]
        mt = g["missing_total"]
        print(f"  {c.name:10s} needle-mass kept {g['needle_mass']/full_needle:.3f} | needle-heads that lost every needle token at step 0: "
              f"{g['needle_heads_lost']}/{n_needle_heads} | oracle-kept-but-evicted mass by type: " +
              " ".join(f"{k}={g['missing_by_type'][k]/mt:.2f}" for k in TYPES))

# which query heads look at the needle (top-32 per query head, step 0), aggregated over prompts
qh = np.zeros((28, 28))
for t in tasks:
    d = ROOT / "data/traces" / t["id"]
    ty = token_types(t)
    needle_pos = set(np.where(ty == "needle")[0].tolist())
    tv, ti = np.load(d / "topv.npy"), np.load(d / "topi.npy")
    hit = np.isin(ti, list(needle_pos))
    qh += (tv.astype(np.float32) * hit).sum(-1).mean(0)
qh /= len(tasks)
top = np.dstack(np.unravel_index(np.argsort(-qh.ravel())[:12], qh.shape))[0]
print("\nquery heads with most top-32 attention on the needle (layer, qhead -> kv head, mean mass/step):")
print("  " + ", ".join(f"L{l}.h{h}(kv{h // 7}) {qh[l, h]:.2f}" for l, h in top))
res["query_heads"] = qh.tolist()
per_layer_needle = np.array(res["0.1"]["full_needle_layer"]) / res["0.1"]["full_needle_mass"]
print("needle-mass share by layer (full cache):", " ".join(f"{v:.2f}" for v in per_layer_needle))
json.dump(res, open(ROOT / "data/sim/failures.json", "w"))
