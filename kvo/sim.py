"""Offline eviction simulation over full-cache attention traces.

A [T, H, N]: group-mean attention of decode step t (query at position L+t) to key positions (N = L+T).
Retained mass at step t = attention (from the full-cache trace) landing on the B *past* tokens in the cache;
the current token is always present and excluded from every number. Policies observe the trace attention
renormalized over the slots they hold, which is what they would see live (up to the per-query-head
normalizer, since a KV head's attention is the mean of 7 query heads; RESEARCH.md D1).
"""
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from . import policies as P


@torch.no_grad()
def simulate(A, stats, L, B, cls, seed=0, **kw):
    T, H, N = A.shape
    pol, keep = P.make(cls, H, B, L, stats, seed=seed, **kw)
    S = B + 1
    rows = torch.arange(H)
    pos = torch.full((H, S), -1, dtype=torch.long)
    pos[:, :B] = keep
    pol.init_state(pos)
    free = torch.full((H,), B, dtype=torch.long)
    mass = torch.zeros(H, dtype=torch.float64)
    ev = torch.zeros(T, H, dtype=torch.long)
    for t in range(T):
        cur = L + t
        pos[rows, free] = cur
        pol.on_insert(free, cur)
        a = A[t].gather(1, pos)
        mass += (a * (pos != cur)).sum(-1).double()
        pol.observe(t, a / a.sum(-1, keepdim=True).clamp_min(1e-12), pos, cur)
        free = pol.evict(t, pos, cur)
        ev[t] = pos[rows, free]
    return dict(mass=mass, keep=keep, evicted=ev)


@torch.no_grad()
def reference(A, L, B):
    """Per head: total past attention mass (full cache) and the per-step top-B ceiling (re-admission allowed)."""
    T, H, N = A.shape
    full = torch.zeros(H, dtype=torch.float64)
    ceil = torch.zeros(H, dtype=torch.float64)
    for t in range(T):
        past = A[t, :, : L + t]
        full += past.sum(-1).double()
        ceil += past.topk(B, -1).values.sum(-1).double()
    return dict(full=full, ceiling=ceil)


def exact_window_lp(a, L, B):
    """LP relaxation of the exact weighted-eviction problem for one head on a short window.
    a: [Tw, L+Tw] numpy attention. Variables x[j,t] = token j held at step t (j < L+t).
    max sum a[t,j] x[j,t]  s.t.  sum_j x[j,t] <= B;  x[j,t] <= x[j,t-1] (no re-admission);  0<=x<=1.
    Returns (LP optimum, max fractional distance from integrality). The LP value upper-bounds the exact optimum."""
    Tw = a.shape[0]
    sizes = L + np.arange(Tw)
    off = np.concatenate([[0], np.cumsum(sizes)])
    n = int(off[-1])
    cost = -np.concatenate([a[t, : sizes[t]] for t in range(Tw)])
    # capacity rows 0..Tw-1
    r_cap = np.repeat(np.arange(Tw), sizes)
    c_cap = np.arange(n)
    # monotone rows: x[j,t] - x[j,t-1] <= 0 for j < L+t-1
    r_mon, c_a, c_b = [], [], []
    m = Tw
    for t in range(1, Tw):
        j = np.arange(sizes[t - 1])
        rr = m + j
        r_mon.append(rr); c_a.append(off[t] + j); c_b.append(off[t - 1] + j)
        m += len(j)
    r_mon = np.concatenate(r_mon) if r_mon else np.zeros(0, int)
    c_a = np.concatenate(c_a) if c_a else np.zeros(0, int)
    c_b = np.concatenate(c_b) if c_b else np.zeros(0, int)
    r = np.concatenate([r_cap, r_mon, r_mon]); c = np.concatenate([c_cap, c_a, c_b])
    v = np.concatenate([np.ones(n), np.ones(len(c_a)), -np.ones(len(c_b))])
    b = np.concatenate([np.full(Tw, float(B)), np.zeros(m - Tw)])
    Aub = coo_matrix((v, (r, c)), shape=(m, n)).tocsr()
    res = linprog(cost, A_ub=Aub, b_ub=b, bounds=(0, 1), method="highs")
    x = res.x
    frac = float(np.minimum(x, 1 - x).max())
    return -res.fun, frac
