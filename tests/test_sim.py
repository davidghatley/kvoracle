import torch, numpy as np
from kvo import policies as P, sim

def synth(T=40, H=6, L=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = L + T
    A = torch.zeros(T, H, N)
    hot = torch.randint(0, L, (H, 12), generator=g)
    for t in range(T):
        x = torch.randn(H, N, generator=g) * 1.5
        x[:, :4] += 5                                   # sinks
        x.scatter_add_(1, hot[:, (t // 5) % 12:(t // 5) % 12 + 3], torch.full((H, 3), 6.0))  # shifting hot tokens
        x[:, L + t + 1:] = -1e9
        x[:, max(0, L + t - 8):L + t + 1] += 2          # recency
        A[t] = x.softmax(-1)
    # fake prefill stats
    st = dict(colsum=torch.rand(H, L, generator=g) * 10, window=torch.rand(H, L, generator=g),
              lastuse=torch.randint(0, L, (H, L), generator=g))
    return A, st, L

def test_bounds_and_replay():
    A, st, L = synth()
    B = 30
    ref = sim.reference(A, L, B)
    fut = P.Future(A)
    res = {c.name: sim.simulate(A, st, L, B, c, future=fut) if c.oracle else sim.simulate(A, st, L, B, c)
           for c in P.ONLINE + P.ORACLES}
    for k, r in res.items():
        assert (r["mass"] <= ref["ceiling"] + 1e-6).all(), k
        assert (r["mass"] <= ref["full"] + 1e-6).all(), k
    w = res["oracle_w"]
    for k in ["random", "lru", "streaming", "h2o", "snapkv"]:
        assert w["mass"].sum() >= res[k]["mass"].sum(), k
    rp = sim.simulate(A, st, L, B, P.Replay, keep=w["keep"], evicted=w["evicted"])
    assert torch.allclose(rp["mass"], w["mass"]), (rp["mass"], w["mass"])
    lp, frac = sim.exact_window_lp(A[:, 0].numpy().astype(np.float64), L, B)
    assert lp >= w["mass"][0].item() - 1e-6
    print({k: round(v["mass"].sum().item() / ref["full"].sum().item(), 3) for k, v in res.items()},
          "ceil", round(ref["ceiling"].sum().item() / ref["full"].sum().item(), 3),
          "lp/greedy head0", round(lp / w["mass"][0].item(), 4), "frac", round(frac, 3))

if __name__ == "__main__":
    test_bounds_and_replay()
