"""Eviction policies over a fixed slot buffer, shared by the offline simulator and live decoding.

Shapes: H heads (each an independent (layer, KV-head) cache), S = B + 1 slots (B past tokens plus the
token currently being decoded). Per decode step t (absolute position cur = L + t):
  1. the free slot receives the current token          -> on_insert(slot, cur)
  2. attention over all S slots is computed            -> observe(t, attn, pos, cur)   (attn sums to 1 over slots)
  3. one slot per head is freed                        -> evict(t, pos, cur) -> slot [H]
Prefill selection picks the B prompt positions to keep from per-head prefill statistics:
  colsum [H,L]  attention summed over all prefill queries      (H2O)
  window [H,L]  attention summed over the last 32 queries      (SnapKV)
  lastuse [H,L] last query index paying > TAU to the token     (LRU)
All online policies see only the past. Oracles (Belady, WeightedGreedy) receive a Future object.
"""
import torch
import torch.nn.functional as F

TAU = 0.01          # "significant use" threshold on group-mean attention, fixed a priori (RESEARCH.md D3)
SINKS = 4           # StreamingLLM default
SNAP_W, SNAP_K = 32, 7   # SnapKV defaults: observation window, max-pool kernel
INF = float("inf")


def _topk_pos(score, k):
    return score.topk(k, dim=-1).indices.sort(-1).values


class Policy:
    name = "base"
    oracle = False

    def __init__(self, H, B, L, device="cpu", seed=0):
        self.H, self.B, self.L, self.dev = H, B, L, device
        self.g = torch.Generator(device=device).manual_seed(seed)

    def select_prefill(self, stats):
        raise NotImplementedError

    def init_state(self, pos):
        pass

    def on_insert(self, slot, cur):
        pass

    def observe(self, t, attn, pos, cur):
        pass

    def evict(self, t, pos, cur):
        raise NotImplementedError

    def _jitter(self, shape):
        return torch.rand(shape, generator=self.g, device=self.dev) * 1e-6

    def _rows(self):
        return torch.arange(self.H, device=self.dev)


class Random(Policy):
    name = "random"

    def select_prefill(self, stats):
        return _topk_pos(torch.rand(self.H, self.L, generator=self.g, device=self.dev), self.B)

    def evict(self, t, pos, cur):
        return torch.randint(0, pos.shape[1], (self.H,), generator=self.g, device=self.dev)


class LRU(Policy):
    """Least recently *significantly* used: a use is group-mean attention > TAU; insertion counts as a use."""
    name = "lru"

    def select_prefill(self, stats):
        ar = torch.arange(self.L, device=self.dev).float()
        lu = torch.maximum(stats["lastuse"].float(), ar)
        return _topk_pos(lu + ar / (self.L + 1), self.B)

    def init_state(self, pos):
        self.lu = pos.clone().float()
        self.lu[:, : self.B] = torch.maximum(self.lu[:, : self.B], self.stats["lastuse"].gather(1, pos[:, : self.B]).float())

    def on_insert(self, slot, cur):
        self.lu[self._rows(), slot] = float(cur)

    def observe(self, t, attn, pos, cur):
        self.lu = torch.where(attn > TAU, float(cur), self.lu)

    def evict(self, t, pos, cur):
        return (self.lu + pos.float() / (cur + 1)).argmin(-1)


class StreamingLLM(Policy):
    name = "streaming"

    def select_prefill(self, stats):
        sinks = torch.arange(SINKS, device=self.dev)
        recent = torch.arange(self.L - (self.B - SINKS), self.L, device=self.dev)
        return torch.cat([sinks, recent]).expand(self.H, -1).clone()

    def evict(self, t, pos, cur):
        return torch.where(pos < SINKS, INF, pos.float()).argmin(-1)


class H2O(Policy):
    """Heavy hitters by accumulated attention + recent window; budget split 50/50 (paper default)."""
    name = "h2o"

    def select_prefill(self, stats):
        self.r = self.B // 2
        sc = stats["colsum"].clone().float()
        sc[:, self.L - self.r:] = INF
        return _topk_pos(sc, self.B)

    def init_state(self, pos):
        self.r = self.B // 2
        self.score = torch.zeros(pos.shape, device=self.dev)
        self.score[:, : self.B] = self.stats["colsum"].gather(1, pos[:, : self.B]).float()

    def on_insert(self, slot, cur):
        self.score[self._rows(), slot] = 0.0

    def observe(self, t, attn, pos, cur):
        self.score += attn

    def evict(self, t, pos, cur):
        protected = pos > cur - self.r
        return torch.where(protected, INF, self.score).argmin(-1)


class SnapKV(Policy):
    """Observation-window voting. Prefill: last W queries vote, max-pooled (kernel 7), window kept.
    Decode (our extension so the budget holds): votes from the last W decode queries; recent W protected;
    pooled prefill votes act as the prior for the first W steps and as a tie-breaker after (RESEARCH.md D5)."""
    name = "snapkv"

    def select_prefill(self, stats):
        v = F.max_pool1d(stats["window"].float()[:, None], SNAP_K, 1, SNAP_K // 2)[:, 0]
        self._pooled = v
        sc = v.clone()
        sc[:, self.L - SNAP_W:] = INF
        return _topk_pos(sc, self.B)

    def init_state(self, pos):
        S = pos.shape[1]
        self.ring = torch.zeros(SNAP_W, self.H, S, device=self.dev)
        self.prior = torch.zeros(self.H, S, device=self.dev)
        self.prior[:, : self.B] = self._pooled.gather(1, pos[:, : self.B])

    def on_insert(self, slot, cur):
        r = self._rows()
        self.ring[:, r, slot] = 0.0
        self.prior[r, slot] = 0.0

    def observe(self, t, attn, pos, cur):
        self.ring[t % SNAP_W] = attn
        self.t = t

    def evict(self, t, pos, cur):
        votes = self.ring.sum(0) + self.prior * (1.0 if t < SNAP_W else 1e-3)
        protected = pos > cur - SNAP_W
        return torch.where(protected, INF, votes).argmin(-1)


# ---------------------------------------------------------------- oracles (need the future)

class Future:
    """Full-cache future for H heads. A: [T, H, N] attention (step t -> key position), N = L + T.
    suf[t] = sum_{t' >= t} A[t'] ; nu[t] = first t' >= t with A[t'] > tau (T if none)."""

    def __init__(self, A, tau=TAU):
        T = A.shape[0]
        self.T = T
        A = A.float()
        self.suf = torch.zeros((T + 1,) + A.shape[1:])
        self.suf[:T] = A.flip(0).cumsum(0).flip(0)
        self.nu = torch.full((T + 1,) + A.shape[1:], T, dtype=torch.int16)
        for t in range(T - 1, -1, -1):
            self.nu[t] = torch.where(A[t] > tau, t, self.nu[t + 1])

    def suf_at(self, t, pos):
        return self.suf[min(t, self.T)].gather(1, pos)

    def nu_at(self, t, pos):
        return self.nu[min(t, self.T)].gather(1, pos.long()).float()


class Belady(Policy):
    """Evict the token whose next significant use (attention > tau) is furthest in the future; random ties."""
    name = "belady"
    oracle = True

    def __init__(self, *a, future=None, **k):
        super().__init__(*a, **k)
        self.f = future

    def select_prefill(self, stats):
        nu0 = self.f.nu[0][:, : self.L].float()
        return _topk_pos(-nu0 + self._jitter(nu0.shape), self.B)

    def evict(self, t, pos, cur):
        nu = self.f.nu_at(t + 1, pos)
        return (nu + self._jitter(nu.shape)).argmax(-1)


class WeightedGreedy(Policy):
    """Evict the token with the least remaining future attention mass."""
    name = "oracle_w"
    oracle = True

    def __init__(self, *a, future=None, **k):
        super().__init__(*a, **k)
        self.f = future

    def select_prefill(self, stats):
        return _topk_pos(self.f.suf[0][:, : self.L], self.B)

    def evict(self, t, pos, cur):
        return self.f.suf_at(t + 1, pos).argmin(-1)


class Replay(Policy):
    """Live replay of an oracle's offline decisions: kept set at prefill + evicted position per step.
    Past the recorded horizon (live output longer than the full-cache run) it evicts the newest token."""
    name = "oracle_replay"
    oracle = True

    def __init__(self, *a, keep=None, evicted=None, **k):
        super().__init__(*a, **k)
        self.keep, self.ev = keep.to(self.dev), evicted.to(self.dev)

    def select_prefill(self, stats):
        return self.keep.clone()

    def evict(self, t, pos, cur):
        if t < self.ev.shape[0]:
            hit = pos == self.ev[t][:, None]
            if hit.any(-1).all():
                return hit.float().argmax(-1)
        return (pos == cur).float().argmax(-1)


ONLINE = [Random, LRU, StreamingLLM, H2O, SnapKV]
ORACLES = [Belady, WeightedGreedy]


def make(cls, H, B, L, stats, device="cpu", seed=0, **kw):
    """Instantiate, run prefill selection, return (policy, keep positions [H,B])."""
    p = cls(H, B, L, device=device, seed=seed, **kw)
    p.stats = stats
    keep = p.select_prefill(stats)
    return p, keep
