"""Qwen2.5-7B-Instruct (NF4) with a hand-written attention forward that exposes attention weights.

Prefill runs in token chunks; each chunk's attention is computed explicitly (scores -> fp32 softmax -> @V)
in query sub-chunks, which yields per-KV-head prefill statistics for the online policies in the same pass.
Decode runs either
  * full-cache ("trace"): records group-mean attention [layer, kv_head, key] per step (+ per-query-head top-k), or
  * live under eviction policies: a fixed slot buffer of B+1 entries per (layer, kv_head).
A KV head's attention to a key = mean over its 7 query heads (GQA group), so it sums to 1 over keys.
"""
import types

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import policies as P

MODEL = "Qwen/Qwen2.5-7B-Instruct"
EOS = {151645, 151643}
TOPK_Q = 32  # per-query-head top-k kept alongside the group-mean trace


class Engine:
    def __init__(self, model=MODEL, prefill_chunk=2048, q_chunk=256):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float16, device_map="cuda:0", attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16))
        self.model.eval()
        cfg = self.model.config
        self.nl, self.nq, self.nkv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.g = self.nq // self.nkv
        self.hd = cfg.hidden_size // self.nq
        self.pc, self.qc = prefill_chunk, q_chunk
        for layer in self.model.model.layers:
            layer.self_attn.forward = types.MethodType(_make_forward(self), layer.self_attn)
        self.mode = None

    # ------------------------------------------------------------------ attention dispatch
    def attend(self, li, q, k, v):
        """q [1,nq,s,d], k/v [1,nkv,s,d] (post-RoPE). Returns [1,nq,s,d]."""
        if self.mode == "prefill":
            return self._prefill_attn(li, q, k, v)
        if self.mode == "trace":
            return self._trace_attn(li, q, k, v)
        return self._live_attn(li, q, k, v)

    def _prefill_attn(self, li, q, k, v):
        s0 = self.off
        Kp, Vp = self.Kp[li], self.Vp[li]
        Kp[:, s0:s0 + k.shape[2]] = k[0]
        Vp[:, s0:s0 + k.shape[2]] = v[0]
        st = self.stats[li]
        s = q.shape[2]
        out = torch.empty_like(q)
        qg = q[0].view(self.nkv, self.g, s, self.hd)
        for a in range(0, s, self.qc):
            b = min(s, a + self.qc)
            e = s0 + b  # keys visible to the last query of this sub-chunk
            sc = torch.matmul(qg[:, :, a:b], Kp[:, :e].transpose(-1, -2).unsqueeze(1)).float() * self.scaling
            qpos = torch.arange(s0 + a, s0 + b, device=q.device)
            sc.masked_fill_(torch.arange(e, device=q.device)[None, :] > qpos[:, None], float("-inf"))
            w = sc.softmax(-1)
            o = torch.matmul(w.half(), Vp[:, :e].unsqueeze(1))
            out[0, :, a:b] = o.reshape(self.nq, b - a, self.hd)
            wm = w.mean(1)  # [nkv, c, e]
            st["colsum"][:, :e] += wm.sum(1)
            hit = (wm > P.TAU) * (qpos[None, :, None] + 1)
            st["lastuse"][:, :e] = torch.maximum(st["lastuse"][:, :e], hit.amax(1) - 1)
            win0 = self.L - P.SNAP_W
            if s0 + b > win0:
                sel = qpos >= win0
                st["window"][:, :e] += wm[:, sel].sum(1)
            del sc, w, wm, hit
        return out

    def _decode_scores(self, q, Ks):
        qg = q[0, :, 0].view(self.nkv, self.g, self.hd)
        return torch.matmul(qg, Ks.transpose(-1, -2)).float() * self.scaling  # [nkv, g, n]

    def _trace_attn(self, li, q, k, v):
        t, L = self.t, self.L
        self.Kg[li][:, t] = k[0, :, 0]
        self.Vg[li][:, t] = v[0, :, 0]
        sc = torch.cat([self._decode_scores(q, self.Kp[li]), self._decode_scores(q, self.Kg[li][:, : t + 1])], -1)
        w = sc.softmax(-1)
        wh = w.half()
        o = torch.matmul(wh[..., :L], self.Vp[li]) + torch.matmul(wh[..., L:], self.Vg[li][:, : t + 1])
        n = L + t + 1
        self.rec[li, :, :n] = w.mean(1).half()
        tv, ti = w.reshape(self.nq, n).topk(TOPK_Q, -1)
        self.rec_tv[li], self.rec_ti[li] = tv.half(), ti.int()
        return o.reshape(1, self.nq, 1, self.hd)

    def _live_attn(self, li, q, k, v):
        c = self.cache[li]
        rows = torch.arange(self.nkv, device=q.device)
        f = c["free"]
        c["K"][rows, f] = k[0, :, 0]
        c["V"][rows, f] = v[0, :, 0]
        c["pos"][rows, f] = self.cur
        pol = c["pol"]
        pol.on_insert(f, self.cur)
        w = self._decode_scores(q, c["K"]).softmax(-1)
        o = torch.matmul(w.half(), c["V"].unsqueeze(1))
        wm = w.mean(1)
        pol.observe(self.t, wm, c["pos"], self.cur)
        c["free"] = pol.evict(self.t, c["pos"], self.cur)
        return o.reshape(1, self.nq, 1, self.hd)

    # ------------------------------------------------------------------ model calls
    def _step(self, ids, start):
        pos = torch.arange(start, start + ids.shape[1], device="cuda")[None]
        h = self.model.model(input_ids=ids, position_ids=pos, use_cache=False).last_hidden_state
        return self.model.lm_head(h[:, -1]).float()

    @torch.no_grad()
    def prefill(self, prompt):
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        L = ids.shape[1]
        self.L = L
        z = lambda: torch.zeros(self.nkv, L, self.hd, dtype=torch.float16, device="cuda")
        self.Kp = [z() for _ in range(self.nl)]
        self.Vp = [z() for _ in range(self.nl)]
        self.stats = [dict(colsum=torch.zeros(self.nkv, L, device="cuda"),
                           window=torch.zeros(self.nkv, L, device="cuda"),
                           lastuse=torch.full((self.nkv, L), -1, dtype=torch.long, device="cuda"))
                      for _ in range(self.nl)]
        self.mode = "prefill"
        for s0 in range(0, L, self.pc):
            self.off = s0
            logits = self._step(ids[:, s0:s0 + self.pc], s0)
        self.first = int(logits.argmax(-1))
        return L

    @torch.no_grad()
    def decode_trace(self, max_new):
        """Full-cache greedy decode. Returns (tokens, trace [T,nl,nkv,L+T] fp16, topv, topi)."""
        L, T = self.L, max_new - 1
        z = lambda: torch.zeros(self.nkv, T, self.hd, dtype=torch.float16, device="cuda")
        self.Kg = [z() for _ in range(self.nl)]
        self.Vg = [z() for _ in range(self.nl)]
        self.rec = torch.zeros(self.nl, self.nkv, L + T, dtype=torch.float16, device="cuda")
        self.rec_tv = torch.zeros(self.nl, self.nq, TOPK_Q, dtype=torch.float16, device="cuda")
        self.rec_ti = torch.zeros(self.nl, self.nq, TOPK_Q, dtype=torch.int32, device="cuda")
        trace = np.zeros((T, self.nl, self.nkv, L + T), dtype=np.float16)
        topv = np.zeros((T, self.nl, self.nq, TOPK_Q), dtype=np.float16)
        topi = np.zeros((T, self.nl, self.nq, TOPK_Q), dtype=np.int32)
        self.mode = "trace"
        toks = [self.first]
        for t in range(T):
            if toks[-1] in EOS:
                break
            self.t = t
            self.rec.zero_()
            logits = self._step(torch.tensor([[toks[-1]]], device="cuda"), L + t)
            trace[t] = self.rec.cpu().numpy()
            topv[t], topi[t] = self.rec_tv.cpu().numpy(), self.rec_ti.cpu().numpy()
            toks.append(int(logits.argmax(-1)))
        steps = len(toks) - 1  # decode forward passes run (last token is never fed back)
        return toks, trace[:steps, :, :, : L + steps], topv[:steps], topi[:steps]

    @torch.no_grad()
    def decode_live(self, factory, B, max_new):
        """Greedy decode under eviction. factory(layer, stats) -> (policy, keep [nkv,B])."""
        L = self.L
        self.cache = []
        rows = torch.arange(self.nkv, device="cuda")[:, None]
        for li in range(self.nl):
            pol, keep = factory(li, self.stats[li])
            keep = keep.cuda()
            S = B + 1
            K = torch.zeros(self.nkv, S, self.hd, dtype=torch.float16, device="cuda")
            V = torch.zeros_like(K)
            pos = torch.full((self.nkv, S), -1, dtype=torch.long, device="cuda")
            K[:, :B], V[:, :B], pos[:, :B] = self.Kp[li][rows, keep], self.Vp[li][rows, keep], keep
            pol.init_state(pos)
            self.cache.append(dict(K=K, V=V, pos=pos, pol=pol,
                                   free=torch.full((self.nkv,), B, dtype=torch.long, device="cuda")))
        self.mode = "live"
        toks = [self.first]
        for t in range(max_new - 1):
            if toks[-1] in EOS:
                break
            self.t, self.cur = t, L + t
            logits = self._step(torch.tensor([[toks[-1]]], device="cuda"), L + t)
            toks.append(int(logits.argmax(-1)))
        self.cache = None
        return toks

    def text(self, toks):
        return self.tok.decode([x for x in toks if x not in EOS], skip_special_tokens=True).strip()


def _make_forward(eng):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kw):
        shp = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_proj(hidden_states).view(shp).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shp).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shp).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        eng.scaling = self.scaling
        o = eng.attend(self.layer_idx, q, k, v)
        o = o.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
        return self.o_proj(o), None

    return forward
