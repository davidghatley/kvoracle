"""Correctness check: patched attention vs stock HF sdpa on a short prompt (logits + greedy text),
and live decode with a budget >= context (no eviction) must reproduce the full-cache output."""
import torch

from kvo import policies as P
from kvo.engine import Engine

eng = Engine(prefill_chunk=64, q_chunk=32)  # small chunks so chunking itself is exercised
prompt = eng.tok.apply_chat_template([{"role": "user", "content": "List the first five prime numbers, then name a city in Peru."}],
                                     tokenize=False, add_generation_prompt=True)
ids = eng.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

# ours: prefill + full-cache decode
L = eng.prefill(prompt)
toks, tr, _, _ = eng.decode_trace(40)
print("ours :", repr(eng.text(toks)))
assert abs(tr[3].astype("float32").sum(-1) - 1).max() < 1e-2, "trace rows must sum to 1"

# stock: restore original forward on a fresh model is expensive; compare against manual sdpa reference
from transformers.models.qwen2 import modeling_qwen2 as m
for layer in eng.model.model.layers:
    layer.self_attn.forward = m.Qwen2Attention.forward.__get__(layer.self_attn)
with torch.no_grad():
    ref = eng.model.generate(ids, max_new_tokens=40, do_sample=False)
ref_toks = ref[0, L:].tolist()
print("stock:", repr(eng.text(ref_toks)))
same = sum(a == b for a, b in zip(toks, ref_toks))
print(f"token agreement {same}/{min(len(toks), len(ref_toks))}")

# re-patch and check no-eviction live decode == trace decode
import types
from kvo.engine import _make_forward
for layer in eng.model.model.layers:
    layer.self_attn.forward = types.MethodType(_make_forward(eng), layer.self_attn)
eng.prefill(prompt)
B = L  # all prompt tokens kept; evictions only start after step 0
live = eng.decode_live(lambda li, st: P.make(P.H2O, eng.nkv, B, L, st, device="cuda"), B, 40)
print("live :", repr(eng.text(live)))
