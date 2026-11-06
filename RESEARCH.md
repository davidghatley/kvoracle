# RESEARCH — How close do KV-cache eviction heuristics get to an offline oracle?

Single source of truth for this project. Read before each step, write back after each step.

## Question
How much of the attention mass that an offline, clairvoyant (Belady-style) eviction oracle
retains do online eviction heuristics (random, LRU, StreamingLLM, H2O, SnapKV) retain, and
how does that translate into task accuracy, across budgets and task types?

## Setup (Step 0, 2026-11-05)
- Hardware: 1× RTX 3060 12 GB (shared; every GPU job goes through `~/anime-pipeline/tools/gpu_run`), 64 GB RAM.
- Model: **Qwen2.5-7B-Instruct** (ungated, 32k native context), bitsandbytes NF4 4-bit weights,
  fp16 compute, HF transformers 5.22.1, torch 2.15.
  28 layers, 28 query heads, 4 KV heads (GQA, 7 query heads per KV head), head dim 128.
- Attention access: "eager" semantics via a patched attention forward (see Decisions D1).

## Predictions (recorded before any run)
- **P1:** heuristics recover >90% of oracle attention mass on summarization.
- **P2:** heuristics recover <60% of oracle attention mass on needle-in-a-haystack retrieval.
- **P3:** the oracle-vs-heuristic gap widens as the cache budget shrinks.

Operationalization (fixed now, before results): "heuristics" = the best online policy among
StreamingLLM / H2O / SnapKV (random and LRU are floors, not "current heuristics"); a prediction is
judged per budget and reported for all of them. P1/P2 are judged at the 10% budget as the headline,
with the other budgets reported. "Recover" = (mass retained by policy) / (mass retained by the
weighted oracle), averaged over prompts, layers and KV heads.

## Decisions
(dated; deviations from the brief are logged here)

- **D1 (2026-11-05) Eviction unit = (layer, KV head); "attention to a token" = mean over the 7 query heads
  of the GQA group.** A KV entry is shared by the group, so it can only be kept or evicted per KV head.
  112 independent caches (28 layers × 4 KV heads), each with the same budget B. Per-query-head detail is kept
  in the traces as top-32 per query head for Step 5. Offline, policies see trace attention renormalized over
  the slots they hold (exact for a single head; approximate for a group mean, because each query head has its
  own softmax normalizer).
- **D2 (2026-11-05) Attention access.** Instead of `attn_implementation="eager"` (which would build a
  16k×16k×28 attention tensor per layer, ~14 GB), a patched attention forward computes the same eager math
  (QKᵀ·scale → fp32 softmax → ·V) in query chunks of 256, prompt chunks of 2048 tokens. Same numbers as
  eager, bounded memory. Verified against stock HF generation (scripts/check_engine.py).
- **D3 (2026-11-05) Protocol.** Prefill always uses the full cache; at the end of prefill each head is
  compressed to B = budget × prompt length past tokens; during decode one token is evicted per step so the
  cache never holds more than B past tokens (+ the current token). Metric (a) counts only attention to past
  tokens. "Significant use" threshold **τ = 0.01** on group-mean attention (≈100× uniform at 10k keys), fixed
  a priori; used by LRU and Belady. Belady is also reported at τ ∈ {0.001, 0.003, 0.03} as a sensitivity
  check on the *oracle* (no online policy is tuned).
- **D4 (2026-11-05) Tasks.** Needle: 10 synthetic, 9.4k–14.5k tokens, depths 10–90%, haystack = gov_report
  documents too short to be eval docs; answer = 7-digit number, scored by exact presence. Summarization:
  LongBench gov_report, first 10 rows in file order with 8k–16k-token prompts, ROUGE-L, generation capped at
  256 tokens (LongBench uses 512; halved for trace size and time). Multi-doc QA: LongBench hotpotqa, same
  selection rule, token F1, max 32 new tokens. One extra prompt per task is a dev prompt for smoke tests only.
  Greedy decoding throughout.
- **D5 (2026-11-05) Heuristic definitions (paper defaults, nothing tuned):**
  random (uniform, seeded); LRU (last significant use, insertion counts as a use); StreamingLLM (4 sinks +
  recent window; original positions kept, no position re-indexing); H2O (accumulated attention incl. prefill,
  50% heavy / 50% recent); SnapKV (last 32 prompt queries vote, max-pool kernel 7, window kept). SnapKV only
  compresses the prompt in the paper; to hold a fixed budget during decode we extend it as a sliding
  observation window: votes from the last 32 decode queries, the 32 most recent tokens protected, pooled
  prefill votes as prior for the first 32 steps and tie-breaker after.
- **D6 (2026-11-05) Oracles.** (i) Belady(τ): evict the token whose next use above τ is furthest away (random
  tie-break). (ii) Weighted greedy: evict the token with least remaining future attention mass; initial keep
  set = top-B by total future mass. (iii) Ceiling: per-step top-B (allows re-admission, so no real policy can
  reach it). (iv) LP relaxation of the exact weighted problem on a 32-step window for a sample of heads, an
  upper bound on the exact optimum, to measure how far the greedy oracle is from optimal. Headline ratio =
  policy / weighted greedy. Live "oracle_replay" replays the weighted greedy's offline decisions.
- **D7 (2026-11-06) The first generated token is always full-cache.** Under prefill-then-compress, the
  first output token comes from the prefill pass. Qwen tokenizes digits singly, so for the needle the first
  of 7 digits is free and eviction acts on the remaining 6 digits + EOS (T = 7 decode steps). This favors
  every policy equally on accuracy; the offline metric only covers the T decode steps.
- **Note on Belady (required by brief):** Belady's MIN is optimal for unit-cost hits with re-fetch on a miss.
  Neither holds here: hits are weighted by attention, and **an evicted KV entry can never be re-fetched**
  (eviction is permanent), so "furthest next use" can throw away a token needed later in favor of one needed
  soon but rarely. The τ-thresholded Belady is therefore an approximation of the optimum, not the optimum.

## Results

### Step 1 — traces (2026-11-06)
Command: `PYTHONPATH=. gpu_run --free .venv/bin/python scripts/record_traces.py --task <needle|summ|mdqa>`.
Stored per prompt in `data/traces/<id>/` (on /mnt/work, not in git).

| task | prompts | prompt tokens | decode steps T | group-mean trace (fp16) | per-query-head top-32 | full-cache score |
|---|---|---|---|---|---|---|
| needle | 10 | 9,418–14,470 | 7 each | 172 MB | 10 MB | 1.000 (exact match) |
| summarization (gov_report) | 10 | 8,165–15,864 | 255 each (all hit the cap) | 6,289 MB | 366 MB | 0.182 ROUGE-L |
| multi-doc QA (hotpotqa) | 10 | 9,275–15,750 | 2–6, one 31 | 194 MB | 10 MB | 0.561 F1 |
| **total** | 30 | | | **6.5 GB** | **0.39 GB** | |

- Format: `trace.npy` fp16 [T, 28 layers, 4 KV heads, L+T], full (not top-k), so the offline sim is exact.
  Storing every *query* head in fp16 would be 7× larger (~45 GB): feasible on disk but unnecessary, so
  per-query-head detail is kept as top-32 only. Prefill stats (`stats.npz`) are ~10 MB per prompt.
- Cost: prefill 9–27 s per prompt (fp32 scores, chunked), decode ~0.04 s/step on the 3060, peak 7.6 GB VRAM.
- Traces are small enough (no "too large" surprise). Summarization dominates (96% of bytes) because T = 255.
- Horizon caveat: needle and QA have 2–7 decode steps, so the oracle sees a very short future there.

## Surprises
- **S1 (2026-11-06, before Step 1) fp16 eager attention breaks Qwen2.5-7B.** The first engine check with QKᵀ in
  fp16 (as HF eager does) produced garbage ("!!!" tokens, 0/40 agreement with stock SDPA generation). Layer-0
  keys reach |k| ≈ 172, so fp16 logits overflow. Fix: scores computed in fp32 (q·scale and K cast to fp32);
  after the fix per-layer outputs match SDPA within 2e-3. Implication: attention traces from "eager fp16"
  setups on Qwen2 models can be silently corrupt; ours are fp32-softmax, stored as fp16 probabilities.
- **S2 (2026-11-06, Step 2 on needle traces) The Belady oracle loses to online heuristics.** At 10% budget on
  needle, Belady(τ=0.01) keeps 0.82 of the weighted oracle's mass; online SnapKV keeps 0.94, LRU 0.87,
  H2O 0.84. Mechanism (checked on needle-00/05): per head only a median of **14–15 prompt tokens ever exceed
  τ** in the 7 decode steps, while B ≈ 1,150–1,450. Belady keeps those and fills the other ~99% of slots by
  tie-break (random among "never used"), discarding the sub-threshold bulk that carries 27–31% of the mass
  (median share of mass above τ: 0.69–0.73). Belady with a binary "use" is the wrong oracle when value is
  diffuse; this is the unit-cost assumption failing, plus no re-fetch. The τ sweep (0.001–0.03) tests
  whether a lower threshold rescues it.
- **S3 (2026-11-06) Needle "attention mass" is dominated by easy mass.** Median per head: 32–34% of
  past-token mass on the 4 sink tokens, 29% on the last 64 prompt tokens (the question), ~33–36% on
  everything else. A policy that keeps sinks + recent already gets ~60% of the mass, so mass ratios on
  needle are high for every structured heuristic; whether that predicts retrieval is what live accuracy
  (Step 4b) and the needle-token analysis (Step 5) must answer.

## Next steps
- [x] Step 0 — setup and predictions
- [x] Step 1 — record attention traces (6.5 GB)
- [ ] Step 2 — build the oracle (Belady + weighted)
- [ ] Step 3 — online baselines
- [ ] Step 4 — measure (offline mass + live accuracy), judge P1–P3
- [ ] Step 5 — read failures, mechanism hypothesis

## Open questions
