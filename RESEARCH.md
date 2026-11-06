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
- **Note on Belady (required by brief):** Belady's MIN is optimal for unit-cost hits with re-fetch on a miss.
  Neither holds here: hits are weighted by attention, and **an evicted KV entry can never be re-fetched**
  (eviction is permanent), so "furthest next use" can throw away a token needed later in favor of one needed
  soon but rarely. The τ-thresholded Belady is therefore an approximation of the optimum, not the optimum.

## Results
_None yet._

## Surprises
_None yet._

## Next steps
- [x] Step 0 — setup and predictions
- [ ] Step 1 — record attention traces
- [ ] Step 2 — build the oracle (Belady + weighted)
- [ ] Step 3 — online baselines
- [ ] Step 4 — measure (offline mass + live accuracy), judge P1–P3
- [ ] Step 5 — read failures, mechanism hypothesis

## Open questions
