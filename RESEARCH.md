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
