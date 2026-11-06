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

### Step 2 — the oracles (2026-11-06)
Scripts: `scripts/run_sim.py` (all 30 prompts × 4 budgets), `scripts/exact_check.py` (LP bound).
All numbers are retained attention mass summed over heads and decode steps, as a fraction of the weighted
oracle, averaged over the 10 prompts of a task.

**Is weighted greedy actually optimal?** For each of 6 prompts (2 per task) × 6 KV heads × budgets 5% and 25%,
I solved the LP relaxation of the exact problem (choose which tokens to hold at every step, at most B held,
no re-admission after eviction) over the first Tw = min(T, 32) decode steps with HiGHS, and compared it with
greedy on the same window.

| task | cases | greedy / LP (mean) | worst | LP solutions fractional |
|---|---|---|---|---|
| needle | 24 | 0.99999 | 0.99992 | 0 of 24 |
| summarization | 24 | 0.99988 | 0.99935 | 0 of 24 |
| multi-doc QA | 24 | 1.00000 | 0.99998 | 0 of 24 |

The LP was integral every time, so its value is the exact optimum. Greedy is within 0.07% of it in the worst case.
So for this objective "weighted oracle" ≈ "optimal", at least on 32-step windows. Why: each token's future
mass changes little over the window, and eviction is one token per step, so evicting the token with the least
remaining mass is nearly always right. (The sim's per-step top-B *ceiling* is above 1.0 because it may
re-admit evicted tokens; it is not achievable under permanent eviction.)

**Belady vs weighted.** Belady (evict the token whose next use with attention > τ is furthest away) diverges
from the weighted oracle in both mass and membership:

| task | budget | Belady τ=0.001 | τ=0.003 | τ=0.01 (pre-registered) | τ=0.03 | keep-set Jaccard vs weighted (τ=0.01) |
|---|---|---|---|---|---|---|
| needle | 5% | 0.929 | 0.892 | 0.841 | 0.753 | 0.033 |
| needle | 10% | 0.909 | 0.873 | 0.824 | 0.741 | 0.046 |
| needle | 25% | 0.904 | 0.876 | 0.836 | 0.769 | 0.142 |
| needle | 50% | 0.935 | 0.922 | 0.904 | 0.872 | 0.358 |
| summarization | 5% | 0.939 | 0.912 | 0.827 | 0.675 | 0.161 |
| summarization | 10% | 0.937 | 0.899 | 0.794 | 0.648 | 0.129 |
| summarization | 25% | 0.937 | 0.872 | 0.767 | 0.635 | 0.151 |
| summarization | 50% | 0.948 | 0.904 | 0.822 | 0.711 | 0.386 |
| multi-doc QA | 5% | 0.914 | 0.860 | 0.788 | 0.683 | 0.034 |
| multi-doc QA | 10% | 0.889 | 0.840 | 0.774 | 0.676 | 0.052 |
| multi-doc QA | 25% | 0.886 | 0.847 | 0.796 | 0.719 | 0.148 |
| multi-doc QA | 50% | 0.921 | 0.900 | 0.873 | 0.833 | 0.355 |

- Belady recovers 64–95% of the weighted oracle's mass, depending on τ, and is monotonically worse as τ grows.
  Even at τ = 0.001 it does not close the gap.
- The two oracles keep almost disjoint sets: Jaccard 0.03–0.16 at ≤ 25% budget. Most held tokens never cross
  τ, so Belady is choosing among ties. That is surprise S2: unit-cost Belady is the wrong oracle when
  hits are weighted, and it loses to SnapKV and H2O.
- The full cache's mass is the absolute upper limit. The weighted oracle gets 0.81–0.99 of it (lowest:
  summarization at 5%, where 255 decode steps spread attention widely).
- From here on, "oracle" means the weighted greedy oracle. Belady@0.01 is reported alongside it for reference.

### Step 3 — online baselines (2026-11-06)
Implemented in `kvo/policies.py` and shared between the offline sim and the live engine (one class per
policy, same interface). Definitions are in D5. None of them sees the future. Each prompt is simulated at
5/10/25/50% budgets with a seed fixed by `zlib.crc32(id)`. Nothing was tuned: hyperparameters are the papers'
defaults (StreamingLLM 4 sinks + recent window; H2O half heavy-hitter / half recent; SnapKV window 32,
pool kernel 7), plus τ = 0.01 for LRU's "use", all fixed before any eval run. The dev prompts (`tasks_dev.jsonl`)
were used only for engine and plumbing checks. Live versions run in `scripts/run_live.py`, alongside an
`oracle_replay` policy that replays the weighted oracle's offline eviction log.

Offline mass / oracle at 10% (full tables in Step 4):

| task | random | LRU | StreamingLLM | H2O | SnapKV |
|---|---|---|---|---|---|
| needle | 0.220 | 0.875 | 0.772 | 0.835 | **0.945** |
| summarization | 0.464 | 0.854 | 0.821 | **0.891** | 0.871 |
| multi-doc QA | 0.186 | 0.889 | 0.842 | 0.885 | **0.924** |

### Step 5 — reading the failures (needle, 2026-11-06)
Script: `scripts/failures.py` (+ digit check). Choice of combination: on live accuracy LRU, StreamingLLM and
H2O tie at 0/10 on needle at 5% and 10%. By the rule "lowest offline mass among the tied", the worst is
**StreamingLLM on needle**; it is also the trivial case (it keeps no needle token by construction, needle mass
kept 0.000). The informative failures are **H2O** (content-aware, still 0/10) and **LRU** (keeps 0.59 of the
needle's mass, still 0/10), so all three are read.

Where the answer lives:
- The needle sentence is 18 tokens (7 of them digits), 0.15% of the prompt, yet it draws **10.1% of all
  past-token attention** during the answer. 432 of 1,120 (prompt × KV head) caches pay > 5% of their attention
  to it, i.e. ~43 of 112 KV heads per prompt.
- Needle attention is concentrated in middle-to-late layers: layers 14 (13%), 22 (13%), 16 (10%), 19 (9%),
  23 (7%). Top query heads by needle attention (mass of their top-32 on needle positions per step): L14.h0 0.99,
  L15.h20 0.96, L22.h3 0.90, L22.h4 0.88, L11.h24 0.86, L14.h6 0.83, L17.h8 0.83. These match the
  "retrieval head" pattern: a few heads that copy from the needle almost exclusively.

What the oracle kept that the heuristics evicted (10% budget, share of the oracle-kept-but-evicted mass):

| policy | needle-token mass kept (vs full cache) | needle heads with zero needle tokens kept | evicted mass by type: needle / haystack / template / sink / question | digits kept in needle heads | needle words kept |
|---|---|---|---|---|---|
| weighted oracle | 0.999 | — | — | — | — |
| SnapKV | 0.976 | 0 / 432 | 0.04 / 0.96 / 0.00 / 0.00 / 0.00 | 0.97 | 1.00 |
| LRU | 0.597 | 0 / 432 | 0.29 / 0.69 / 0.02 / 0.00 / 0.00 | **0.43** | 0.78 |
| H2O | 0.170 | 57 / 432 | 0.46 / 0.51 / 0.03 / 0.00 / 0.00 | **0.01** | 0.25 |
| StreamingLLM | 0.000 | 432 / 432 | 0.40 / 0.50 / 0.10 / 0.00 / 0.00 | 0.00 | 0.00 |

(5% budget: same picture; SnapKV 0.960, LRU 0.587, H2O 0.128, StreamingLLM 0.000.)
No heuristic loses sinks or the question: those are the easy 60% of mass (S3).

**Mechanism hypothesis.** Retrieval is carried by a sparse set of mid-to-late-layer heads that, at answer
time, put almost all their attention on the needle's ~18 tokens. During prefill, before the question arrives,
those tokens are *cold*: nothing in the haystack has reason to attend to a random number, so
(i) H2O's accumulated attention ranks them below early-position haystack tokens (cumulative scores have a
positional bias: an early token is visible to more queries); (ii) LRU's last-use refreshes the sentence's word
tokens (local, syntactic attention from neighbors) but not most digits, so it keeps a partial needle and the
model copies the wrong digits (live outputs like 6085932 for 6088743, 578932 for 5728454); (iii) StreamingLLM
is position-only. SnapKV succeeds because its observation window *is the question*: the last 32 prompt tokens
already query for the number, so their votes point at exactly the tokens the retrieval heads will read.
Prediction from the hypothesis: (a) moving the question to the start of the prompt (before the haystack) should
break SnapKV too; (b) H2O/LRU should recover if the question is repeated before compression; (c) "attention
mass on the answer tokens in retrieval heads" should predict live accuracy where total mass does not. None
of these were tested yet.

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
- **S4 (2026-11-06, Step 4b needle live) Mass does not predict retrieval.** Live needle exact-match (10 prompts):

  | budget | random | LRU | StreamingLLM | H2O | SnapKV | oracle replay |
  |---|---|---|---|---|---|---|
  | 5% | 0.0 | 0.0 | 0.0 | 0.0 | **1.0** | **1.0** |
  | 10% | 0.0 | 0.0 | 0.0 | 0.0 | **1.0** | **1.0** |
  | 25% | 0.0 | 0.3 | 0.2 | 0.1 | **1.0** | **1.0** |
  | 50% | 0.2 | 0.4 | 0.6 | 0.2 | **1.0** | **1.0** |

  LRU and H2O keep 0.84–0.88 of oracle mass at 10% yet score 0/10; SnapKV at 0.95 scores 10/10. Failures
  are mechanistic: the first digit is always right (it comes from prefill, D7), later digits drift or the
  answer stops early ('803' for 8056020). At 50%, StreamingLLM succeeds exactly on the needles at depth
  ≥ 0.5 (inside its recent window) and H2O exactly on depth ≥ 0.8 (inside its 25% recent half), so H2O's
  heavy-hitter half never captured the needle. Accumulated prefill attention favors early/sink tokens, not
  the needle. SnapKV works because its observation window is the question, which points at the needle.
  Consequence: total attention mass is the wrong yardstick for retrieval; the few percent of mass that
  separates SnapKV from H2O is the needle. Step 5 will measure mass on needle tokens directly.

- **S5 (2026-11-06, Step 4) Early layers tie; the policies separate only in middle and late layers.**
  At 10% budget, in layers 1, 5 and 6, LRU, StreamingLLM, H2O and SnapKV all sit within 0.01–0.04 of each
  other at 0.90–0.98 of oracle mass, on every task (summarization layer 6: 0.972–0.977). The spread is
  largest in layers 14, 16, 19, 22 and 27 (needle layer 14: StreamingLLM 0.50, H2O 0.61, LRU 0.72,
  SnapKV 0.94), which are the same layers that hold the needle heads (Step 5). Early-layer attention is
  almost entirely sinks plus recent tokens, so any policy that keeps both is near-optimal there. Belady is
  the outlier in the other direction: worst in layers 0–2 (0.43–0.65, where diffuse attention rarely
  crosses τ) but the *best* policy in summarization layers 14 and 22 (0.80/0.85 vs H2O 0.77/0.78), where
  attention is peaky. Implication: a per-layer budget (small for early layers, large for 14–27) should beat
  a uniform one. Not tested here.

## Next steps
- [x] Step 0 — setup and predictions
- [x] Step 1 — record attention traces (6.5 GB)
- [x] Step 2 — build the oracle (Belady + weighted; greedy ≈ LP optimum)
- [x] Step 3 — online baselines
- [ ] Step 4 — measure (offline mass + live accuracy), judge P1–P3
- [x] Step 5 — read failures, mechanism hypothesis (needle)

## Open questions
