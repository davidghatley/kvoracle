# kvoracle

Measures how close KV-cache eviction heuristics (random, LRU, StreamingLLM, H2O, SnapKV) get to an
offline Belady-style oracle, on Qwen2.5-7B-Instruct (4-bit). `RESEARCH.md` is the single source of
truth: read it before each step, write results back after each step.

## Rules
- GPU only through `~/anime-pipeline/tools/gpu_run --free ...`; keep batches under ~30 min.
- Never tune a policy on the evaluation prompts. Hyperparameters come from the papers' defaults.
- Surprises get written under "Surprises" in RESEARCH.md before continuing.
- Never delete data; move aside. Big outputs live in `data/` -> `/mnt/work/kvoracle` (gitignored).
- Commit after each step, staging explicit paths.

## Layout
- `kvo/` library code (model patching, trace recording, policies, oracle, simulation)
- `scripts/` entry points, `docs/figs/` charts referenced from RESEARCH.md
- `data/` traces, datasets, live-run outputs (not in git)
