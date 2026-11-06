#!/bin/bash
# Live runs as short separate gpu_run jobs so other agents can take turns between batches.
cd "$(dirname "$0")/.."
run() { PYTHONPATH=. ~/anime-pipeline/tools/gpu_run --free --need 10000 .venv/bin/python scripts/run_live.py "$@" >> data/live_rest.log 2>&1; }
run --task mdqa
ids=$(.venv/bin/python -c "import json;print(' '.join(json.loads(l)['id'] for l in open('data/tasks.jsonl') if json.loads(l)['task']=='summ'))")
set -- $ids
while [ $# -gt 0 ]; do
  batch=$(echo "$1 $2" | xargs | tr ' ' ','); shift 2 2>/dev/null || shift $#
  run --only "$batch"
done
echo ALLDONE >> data/live_rest.log
