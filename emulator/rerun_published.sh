#!/bin/bash
# Re-measure every number shown on the website / README with the current physics.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
mv runs/layer_bench.jsonl runs/layer_bench_previous.jsonl 2>/dev/null
.venv/bin/python emulate_hw.py --chips 0,1,2 > logs_rerun_emulate.txt 2>&1
.venv/bin/python context_1k.py > logs_rerun_context.txt 2>&1
.venv/bin/python layer_bench.py > logs_rerun_layer_bench.txt 2>&1
.venv/bin/python ablate_chain.py > logs_rerun_ablate.txt 2>&1
echo done > logs_rerun_done.txt
