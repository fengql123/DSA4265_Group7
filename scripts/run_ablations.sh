#!/usr/bin/env bash
# Run Phase 7 ablations: 4 × 60 pipeline runs, one per disabled sub-agent.
# Each disabled variant writes to its own parquet so resume logic is per-variant.
set -e

cd "$(dirname "$0")/.."

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate dsa4265

for agent in sentiment fundamental technical risk; do
    out="data/backtest/signals_no_${agent}.parquet"
    log="/tmp/ablate_no_${agent}.log"
    echo "=========================================="
    echo "ablation: no_${agent} -> ${out}"
    echo "=========================================="
    python scripts/run_backtest.py \
        --disabled-agents "${agent}" \
        --signals-out "${out}" \
        --chunk 5 2>&1 | tee "${log}"
    echo "done: no_${agent}"
done

echo "all ablations complete"
