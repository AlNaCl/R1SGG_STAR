#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg

export OMP_NUM_THREADS=1
export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/STAR}
export R1SGG_DATA_ROOT=${R1SGG_DATA_ROOT:-/root/autodl-tmp/STAR/r1sgg_data}
export STAR_RAW_ROOT=${STAR_RAW_ROOT:-/root/autodl-tmp/STAR/STAR}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}

CONFIG=${CONFIG:-configs/agentic_grpo.yaml}

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/predictions" "${OUTPUT_ROOT}/eval_results" "${OUTPUT_ROOT}/tmp"

python -m src.rl.grpo_trainer --config "${CONFIG}" --train-smoke
