#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg

export OMP_NUM_THREADS=1
export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/STAR}
export R1SGG_DATA_ROOT=${R1SGG_DATA_ROOT:-/root/autodl-tmp/STAR/r1sgg_data}
export STAR_RAW_ROOT=${STAR_RAW_ROOT:-/root/autodl-tmp/STAR/STAR}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/checkpoints/sft_text_before_vision" "${OUTPUT_ROOT}/tmp"

echo "Phase 8 does not launch cold-start SFT automatically. Reuse existing scripts/sft_local or scripts/sft entrypoints after selecting the base model and text-before-vision data."
echo "DATA_ROOT=${DATA_ROOT}"
echo "R1SGG_DATA_ROOT=${R1SGG_DATA_ROOT}"
echo "STAR_RAW_ROOT=${STAR_RAW_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "SFT_CHECKPOINT_DIR=${OUTPUT_ROOT}/checkpoints/sft_text_before_vision"
