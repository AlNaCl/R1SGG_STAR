#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/STAR}
export R1SGG_DATA_ROOT=${R1SGG_DATA_ROOT:-/root/autodl-tmp/STAR/r1sgg_data}
export STAR_RAW_ROOT=${STAR_RAW_ROOT:-/root/autodl-tmp/STAR/STAR}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/root/autodl-tmp/torch_ext}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/autodl-tmp/triton_cache}

CONFIG=${CONFIG:-configs/agentic_grpo.yaml}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/root/autodl-tmp/STAR/r1sgg_data/checkpoints/qwen25vl-7b-sft-star-close-20260507_182608Z}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/predictions" "${OUTPUT_ROOT}/eval_results" "${OUTPUT_ROOT}/tmp"

args=(
  --config "${CONFIG}"
  --model-path "${MODEL_NAME_OR_PATH}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

if [[ -n "${PROCESSOR_PATH:-}" ]]; then
  args+=(--processor-path "${PROCESSOR_PATH}")
fi
if [[ -n "${SPLIT:-}" ]]; then
  args+=(--split "${SPLIT}")
fi
if [[ -n "${SAMPLE_INDEX:-}" ]]; then
  args+=(--sample-index "${SAMPLE_INDEX}")
fi

python -m src.rl.generation_smoke "${args[@]}"
