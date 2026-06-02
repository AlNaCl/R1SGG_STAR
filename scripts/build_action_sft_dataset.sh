#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg

export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/STAR}
export R1SGG_DATA_ROOT=${R1SGG_DATA_ROOT:-/root/autodl-tmp/STAR/r1sgg_data}
export STAR_RAW_ROOT=${STAR_RAW_ROOT:-/root/autodl-tmp/STAR/STAR}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}

CONFIG=${CONFIG:-configs/agentic_grpo.yaml}
SPLIT=${SPLIT:-train}
MAX_SAMPLES=${MAX_SAMPLES:-256}
PROMPT_MODE=${PROMPT_MODE:-}
TARGET_MODE=${TARGET_MODE:-}
MAX_OBJECTS=${MAX_OBJECTS:-}
MAX_RELATIONSHIPS=${MAX_RELATIONSHIPS:-}

args=(
  --config "${CONFIG}"
  --split "${SPLIT}"
  --max-samples "${MAX_SAMPLES}"
)
if [[ -n "${PROMPT_MODE}" ]]; then
  args+=(--prompt-mode "${PROMPT_MODE}")
fi
if [[ -n "${TARGET_MODE}" ]]; then
  args+=(--target-mode "${TARGET_MODE}")
fi
if [[ -n "${MAX_OBJECTS}" ]]; then
  args+=(--max-objects "${MAX_OBJECTS}")
fi
if [[ -n "${MAX_RELATIONSHIPS}" ]]; then
  args+=(--max-relationships "${MAX_RELATIONSHIPS}")
fi

if [[ -n "${ACTION_SFT_OUTPUT_DIR:-}" ]]; then
  args+=(--output-dir "${ACTION_SFT_OUTPUT_DIR}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

python -m src.data.action_sft_dataset "${args[@]}"
