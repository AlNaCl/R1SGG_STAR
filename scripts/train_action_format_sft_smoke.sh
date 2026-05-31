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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

CONFIG=${CONFIG:-configs/agentic_grpo.yaml}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/root/autodl-tmp/STAR/r1sgg_data/checkpoints/qwen25vl-7b-sft-star-close-20260507_182608Z}
DATASET_NAME=${DATASET_NAME:-}
MAX_SAMPLES=${MAX_SAMPLES:-32}
MAX_OBJECTS=${MAX_OBJECTS:-16}
MAX_RELATIONSHIPS=${MAX_RELATIONSHIPS:-32}
MAX_STEPS=${MAX_STEPS:-1}
MAX_PIXELS=${MAX_PIXELS:-802816}
MAX_TOKEN_LENGTH=${MAX_TOKEN_LENGTH:-2048}
OUTPUT_DIR=${OUTPUT_DIR:-}
RUN_NAME=${RUN_NAME:-agentic_action_format_sft_smoke}
REPORT_TO=${REPORT_TO:-none}
ATTN_IMPL=${ATTN_IMPL:-sdpa}
USE_PEFT=${USE_PEFT:-true}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

if [[ -z "${DATASET_NAME}" ]]; then
  build_json=$(MAX_SAMPLES="${MAX_SAMPLES}" MAX_OBJECTS="${MAX_OBJECTS}" MAX_RELATIONSHIPS="${MAX_RELATIONSHIPS}" PROMPT_MODE=dataset bash scripts/build_action_sft_dataset.sh)
  echo "${build_json}"
  DATASET_NAME=$(python -c 'import json,sys; print(json.load(sys.stdin)["hf_dataset_path"])' <<<"${build_json}")
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  stamp=$(date -u +%Y%m%d_%H%M%SZ)
  OUTPUT_DIR="${OUTPUT_ROOT}/checkpoints/sft_action_format_smoke_${stamp}"
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[ERROR] OUTPUT_DIR already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

PEFT_ARGS=()
if [[ "${USE_PEFT}" == "true" ]]; then
  IFS="," read -r -a LORA_TARGET_ARRAY <<< "${LORA_TARGET_MODULES}"
  PEFT_ARGS+=(--use_peft true --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" --lora_target_modules "${LORA_TARGET_ARRAY[@]}")
fi

python src/sft_sgg.py   --model_name_or_path "${MODEL_NAME_OR_PATH}"   --dataset_name "${DATASET_NAME}"   --use_dataset_messages true   --learning_rate 5e-6   --per_device_train_batch_size 1   --gradient_accumulation_steps 1   --warmup_ratio 0.0   --max_grad_norm 0.3   --logging_steps 1   --dataloader_num_workers 0   --dataloader_pin_memory false   --bf16 true   --tf32 true   --report_to "${REPORT_TO}"   --attn_implementation "${ATTN_IMPL}"   --max_pixels "${MAX_PIXELS}"   --max_token_length "${MAX_TOKEN_LENGTH}"   --adaptive_image_resize false   --adaptive_tile_risky_sample false   --remove_unused_columns false   --num_train_epochs 1   --max_steps "${MAX_STEPS}"   --save_steps "${MAX_STEPS}"   --save_only_model true   --torch_dtype bfloat16   "${PEFT_ARGS[@]}"   --run_name "${RUN_NAME}"   --output_dir "${OUTPUT_DIR}"   --seed 42
