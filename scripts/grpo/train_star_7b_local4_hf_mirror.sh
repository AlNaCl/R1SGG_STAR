#!/usr/bin/env bash
# STAR 闭集 + Qwen2-VL-7B，单机 4 卡；使用国内 HF 镜像（需能访问 hf-mirror.com）
#
# 模型路径：
#   - 默认：从 Hub 拉 Qwen/Qwen2-VL-7B-Instruct（易超时，建议先单独 huggingface-cli download）
#   - 推荐：本地已下载目录，例如：
#       export MODEL_NAME_OR_PATH=/root/shared-nvme/models/Qwen2-VL-7B-Instruct
#       bash scripts/grpo/train_star_7b_local4_hf_mirror.sh
set -euo pipefail

cd /root/work/R1SGG_STAR

# 国内镜像：见 https://hf-mirror.com/（本地模型仍保留，其它组件若访问 Hub 时可能用到）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2-VL-7B-Instruct}"

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export DEBUG_MODE="${DEBUG_MODE:-True}"
export WANDB_PROJECT="${WANDB_PROJECT:-RL4SGG}"

export OUTPUT_DIR="${OUTPUT_DIR:-/root/shared-nvme/models/qwen2vl-7b-close-grpo-star-local4x4090}"
mkdir -p "$OUTPUT_DIR"
export LOG_PATH="${OUTPUT_DIR}/debug.log"

MAX_PIXELS=$((512 * 28 * 28))
GROUP_SIZE=8

TORCHRUN="${TORCHRUN:-/root/.conda/envs/r1sgg/bin/torchrun}"

exec "$TORCHRUN" --nnodes 1 --nproc_per_node 4 \
  --rdzv_backend c10d \
  --rdzv_endpoint localhost:29500 \
  open_r1/grpo.py \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset_name "/root/shared-nvme/datasets/STAR/star_r1sgg_hf" \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --custom_per_device_train_batch_size 4 \
  --deepspeed ./local_scripts/zero2.json \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-7 \
  --use_predefined_cats true \
  --logging_steps 1 \
  --use_vllm true \
  --use_local_vllm true \
  --bf16 true \
  --tf32 true \
  --report_to wandb \
  --gradient_checkpointing true \
  --max_pixels ${MAX_PIXELS} \
  --temperature 1.0 \
  --top_p 0.9 \
  --top_k 50 \
  --num_train_epochs 1 \
  --run_name "qwen2vl-7b-close-grpo-star-local4" \
  --save_steps 100 \
  --num_generations ${GROUP_SIZE} \
  --num_iterations 1 \
  --beta 0.0 \
  --vllm_max_model_len 4096 \
  --vllm_gpu_memory_utilization 0.25 \
  --save_only_model true \
  --seed 42
