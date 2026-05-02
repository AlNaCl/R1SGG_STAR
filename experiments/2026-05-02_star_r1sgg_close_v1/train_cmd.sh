#!/usr/bin/env bash
set -euo pipefail

cd /root/R1SGG_STAR
source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg

export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-RL4SGG}"
export DATASET_NAME="${DATASET_NAME:-/root/autodl-tmp/STAR/r1sgg_data/star_r1sgg_hf_closed}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2-VL-2B-Instruct}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/STAR/r1sgg_data/checkpoints/qwen2vl-2b-star-close-sft}"

mkdir -p "${OUTPUT_DIR}"

torchrun --nnodes 1 \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --node_rank 0 \
  src/sft_sgg.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --warmup_ratio 0.05 \
  --max_grad_norm 0.3 \
  --logging_steps 1 \
  --bf16 true \
  --tf32 true \
  --report_to none \
  --attn_implementation sdpa \
  --max_objects 160 \
  --max_relationships 600 \
  --max_token_length 3072 \
  --random_subgraph_sampling true \
  --adaptive_image_resize true \
  --num_train_epochs 3 \
  --run_name qwen2vl-2b-star-close-sft \
  --save_steps 100 \
  --save_only_model true \
  --torch_dtype bfloat16 \
  --use_predefined_cats true \
  --output_dir "${OUTPUT_DIR}" \
  --seed 42
