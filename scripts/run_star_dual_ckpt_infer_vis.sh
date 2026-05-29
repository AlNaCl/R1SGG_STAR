#!/usr/bin/env bash
# STAR 闭集：两个 checkpoint 依次推理并导出 preds.json + images/ 可视化。
# 用法：
#   bash scripts/run_star_dual_ckpt_infer_vis.sh
#   TOP_K_SMALLEST=50 bash scripts/run_star_dual_ckpt_infer_vis.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

DATASET="${DATASET:-/root/autodl-tmp/STAR/r1sgg_data/star_r1sgg_hf_closed}"
OUT_PARENT="${OUT_PARENT:-/root/autodl-tmp/STAR/r1sgg_data/eval_visualizations}"
SPLIT="${SPLIT:-val}"
TOP_K="${TOP_K_SMALLEST:-30}"

CKPT_2B="${CKPT_2B:-/root/autodl-tmp/STAR/r1sgg_data/checkpoints/qwen2vl-2b-star-close-sft_4gpu_20260505_1858}"
CKPT_7B="${CKPT_7B:-/root/autodl-tmp/STAR/r1sgg_data/checkpoints/qwen25vl-7b-sft-star-close-20260507_182608Z}"

TAG_2B="${TAG_2B:-qwen2vl-2b-star-close-sft_4gpu}"
TAG_7B="${TAG_7B:-qwen25vl-7b-sft-star-close}"

GPU="${CUDA_VISIBLE_DEVICES:-0}"

for need in "${DATASET}/dataset_dict.json" "${CKPT_2B}/config.json" "${CKPT_7B}/config.json"; do
  if [[ ! -f "${need}" ]]; then
    echo "缺失: ${need}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_PARENT}"

run_one() {
  local model_path="$1"
  local tag="$2"
  local out="${OUT_PARENT}/${tag}_k${TOP_K}_$(date +%Y%m%d_%H%M%S)"
  echo "======== ${tag} ========"
  echo "model_path=${model_path}"
  echo "output_dir=${out}"
  CUDA_VISIBLE_DEVICES="${GPU}" python scripts/star_closed_baseline_smallest_k.py \
    --model_path "${model_path}" \
    --dataset_path "${DATASET}" \
    --split "${SPLIT}" \
    --top_k_smallest "${TOP_K}" \
    --processor_path "${model_path}" \
    --output_dir "${out}"
  echo "完成: preds.json -> ${out}/preds.json"
  echo "可视化: ${out}/images/"
  echo "评测: python scripts/eval_star_closed_predictions.py --dataset_path ${DATASET} --split ${SPLIT} --preds_json ${out}/preds.json"
  echo ""
}

run_one "${CKPT_2B}" "${TAG_2B}"
run_one "${CKPT_7B}" "${TAG_7B}"

echo "全部结束。两次输出目录均在: ${OUT_PARENT}"
