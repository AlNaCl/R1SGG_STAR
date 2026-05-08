#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
LOW_MEM_MODE="${LOW_MEM_MODE:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
# none | deepspeed_zero2 | deepspeed_zero3 | fsdp  (默认：4 卡 ZeRO-3)
DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-deepspeed_zero3}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
FSDP_CONFIG="${FSDP_CONFIG:-}"

PROJECT_DIR="/root/R1SGG_STAR"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="r1sgg"
LOG_DIR="/root/autodl-tmp/STAR/r1sgg_data/logs"
LOG_FILE="${LOG_DIR}/qwen2vl-2b-star-close-sft.log"
PID_FILE="${LOG_DIR}/qwen2vl-2b-star-close-sft.pid"

mkdir -p "${LOG_DIR}"

is_running() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

find_existing_torchrun_pid() {
  pgrep -f "torchrun .*src/sft_sgg.py .*qwen2vl-2b-star-close-sft" | head -n 1 || true
}

append_run_header() {
  local mode_str
  mode_str="full"
  if [[ "${LOW_MEM_MODE}" == "1" ]]; then
    mode_str="low_mem"
  fi

  {
    echo
    echo "==================== RUN START $(date '+%F %T') ===================="
    echo "mode=${mode_str} attn_implementation=${ATTN_IMPLEMENTATION} gradient_checkpointing=true"
    echo "distributed_strategy=${DISTRIBUTED_STRATEGY} gpus_per_node=${GPUS_PER_NODE}"
    if [[ -n "${HTTPS_PROXY:-}" ]] || [[ -n "${HTTP_PROXY:-}" ]]; then
      echo "proxy=https/http (set, values omitted)"
    else
      echo "proxy=(unset)"
    fi
    echo "max_objects=${MAX_OBJECTS} max_relationships=${MAX_RELATIONSHIPS} max_token_length=${MAX_TOKEN_LENGTH}"
    echo "max_pixels=${MAX_PIXELS} adaptive_risk_max_pixels=${ADAPTIVE_RISK_MAX_PIXELS} adaptive_tile_max_pixels=${ADAPTIVE_TILE_MAX_PIXELS}"
    echo "===================================================================="
  } >> "${LOG_FILE}"
}

configure_attention_impl() {
  if [[ "${ATTN_IMPLEMENTATION}" == "auto" ]]; then
    # flash_attn may be installed but built without kernels for this GPU arch (e.g. Blackwell sm_120).
    if python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("flash_attn") is None:
    sys.exit(1)
import torch

if not torch.cuda.is_available():
    sys.exit(1)
try:
    from flash_attn import flash_attn_func

    q = torch.randn(1, 16, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 16, 4, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 16, 4, 64, device="cuda", dtype=torch.bfloat16)
    flash_attn_func(q, k, v)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    then
      ATTN_IMPLEMENTATION="flash_attention_2"
    else
      ATTN_IMPLEMENTATION="sdpa"
    fi
  fi
}

configure_limits() {
  # Default: keep original experiment settings for result consistency.
  MAX_OBJECTS=160
  MAX_RELATIONSHIPS=600
  MAX_TOKEN_LENGTH=3072
  MAX_PIXELS=0
  ADAPTIVE_RISK_MAX_PIXELS=0
  ADAPTIVE_TILE_MAX_PIXELS=0

  # Optional low-memory mode for quick smoke tests only.
  if [[ "${LOW_MEM_MODE}" == "1" ]]; then
    MAX_OBJECTS=120
    MAX_RELATIONSHIPS=300
    MAX_TOKEN_LENGTH=2048
    MAX_PIXELS=524288
    ADAPTIVE_RISK_MAX_PIXELS=524288
    ADAPTIVE_TILE_MAX_PIXELS=524288
  fi
}

start_job() {
  if is_running; then
    echo "Training is already running with PID $(cat "${PID_FILE}")."
    return 0
  fi

  local ext_pid
  ext_pid="$(find_existing_torchrun_pid)"
  if [[ -n "${ext_pid}" ]]; then
    echo "Detected existing training process (PID ${ext_pid})."
    echo "Skip starting a duplicate job."
    return 0
  fi

  if ! [[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid GPUS_PER_NODE='${GPUS_PER_NODE}' (expected positive integer)."
    exit 1
  fi

  if [[ "${DISTRIBUTED_STRATEGY}" == "fsdp" ]] && [[ "${GPUS_PER_NODE}" -lt 2 ]]; then
    echo "DISTRIBUTED_STRATEGY=fsdp requires GPUS_PER_NODE>=2 (FSDP needs multiple ranks)."
    exit 1
  fi

  cd "${PROJECT_DIR}"
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"

  export PYTHONPATH="/root/R1SGG_STAR:${PYTHONPATH:-}"
  export HF_ENDPOINT="https://hf-mirror.com"
  # AutoDL built-in academic proxy (github / huggingface, etc.). Disable with USE_NETWORK_TURBO=0.
  if [[ "${USE_NETWORK_TURBO:-1}" != "0" ]] && [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
  fi
  # Pass through user proxy for pip / hub / GitHub installs in child tools.
  [[ -n "${HTTP_PROXY:-}" ]] && export HTTP_PROXY
  [[ -n "${HTTPS_PROXY:-}" ]] && export HTTPS_PROXY
  [[ -n "${ALL_PROXY:-}" ]] && export ALL_PROXY
  [[ -n "${NO_PROXY:-}" ]] && export NO_PROXY
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export WANDB_PROJECT=RL4SGG
  export DATASET_NAME="/root/autodl-tmp/STAR/r1sgg_data/star_r1sgg_hf_closed"
  export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2-VL-2B-Instruct}"
  export OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/STAR/r1sgg_data/checkpoints/qwen2vl-2b-star-close-sft}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  mkdir -p "${OUTPUT_DIR}"
  configure_limits
  configure_attention_impl
  append_run_header

  local -a dist_args=()
  case "${DISTRIBUTED_STRATEGY}" in
    none) ;;
    deepspeed_zero2)
      if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
        dist_args+=(--deepspeed "${DEEPSPEED_CONFIG}")
      else
        dist_args+=(--deepspeed "${PROJECT_DIR}/local_scripts/zero2.json")
      fi
      ;;
    deepspeed_zero3)
      if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
        dist_args+=(--deepspeed "${DEEPSPEED_CONFIG}")
      else
        dist_args+=(--deepspeed "${PROJECT_DIR}/local_scripts/zero3.json")
      fi
      ;;
    fsdp)
      if [[ -n "${FSDP_CONFIG}" ]]; then
        dist_args+=(--fsdp full_shard --fsdp_config "${FSDP_CONFIG}")
      else
        dist_args+=(--fsdp full_shard --fsdp_config "${PROJECT_DIR}/local_scripts/fsdp_qwen2vl.json")
      fi
      ;;
    *)
      echo "Unknown DISTRIBUTED_STRATEGY='${DISTRIBUTED_STRATEGY}' (use none|deepspeed_zero2|deepspeed_zero3|fsdp)."
      exit 1
      ;;
  esac

  if [[ "${DISTRIBUTED_STRATEGY}" != none ]] && [[ "${GPUS_PER_NODE}" -eq 1 ]]; then
    echo "Note: ZeRO/FSDP sharding across GPUs needs GPUS_PER_NODE>=2; with 1 GPU, DeepSpeed flags have little effect on partitioned state."
  fi

  local -a train_args=(
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --dataset_name "${DATASET_NAME}"
    --learning_rate 1e-5
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 8
    --warmup_ratio 0.05
    --max_grad_norm 0.3
    --logging_steps 1
    --bf16 true
    --tf32 true
    --report_to none
    --attn_implementation "${ATTN_IMPLEMENTATION}"
    --gradient_checkpointing true
    --max_objects "${MAX_OBJECTS}"
    --max_relationships "${MAX_RELATIONSHIPS}"
    --max_token_length "${MAX_TOKEN_LENGTH}"
    --max_pixels "${MAX_PIXELS}"
    --adaptive_risk_max_pixels "${ADAPTIVE_RISK_MAX_PIXELS}"
    --adaptive_tile_max_pixels "${ADAPTIVE_TILE_MAX_PIXELS}"
    --random_subgraph_sampling true
    --adaptive_image_resize true
    --num_train_epochs 3
    --run_name qwen2vl-2b-star-close-sft
    --save_steps 100
    --save_only_model true
    --torch_dtype bfloat16
    --use_predefined_cats true
    --output_dir "${OUTPUT_DIR}"
    --seed 42
  )
  train_args+=("${dist_args[@]}")

  # DeepSpeed/FSDP need a distributed launcher (torchrun). Plain `python` leaves torch.distributed
  # uninitialized; DeepSpeed then falls back to MPI and requires mpi4py (often missing).
  if [[ "${GPUS_PER_NODE}" == "1" ]] && [[ "${DISTRIBUTED_STRATEGY}" == "none" ]]; then
    nohup python src/sft_sgg.py "${train_args[@]}" >> "${LOG_FILE}" 2>&1 &
  else
    nohup torchrun --nnodes 1 \
      --nproc_per_node "${GPUS_PER_NODE}" \
      --node_rank 0 \
      --master_port "${MASTER_PORT:-29501}" \
      src/sft_sgg.py "${train_args[@]}" >> "${LOG_FILE}" 2>&1 &
  fi

  echo $! > "${PID_FILE}"
  echo "Started training in daemon mode."
  echo "PID: $(cat "${PID_FILE}")"
  echo "Log: ${LOG_FILE}"
}

stop_job() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    kill "${pid}"
    rm -f "${PID_FILE}"
    echo "Stopped training process ${pid}."
    return 0
  fi

  local ext_pid
  ext_pid="$(find_existing_torchrun_pid)"
  if [[ -n "${ext_pid}" ]]; then
    kill "${ext_pid}"
    echo "Stopped external training process ${ext_pid}."
    return 0
  fi

  echo "No active daemon process found."
  rm -f "${PID_FILE}"
}

status_job() {
  if is_running; then
    echo "RUNNING (PID: $(cat "${PID_FILE}"))"
    echo "Log: ${LOG_FILE}"
    return 0
  fi

  local ext_pid
  ext_pid="$(find_existing_torchrun_pid)"
  if [[ -n "${ext_pid}" ]]; then
    echo "RUNNING (external PID: ${ext_pid}, not managed by pid file)"
    echo "Log: ${LOG_FILE}"
    return 0
  fi

  echo "NOT RUNNING"
}

logs_job() {
  touch "${LOG_FILE}"
  tail -n 80 "${LOG_FILE}"
}

health_job() {
  status_job
  echo "--- recent error scan ---"
  if [[ -f "${LOG_FILE}" ]]; then
    if command -v rg >/dev/null 2>&1; then
      tail -n 400 "${LOG_FILE}" | rg -n "OutOfMemoryError|RuntimeError: CUDA error|Traceback|ImportError" | tail -n 10 || true
    else
      tail -n 400 "${LOG_FILE}" | grep -nE "OutOfMemoryError|RuntimeError: CUDA error|Traceback|ImportError" | tail -n 10 || true
    fi
  else
    echo "No log file found."
  fi
}

case "${ACTION}" in
  start)
    start_job
    ;;
  stop)
    stop_job
    ;;
  restart)
    stop_job
    start_job
    ;;
  status)
    status_job
    ;;
  logs)
    logs_job
    ;;
  health)
    health_job
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs|health}"
    exit 1
    ;;
esac
