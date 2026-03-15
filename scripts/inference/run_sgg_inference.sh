#!/bin/bash

# 使用 HuggingFace 镜像（国内网络可避免 connection unreachable）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export GPUS_PER_NODE=2


# DATASET=$1        # 数据集名称
# MODEL_NAME=$2     # 模型名称
# OUTPUT_DIR=$3     # 输出目录
# USE_CATS=$4       # 是否使用预定义类别（true/false）
# PROMPT_TYPE=$5    # 是否使用系统提示（true/false）

# 从命令行参数读取变量，如果没有提供则使用默认值
DATASET=$1                # 数据集名称
MODEL_NAME=$2             # 模型名称
OUTPUT_DIR=$3             # 输出目录
USE_CATS=${4:-false}      # 是否使用预定义类别，默认为 false
PROMPT_TYPE=${5:-false}   # 是否使用系统提示，默认为 false
BATCH_SIZE=${6:-1}        # 批大小，默认为 1
MAX_MODEL_LEN=${7:-2048}  # 模型最大长度，默认为 2048
MASTER_PORT=${8:-29601}

# BATCH_SIZE=${6:-2}

# echo "MODEL_NAME: $MODEL_NAME, OUTPUT_DIR: $OUTPUT_DIR"
# echo "USE_CATS: $USE_CATS, PROMPT_TYPE: $PROMPT_TYPE"

# ARGS="--dataset $DATASET --model $MODEL_NAME --output_dir $OUTPUT_DIR --max_model_len 2048 --batch_size $BATCH_SIZE"


echo "MODEL_NAME: $MODEL_NAME"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "USE_CATS: $USE_CATS"
echo "PROMPT_TYPE: $PROMPT_TYPE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "MASTER_PORT: $MASTER_PORT"

ARGS="--dataset $DATASET --model $MODEL_NAME --output_dir $OUTPUT_DIR --max_model_len $MAX_MODEL_LEN --batch_size $BATCH_SIZE"



if [ "$PROMPT_TYPE" == "true" ]; then
  ARGS="$ARGS --use_think_system_prompt"
fi

if [ "$USE_CATS" == "true" ]; then
  ARGS="$ARGS --use_predefined_cats"
fi

echo "ARGS:$ARGS"

torchrun --nnodes 1 \
  --nproc_per_node $GPUS_PER_NODE \
  --node_rank 0 \
  --master_port $MASTER_PORT \
  src/sgg_inference_vllm.py -- $ARGS
