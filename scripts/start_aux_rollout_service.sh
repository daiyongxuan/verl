#!/bin/bash

# 启动辅助推理服务的脚本

export CUDA_DEVICE_MAX_CONNECTIONS=1  # For megatron communication/computation overlapping
export RAY_DEDUP_LOGS=0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CONFIG_PATH="${CONFIG_PATH:-verl/trainer/config}"
CONFIG_NAME="${CONFIG_NAME:-ppo_megatron_trainer}"
CONFIG_FILE="${CONFIG_FILE:-}"

# 服务端口
SERVICE_PORT="${SERVICE_PORT:-5555}"

# 日志文件
LOG_FILE="${LOG_FILE:-./log/aux_rollout_service.log}"

# 模型路径和其他配置覆盖（从环境变量或默认值）
MODEL_PATH="${MODEL_PATH:-/mnt/public/daiyongxuan/DeepSeek-R1-Distill-Qwen-1.5B}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"  # rollout tensor parallel size
ROLLOUT_PP="${ROLLOUT_PP:-2}"  # rollout pipeline parallel size
REF_TP="${REF_TP:-2}"          # reference tensor parallel size
REF_PP="${REF_PP:-2}"          # reference pipeline parallel size
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.4}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-4}"

# 确保日志目录存在
mkdir -p "$(dirname "$LOG_FILE")"

# 打印配置信息
echo "启动辅助推理服务..."
echo "配置系统: Hydra"
echo "配置路径: $CONFIG_PATH"
echo "配置名称: $CONFIG_NAME"
echo "服务端口: $SERVICE_PORT"
echo "模型路径: $MODEL_PATH"
echo "GPU设备: $CUDA_VISIBLE_DEVICES"
echo "GPU内存利用率: $GPU_MEMORY_UTIL"
echo "Rollout并行度: TP=$ROLLOUT_TP, PP=$ROLLOUT_PP"
echo "Reference并行度: TP=$REF_TP, PP=$REF_PP"
echo "日志文件: $LOG_FILE"
echo "环境变量:"
echo "  CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS"
echo "  RAY_DEDUP_LOGS=$RAY_DEDUP_LOGS"

# 构建启动参数
AUX_ROLLOUT_ARGS=""

AUX_ROLLOUT_ARGS="$AUX_ROLLOUT_ARGS --port $SERVICE_PORT"

# 添加配置覆盖参数（模拟训练脚本的参数）
HYDRA_OVERRIDES=(
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$ROLLOUT_PP"
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$ROLLOUT_TP"
    "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$REF_PP"
    "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$REF_TP"
    "aux_rollout.n_gpus_per_node=4"
    "aux_rollout.nnodes=1"
    "trainer.n_gpus_per_node=4"
    "trainer.nnodes=1"
)

# 如果使用Hydra配置系统，添加覆盖参数
if [ -z "$CONFIG_FILE" ]; then
    for override in "${HYDRA_OVERRIDES[@]}"; do
        AUX_ROLLOUT_ARGS="$AUX_ROLLOUT_ARGS $override"
    done
fi

AUX_ROLLOUT_ARGS="$AUX_ROLLOUT_ARGS $@"

echo "最终启动命令参数: $AUX_ROLLOUT_ARGS"

python -m verl.trainer.ppo.aux_rollout_service $AUX_ROLLOUT_ARGS 2>&1 | tee "$LOG_FILE" 