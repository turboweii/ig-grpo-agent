#!/bin/bash
# Vanilla GRPO 基线训练脚本
# 不使用 IG 奖励

set -e

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Hydra 配置
CONFIG_FILE=configs/train/grpo/vanilla.yaml

# 训练参数
TOTAL_STEPS=300

# 输出目录
OUTPUT_DIR=outputs/vanilla_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

# 日志文件
LOG_FILE="$OUTPUT_DIR/train.log"

echo "======================================"
echo "Vanilla GRPO Training (No IG)"
echo "======================================"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Steps: $TOTAL_STEPS"
echo "======================================"

# 启动训练 (使用 Hydra 配置)
python -m verl.trainers.main \
    --config-path="$PROJECT_ROOT/configs/train/grpo" \
    --config-name=vanilla \
    trainer.total_training_steps=$TOTAL_STEPS \
    2>&1 | tee "$LOG_FILE"

echo "Training completed! Results saved to $OUTPUT_DIR"
