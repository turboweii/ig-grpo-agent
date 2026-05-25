#!/bin/bash
# Experiment 1: IG-Fixed 训练脚本
# 固定 IG 权重，验证 IG 信号有效性

set -e

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Hydra 配置
CONFIG_FILE=configs/train/grpo/ig_fixed.yaml

# 训练参数
TOTAL_STEPS=300

# 输出目录
OUTPUT_DIR=outputs/ig_fixed_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

# 日志文件
LOG_FILE="$OUTPUT_DIR/train.log"

echo "======================================"
echo "IG-Fixed Training (Experiment 1)"
echo "======================================"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Steps: $TOTAL_STEPS"
echo "IG Weight: 0.5 (fixed, no curriculum)"
echo "======================================"

# 启动训练 (使用 Hydra 配置)
python -m verl.trainers.main \
    --config-path="$PROJECT_ROOT/configs/train/grpo" \
    --config-name=ig_fixed \
    trainer.total_training_steps=$TOTAL_STEPS \
    2>&1 | tee "$LOG_FILE"

echo "Training completed! Results saved to $OUTPUT_DIR"
