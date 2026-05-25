#!/bin/bash
# Experiment 2: IG-Combo 训练脚本
# 添加工具组合奖励和工具依赖图

set -e

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Hydra 配置
CONFIG_FILE=configs/train/grpo/ig_combo.yaml

# 训练参数
TOTAL_STEPS=300

# 输出目录
OUTPUT_DIR=outputs/ig_combo_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

# 日志文件
LOG_FILE="$OUTPUT_DIR/train.log"

echo "======================================"
echo "IG-Combo Training (Experiment 2)"
echo "======================================"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Steps: $TOTAL_STEPS"
echo "IG Weight: 0.5 (fixed)"
echo "Tool Combo Weight: 0.5 (increased)"
echo "======================================"

# 启动训练 (使用 Hydra 配置)
python -m verl.trainers.main \
    --config-path="$PROJECT_ROOT/configs/train/grpo" \
    --config-name=ig_combo \
    trainer.total_training_steps=$TOTAL_STEPS \
    2>&1 | tee "$LOG_FILE"

echo "Training completed! Results saved to $OUTPUT_DIR"
