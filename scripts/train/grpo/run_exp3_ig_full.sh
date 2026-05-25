#!/bin/bash
# Experiment 3: IG-Full 训练脚本
# 完整 IG-GRPO 方案：JIG + 课程学习 + 探索持续性

set -e

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Hydra 配置
CONFIG_FILE=configs/train/grpo/ig_full.yaml

# 训练参数
TOTAL_STEPS=300

# 输出目录
OUTPUT_DIR=outputs/ig_full_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

# 日志文件
LOG_FILE="$OUTPUT_DIR/train.log"

echo "======================================"
echo "IG-Full Training (Experiment 3)"
echo "======================================"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Steps: $TOTAL_STEPS"
echo "Features: JIG + Curriculum + Sustained + G-Norm"
echo "======================================"

# 启动训练 (使用 Hydra 配置)
python -m verl.trainers.main \
    --config-path="$PROJECT_ROOT/configs/train/grpo" \
    --config-name=ig_full \
    trainer.total_training_steps=$TOTAL_STEPS \
    2>&1 | tee "$LOG_FILE"

echo "Training completed! Results saved to $OUTPUT_DIR"
echo "Checkpoints saved at steps: 50, 100, 150, 200, 250, 300"

echo "Training completed! Results saved to $OUTPUT_DIR"
echo "Checkpoints saved at steps: 50, 100, 150, 200, 250, 300"
