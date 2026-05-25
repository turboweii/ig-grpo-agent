#!/bin/bash
# IG-Full 训练脚本 - 使用 veRL 框架
set -e

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export VLLM_USE_V1=1

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================"
echo "IG-GRPO Training"
echo "======================================"
echo "Project Root: $PROJECT_ROOT"
echo "Config: configs/train/grpo/ig_full.yaml"
echo "======================================"

# 启动训练
python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_ROOT/configs" \
    --config-name=train/grpo/ig_full \
    2>&1 | tee outputs/ig_full/training.log

echo "Training completed!"
