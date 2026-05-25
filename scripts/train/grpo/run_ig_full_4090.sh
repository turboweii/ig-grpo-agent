#!/bin/bash
# IG-GRPO 训练脚本 - 适配 4×4090 (24GB)
set -e

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1
export VLLM_USE_V1=1

# 显存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
# VLLM 优化
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_PREEMPTION=true

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================"
echo "IG-GRPO Training (4×4090)"
echo "======================================"
echo "GPUs: 4 × 4090 (24GB)"
echo "Config: configs/train/grpo/ig_full_4090.yaml"
echo "======================================"

# 创建输出目录
mkdir -p outputs/ig_full_4090

# 启动训练
python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_ROOT/configs" \
    --config-name=train/grpo/ig_full_4090 \
    2>&1 | tee outputs/ig_full_4090/training.log

echo "Training completed!"
