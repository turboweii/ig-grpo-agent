#!/bin/bash
# IG-GRPO: SFT 训练脚本 (Retail 域)
# 使用 8×4090 多卡训练

set -e

echo "======================================"
echo "IG-GRPO SFT Training (Retail)"
echo "======================================"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用 8 卡
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 创建输出目录
mkdir -p experiments/sft_retail

echo "配置: configs/train/sft/sft_retail_lora_8gpu.yaml"
echo "输出: experiments/sft_retail"
echo "GPU: 8 × 4090 (24GB)"
echo "等效 batch size: 128 (8卡 × 8 × 2 grad_accum)"
echo ""

# 启动 SFT 训练 (使用 torchrun)
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    scripts/train/sft/sft_train.py \
    --config configs/train/sft/sft_retail_lora_8gpu.yaml

echo ""
echo "SFT 训练完成!"
echo "LoRA 检查点: experiments/sft_retail"
echo ""
echo "合并 LoRA 权重:"
echo "  bash scripts/train/sft/run_merge_lora.sh"
