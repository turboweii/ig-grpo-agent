#!/bin/bash
# IG-GRPO 完整训练流程
# Step 1: SFT (4×4090) → Step 2: GRPO (4×4090)
# 适配 4×4090 (24GB) 多卡训练

set -e

echo "======================================"
echo "IG-GRPO 完整训练流程"
echo "======================================"
echo "硬件: 4 × 4090 (24GB)"
echo "======================================"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Step 1: SFT 训练
echo ""
echo "=== Step 1: SFT 训练 (4卡) ==="
echo "配置: configs/train/sft/sft_retail_lora.yaml"
echo "等效 batch: 32 (4卡 × 2 × 4 grad_accum)"
echo ""

# 检查是否已有 SFT 模型
if [ -f "experiments/sft_lora_merged/config.json" ]; then
    echo "SFT 模型已存在，跳过 SFT 训练"
else
    echo "开始 SFT 训练..."
    bash scripts/train/sft/run_sft_retail.sh

    echo "合并 LoRA 权重..."
    bash scripts/train/sft/run_merge_lora.sh
fi

# Step 2: GRPO 训练
echo ""
echo "=== Step 2: GRPO 训练 (4卡) ==="
echo "配置: configs/train/grpo/ig_full_4090.yaml"
echo "2卡 vLLM rollout + 2卡 FSDP training"
echo ""

python -m verl.trainer.main_ppo \
    --config-path=configs \
    --config-name=train/grpo/ig_full_4090 \
    2>&1 | tee outputs/ig_full_4090/training.log

echo ""
echo "======================================"
echo "训练完成!"
echo "======================================"
echo "SFT 模型: experiments/sft_lora_merged"
echo "GRPO 检查点: experiments/ig_full_4090/checkpoints"
