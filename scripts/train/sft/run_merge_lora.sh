#!/bin/bash
# IG-GRPO: LoRA 权重合并脚本
# 将 SFT 训练的 LoRA 权重合并到基础模型

set -e

echo "======================================"
echo "合并 LoRA 权重"
echo "======================================"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# 默认参数
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
LORA_PATH=${LORA_PATH:-"experiments/sft_retail"}
OUTPUT_PATH=${OUTPUT_PATH:-"experiments/sft_lora_merged"}

echo "基础模型: $BASE_MODEL"
echo "LoRA 路径: $LORA_PATH"
echo "输出路径: $OUTPUT_PATH"
echo ""

python scripts/train/sft/merge_lora.py \
    --base_model "$BASE_MODEL" \
    --lora_path "$LORA_PATH" \
    --output_path "$OUTPUT_PATH"

echo ""
echo "合并完成! 模型保存在: $OUTPUT_PATH"
echo "可用于后续 GRPO 训练"
