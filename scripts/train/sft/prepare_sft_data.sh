#!/bin/bash
# IG-GRPO: SFT 数据准备脚本
# 将采集的原始数据复制到训练目录

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]")/../.." && pwd)"
cd "$PROJECT_ROOT"

SOURCE_DIR="experiments/sft_retail/raw"
TARGET_DIR="experiments/sft_retail"
TARGET_FILE="$TARGET_DIR/train.jsonl"

echo "======================================"
echo "IG-GRPO SFT Data Preparation"
echo "======================================"

# 检查源数据是否存在
if [ ! -f "$SOURCE_DIR/train.jsonl" ]; then
    echo "错误: 采集数据不存在！"
    echo "请先运行数据采集:"
    echo "  bash scripts/train/sft/collect_retail_data.sh"
    exit 1
fi

# 统计数据量
NUM_SAMPLES=$(wc -l < "$SOURCE_DIR/train.jsonl")
echo "源数据: $SOURCE_DIR/train.jsonl"
echo "样本数: $NUM_SAMPLES"

# 复制数据
mkdir -p "$TARGET_DIR"
cp "$SOURCE_DIR/train.jsonl" "$TARGET_FILE"

echo ""
echo "数据准备完成!"
echo "输出: $TARGET_FILE"
echo ""
echo "可以开始 SFT 训练:"
echo "  bash scripts/train/sft/run_sft_retail.sh"
