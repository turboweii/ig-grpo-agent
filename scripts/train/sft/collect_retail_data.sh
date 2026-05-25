#!/bin/bash
# IG-GRPO: SFT 数据采集脚本 (Retail 域)
# 适配 4×4090 硬件配置

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================"
echo "IG-GRPO SFT Data Collection (Retail)"
echo "======================================"
echo "Hardware: 4 × 4090 (24GB)"
echo "Domain:   Retail (40 tools)"
echo "======================================"

# 检查 vLLM 服务是否运行
check_service() {
    local url=$1
    local name=$2
    if curl -s "$url" > /dev/null 2>&1; then
        echo "✓ $name 服务运行中 ($url)"
        return 0
    else
        echo "✗ $name 服务未运行 ($url)"
        return 1
    fi
}

# 检查必要的服务
SERVICES_OK=true
check_service "http://localhost:8000/health" "7B 教师策略" || SERVICES_OK=false
check_service "http://localhost:8001/health" "72B 用户模拟器" || SERVICES_OK=false

if [ "$SERVICES_OK" = false ]; then
    echo ""
    echo "错误: vLLM 服务未启动！"
    echo "请先启动服务:"
    echo "  bash scripts/vllm_server/7b.sh"
    echo "  bash scripts/vllm_server/72b.sh"
    exit 1
fi

echo ""
echo "开始采集 SFT 数据..."
echo "配置: configs/train/sft/sft_collect_retail.yaml"
echo ""

# 创建输出目录
mkdir -p experiments/sft_retail/raw

# 运行数据采集
python scripts/train/sft/collect_sft_data.py \
    --config configs/train/sft/sft_collect_retail.yaml

echo ""
echo "======================================"
echo "数据采集完成!"
echo "======================================"
echo "输出位置:"
echo "  - 原始数据: experiments/sft_retail/raw/"
echo "  - 训练集:   experiments/sft_retail/raw/train.jsonl"
echo "  - 保留集:   experiments/sft_retail/raw/holdout_train.jsonl"
echo "  - 统计:     experiments/sft_retail/raw/summary.json"
echo ""

# 自动准备训练数据
echo "准备 SFT 训练数据..."
bash scripts/train/sft/prepare_sft_data.sh

echo ""
echo "全部完成! 现在可以开始 SFT 训练:"
echo "  bash scripts/train/sft/run_sft_retail.sh"
