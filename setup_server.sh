#!/bin/bash
# IG-GRPO Agent 环境搭建脚本
# 用法: bash setup_server.sh

set -e

ENV_NAME="ig-grpo"
PYTHON_VERSION="3.10"

echo "=== [1/6] 创建 conda 环境: $ENV_NAME ==="
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 $ENV_NAME 已存在，跳过创建步骤..."
else
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi
source activate $ENV_NAME || conda activate $ENV_NAME

echo "=== [2/6] 安装 PyTorch (CUDA 12.6) ==="
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

echo "=== [3/6] 安装 vLLM ==="
pip install vllm==0.9.2

echo "=== [4/6] 安装 flash-attn (需编译，可能需要几分钟) ==="
pip install flash-attn --no-build-isolation

echo "=== [5/6] 安装项目依赖 ==="
pip install -r requirements.txt

echo "=== [6/6] 安装 τ-bench ==="
# 检查 tau-bench 是否已存在
if [ -d "../tau-bench" ]; then
    echo "tau-bench 已存在，跳过克隆..."
else
    echo "克隆 tau-bench..."
    cd ..
    git clone https://github.com/sierra-research/tau-bench
    cd tau-bench
    pip install -e .
    cd ../ig-grpo-agent
fi

echo ""
echo "=== 搭建完成 ==="
echo ""
echo "激活环境: conda activate $ENV_NAME"
echo ""
echo "设置环境变量:"
echo "  export VLLM_USE_V1=1"
echo ""
echo "检查环境: python scripts/deploy/check_server_env.py"
