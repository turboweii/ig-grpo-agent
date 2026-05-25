#!/bin/bash
# IG-GRPO 环境一键搭建脚本

set -e

echo "======================================"
echo "IG-GRPO Environment Setup"
echo "======================================"

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# 创建 conda 环境
CONDA_ENV_NAME="iggrpo"

echo "Creating conda environment: $CONDA_ENV_NAME"
conda create -n $CONDA_ENV_NAME python=3.10 -y

# 激活环境
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

# 安装 PyTorch
echo "Installing PyTorch 2.7.0 with CUDA 12.6..."
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install torchvision

# 安装依赖
echo "Installing dependencies..."
pip install -r requirements.txt

# 安装 τ-bench（开发版本）
echo "Installing τ-bench..."
if [ -d "../tau-bench" ]; then
    echo "Using local τ-bench..."
    cd ../tau-bench
    pip install -e .
    cd ../ig-grpo-agent
else
    echo "Cloning τ-bench..."
    cd ..
    git clone https://github.com/sierra-research/tau-bench.git
    cd tau-bench
    pip install -e .
    cd ../ig-grpo-agent
fi

# 安装 veRL（开发版本）
echo "Installing veRL..."
if [ -d "../verl" ]; then
    echo "Using local veRL..."
    cd ../verl
    pip install -e .
    cd ../ig-grpo-agent
else
    echo "Cloning veRL..."
    cd ..
    git clone https://github.com/volcengine/verl.git
    cd verl
    pip install -e .
    cd ../ig-grpo-agent
fi

# 创建必要的目录
echo "Creating output directories..."
mkdir -p outputs/vanilla
mkdir -p outputs/ig_fixed
mkdir -p outputs/ig_combo
mkdir -p outputs/ig_full
mkdir -p logs
mkdir -p checkpoints

# 设置环境变量
echo "Setting environment variables..."
cat > .env << EOF
# IG-GRPO Environment Variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
EOF

echo "======================================"
echo "Setup completed!"
echo "======================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $CONDA_ENV_NAME"
echo ""
echo "To start training:"
echo "  1. Start vLLM servers:"
echo "     bash scripts/vllm_server/7b.sh &"
echo "     bash scripts/vllm_server/72b.sh &"
echo ""
echo "  2. Run training:"
echo "     bash scripts/train/grpo/run_exp3_ig_full.sh"
echo ""
echo "======================================"
