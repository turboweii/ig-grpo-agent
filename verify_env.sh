#!/bin/bash
# IG-GRPO: 环境验证脚本

set -e

echo "======================================"
echo "IG-GRPO Environment Check"
echo "======================================"

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  警告: 未检测到虚拟环境"
    echo "建议先激活: source venv/bin/activate"
    echo ""
fi

# 检查 GPU
echo -n "GPU 数量: "
N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "$N_GPUS"

if [ "$N_GPUS" -lt 4 ]; then
    echo "⚠️  警告: 检测到 $N_GPUS 个 GPU，配置需要 4 个"
fi

echo ""
echo "GPU 详情:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# 检查 Python
echo "Python 版本:"
python --version
echo ""

# 检查 CUDA
echo "CUDA 支持:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA 可用: {torch.cuda.is_available()}')
print(f'  CUDA 版本: {torch.version.cuda}')
print(f'  GPU 数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.get_device_name(i)}')
"
echo ""

# 检查关键包
echo "Python 包检查:"
packages=(
    "torch:PyTorch"
    "vllm:vLLM"
    "transformers:Transformers"
    "peft:PEFT"
    "pybloom_live:PyBloom"
    "omegaconf:OmegaConf"
    "hydra:Hydra"
)

for pkg_info in "${packages[@]}"; do
    IFS=':' read -r module name <<< "$pkg_info"
    if python -c "import $module" 2>/dev/null; then
        version=$(python -c "import $module; print(getattr($module, '__version__', 'OK'))" 2>/dev/null)
        echo "  ✓ $name ($version)"
    else
        echo "  ✗ $name (未安装)"
    fi
done

# 检查 tau-bench 和 verl
echo ""
echo "子模块检查:"
if python -c "import tau_bench" 2>/dev/null; then
    echo "  ✓ tau-bench"
else
    echo "  ✗ tau-bench (未安装)"
fi

if python -c "import verl" 2>/dev/null; then
    echo "  ✓ verl"
else
    echo "  ✗ verl (未安装)"
fi

# 检查模型文件
echo ""
echo "模型文件检查:"
MODEL_DIR="${HF_HOME:-/root/models}"
if [ -d "$MODEL_DIR/Qwen2.5-7B-Instruct" ]; then
    size=$(du -sh "$MODEL_DIR/Qwen2.5-7B-Instruct" 2>/dev/null | cut -f1)
    echo "  ✓ Qwen2.5-7B-Instruct ($size)"
else
    echo "  ✗ Qwen2.5-7B-Instruct (不存在)"
    echo "    下载命令: huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $MODEL_DIR/Qwen2.5-7B-Instruct"
fi

if [ -d "$MODEL_DIR/Qwen2.5-72B-Instruct-AWQ" ]; then
    size=$(du -sh "$MODEL_DIR/Qwen2.5-72B-Instruct-AWQ" 2>/dev/null | cut -f1)
    echo "  ✓ Qwen2.5-72B-Instruct-AWQ ($size)"
else
    echo "  ✗ Qwen2.5-72B-Instruct-AWQ (不存在)"
    echo "    下载命令: huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ --local-dir $MODEL_DIR/Qwen2.5-72B-Instruct-AWQ"
fi

# 检查配置文件
echo ""
echo "配置文件检查:"
configs=(
    "configs/train/sft/sft_retail_lora.yaml"
    "configs/train/sft/sft_collect_retail.yaml"
    "configs/train/grpo/ig_full_4090.yaml"
)

for config in "${configs[@]}"; do
    if [ -f "$config" ]; then
        echo "  ✓ $config"
    else
        echo "  ✗ $config (不存在)"
    fi
done

# 检查磁盘空间
echo ""
echo "磁盘空间:"
df -h . | tail -1 | awk '{print "  可用: "$4" / 总量: "$2}'

# 检查端口
echo ""
echo "端口检查:"
if command -v nc &> /dev/null; then
    if nc -z localhost 8000 2>/dev/null; then
        echo "  ⚠️  端口 8000 已被占用"
    else
        echo "  ✓ 端口 8000 可用"
    fi
    if nc -z localhost 8001 2>/dev/null; then
        echo "  ⚠️  端口 8001 已被占用"
    else
        echo "  ✓ 端口 8001 可用"
    fi
else
    echo "  (跳过: nc 命令不可用)"
fi

echo ""
echo "======================================"
echo "检查完成!"
echo "======================================"
