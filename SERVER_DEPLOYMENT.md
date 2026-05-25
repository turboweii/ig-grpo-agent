# IG-GRPO 服务器部署完整流程

> 适用于 4×4090 (24GB) 服务器，包含模型下载、环境配置、数据采集全流程

---

## 📋 部署前准备

### 硬件要求
- **GPU**: 4 × NVIDIA 4090 (24GB)
- **内存**: 至少 128GB
- **存储**: 至少 200GB SSD (模型 ~60GB + 数据 ~20GB + 中间结果)

### 软件要求
- **CUDA**: 12.1+
- **Python**: 3.10+
- **驱动**: NVIDIA 525+

---

## 🚀 完整部署流程

### 第一步：准备模型文件

#### 需要下载的模型
```
1. Qwen/Qwen2.5-7B-Instruct (~15GB)
2. Qwen/Qwen2.5-72B-Instruct-AWQ (~45GB)
```

#### 方案 A: 使用 Hugging Face CLI (本地下载后上传)

```bash
# 在本地机器上
pip install huggingface_hub

# 下载 7B 模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir ./models/Qwen2.5-7B-Instruct \
    --local-dir-use-symlinks False

# 下载 72B-AWQ 模型
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ \
    --local-dir ./models/Qwen2.5-72B-Instruct-AWQ \
    --local-dir-use-symlinks False
```

#### 方案 B: 在服务器直接下载 (推荐，如果网络好)

```bash
# 在服务器上
pip install huggingface_hub
export HF_HOME=/root/models
export HF_HUB_ENABLE_HF_TRANSFER=1  # 启用加速传输

# 下载 7B
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir /root/models/Qwen2.5-7B-Instruct \
    --local-dir-use-symlinks False

# 下载 72B-AWQ
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ \
    --local-dir /root/models/Qwen2.5-72B-Instruct-AWQ \
    --local-dir-use-symlinks False
```

#### 方案 C: 使用镜像站 (国内)

```bash
# 使用 hf-mirror 加速
export HF_ENDPOINT=https://hf-mirror.com

# 然后使用上面的下载命令
```

---

### 第二步：上传项目文件

#### 在本地机器上打包上传

```bash
# 1. 打包项目 (排除不必要文件)
cd ig-grpo-agent
tar -czf ../ig-grpo-agent.tar.gz \
    --exclude='outputs' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='.git' .

# 2. 上传到服务器
scp ../ig-grpo-agent.tar.gz user@server:/root/

# 3. 或使用 rsync (支持断点续传)
rsync -avz --progress \
    --exclude='outputs' --exclude='__pycache__' \
    ig-grpo-agent/ user@server:/root/ig-grpo-agent/

# 4. 上传模型 (如果本地下载了，这会花很长时间)
rsync -avz --progress \
    ./models/Qwen2.5-7B-Instruct/ \
    user@server:/root/models/Qwen2.5-7B-Instruct/
```

#### 在服务器上解压

```bash
# SSH 登录服务器
ssh user@server

# 解压项目
cd /root
tar -xzf ig-grpo-agent.tar.gz
```

---

### 第三步：服务器环境配置

```bash
# 1. 创建工作目录
mkdir -p ~/ig-grpo
cd ~/ig-grpo

# 2. 检查 Python 版本
python --version  # 需要 >= 3.10

# 3. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 4. 安装 PyTorch (CUDA 12.1)
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu121

# 5. 安装项目依赖
cd ig-grpo-agent
pip install -r requirements.txt

# 6. 安装子模块 (假设与项目同级)
cd ..
# 如果需要手动克隆 tau-bench 和 verl
git clone https://github.com/volcengine/verl
git clone https://github.com/sierra-research/tau-bench

cd tau-bench && pip install -e . && cd ..
cd verl && pip install -e . && cd ..
cd ig-grpo-agent
```

---

### 第四步：配置环境变量和路径

```bash
# 创建环境配置文件
cat > ~/ig-grpo/ig-grpo-agent/.env << 'EOF'
# 模型路径配置
export HF_HOME=/root/models
export TRANSFORMERS_CACHE=/root/models/hub

# 离线模式 (服务器无外网时)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# vLLM 配置
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 显存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
EOF

# 添加到启动脚本
echo "source ~/ig-grpo/ig-grpo-agent/.env" >> ~/.bashrc
source ~/.bashrc
```

---

### 第五步：修改配置文件中的模型路径

```bash
# 编辑 vLLM 服务脚本
nano scripts/vllm_server/7b.sh
# 修改 model_path 为服务器上的路径
# model_path="/root/models/Qwen2.5-7B-Instruct"

nano scripts/vllm_server/72b.sh
# 修改 model_path="/root/models/Qwen2.5-72B-Instruct-AWQ"

# 编辑训练配置
nano configs/train/grpo/ig_full_4090.yaml
# 确认模型路径正确
```

---

### 第六步：验证环境

```bash
cd ~/ig-grpo/ig-grpo-agent
source venv/bin/activate
source .env

# 检查 GPU
nvidia-smi

# 检查 CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# 检查关键包
python -c "
import torch, vllm, transformers, verl
import tau_bench, peft, pybloom_live
print('All packages OK!')
"

# 检查模型文件
ls -lh /root/models/Qwen2.5-7B-Instruct/
ls -lh /root/models/Qwen2.5-72B-Instruct-AWQ/
```

---

## 📝 完整训练流程

### 1. 启动 vLLM 服务

```bash
cd ~/ig-grpo/ig-grpo-agent
source venv/bin/activate
source .env

# 使用 screen 或 tmux 后台运行
screen -S vllm_7b
bash scripts/vllm_server/7b.sh
# Ctrl+A, D 分离

screen -S vllm_72b
bash scripts/vllm_server/72b.sh
# Ctrl+A, D 分离

# 验证服务运行
curl http://localhost:8000/health
curl http://localhost:8001/health
```

### 2. 采集 SFT 数据

```bash
# 完整采集 (约 2-4 小时)
bash scripts/train/sft/collect_retail_data.sh

# 或快速测试 (约 5 分钟)
python scripts/train/sft/collect_sft_data.py \
    --config configs/train/sft/sft_collect_retail_tiny.yaml
```

### 3. 运行 SFT 训练

```bash
bash scripts/train/sft/run_sft_retail.sh

# 训练完成后合并 LoRA
bash scripts/train/sft/run_merge_lora.sh
```

### 4. 开始 IG-GRPO 训练

```bash
# 使用 screen 后台运行
screen -S grpo_train
bash scripts/train/grpo/run_ig_full_4090.sh
# Ctrl+A, D 分离

# 查看训练进度
tail -f outputs/ig_full_4090/training.log

# 查看 GPU 使用
watch -n 1 nvidia-smi
```

---

## 🔧 快速部署脚本

保存为 `~/quick_deploy.sh`:

```bash
#!/bin/bash
set -e

echo "======================================"
echo "IG-GRPO Quick Deployment"
echo "======================================"

# 配置
PROJECT_DIR="$HOME/ig-grpo"
MODEL_DIR="$HOME/models"

# 创建目录
mkdir -p $PROJECT_DIR $MODEL_DIR
cd $PROJECT_DIR

# 检查环境
echo "检查环境..."
python --version || (echo "需要 Python 3.10+" && exit 1)

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python -m venv venv
fi
source venv/bin/activate

# 检查项目
if [ ! -d "ig-grpo-agent" ]; then
    echo "错误: ig-grpo-agent 目录不存在!"
    echo "请先上传项目文件"
    exit 1
fi

cd ig-grpo-agent

# 安装依赖
echo "安装 PyTorch..."
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu121 -q

echo "安装项目依赖..."
pip install -r requirements.txt -q

# 检查模型
echo ""
echo "检查模型文件..."
if [ -d "$MODEL_DIR/Qwen2.5-7B-Instruct" ]; then
    echo "✓ Qwen2.5-7B-Instruct"
else
    echo "✗ Qwen2.5-7B-Instruct 不存在"
fi

if [ -d "$MODEL_DIR/Qwen2.5-72B-Instruct-AWQ" ]; then
    echo "✓ Qwen2.5-72B-Instruct-AWQ"
else
    echo "✗ Qwen2.5-72B-Instruct-AWQ 不存在"
fi

echo ""
echo "======================================"
echo "部署完成!"
echo "======================================"
echo ""
echo "下一步:"
echo "  1. 下载模型 (如果还没有)"
echo "  2. 修改配置文件中的模型路径"
echo "  3. 启动 vLLM 服务"
echo "  4. 采集 SFT 数据"
echo ""
echo "运行验证:"
echo "  source venv/bin/activate"
echo "  bash verify_env.sh"
```

---

## 📊 监控训练

```bash
# 实时 GPU 监控
watch -n 1 nvidia-smi

# 查看训练日志
tail -f outputs/ig_full_4090/training.log

# 查看检查点
ls -lh experiments/ig_full_4090/checkpoints/

# 查看 SFT 采集统计
cat experiments/sft_retail/raw/summary.json
```

---

## ⚠️ 常见问题

### 显存不足
```yaml
# 编辑 configs/train/grpo/ig_full_4090.yaml
train_batch_size: 1          # 降低 batch size
max_model_len: 12288         # 降低序列长度
ppo_micro_batch_size_per_gpu: 1
```

### 多卡通信问题
```bash
# 添加到 .env
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

### 模型加载失败
```bash
# 检查模型路径
ls -lh /root/models/Qwen2.5-7B-Instruct/config.json

# 检查环境变量
echo $HF_HOME
echo $TRANSFORMERS_CACHE
```

---

## ✅ 部署检查清单

- [ ] 4×4090 GPU 可用
- [ ] Python 3.10+ 已安装
- [ ] CUDA 12.1+ 已安装
- [ ] 项目文件已上传
- [ ] 模型文件已下载并验证
- [ ] 虚拟环境已创建
- [ ] 依赖包已安装 (PyTorch, vLLM, tau-bench, veRL)
- [ ] 环境变量已配置
- [ ] 配置文件中的路径已修改
- [ ] vLLM 服务可正常启动
- [ ] 端口未被占用 (8000, 8001)
