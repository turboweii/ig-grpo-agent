# IG-GRPO 云服务器部署完整步骤

> GT-Ubuntu22.04-CMD-V3.0 环境 + 8×4090

---

## 第一步：登录服务器并检查环境

```bash
# SSH 登录（使用云平台提供的终端或 SSH）
# 登录后首先检查 GPU

nvidia-smi
# 应该看到 8 张 4090，每张 24GB

# 检查 Python 版本
python --version
# 需要 >= 3.10，如果不是，告诉管理员

# 检查 CUDA 版本
nvcc --version
# 或者
cat /usr/local/cuda/version.txt
# 需要 >= 12.1
```

---

## 第二步：下载模型

```bash
# 安装 huggingface-cli
pip install huggingface_hub -q

# 设置加速传输
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/root/models

# 创建模型目录
mkdir -p /root/models

# 下载 7B 模型 (~15GB, 约10-30分钟)
echo "下载 Qwen2.5-7B-Instruct..."
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir /root/models/Qwen2.5-7B-Instruct \
    --local-dir-use-symlinks False

# 下载 72B-AWQ 模型 (~45GB, 约30-60分钟)
echo "下载 Qwen2.5-72B-Instruct-AWQ..."
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ \
    --local-dir /root/models/Qwen2.5-72B-Instruct-AWQ \
    --local-dir-use-symlinks False

# 验证下载完成
ls -lh /root/models/
# 应该看到两个目录
```

---

## 第三步：上传项目文件

### 方式 A：在云服务器上直接创建（如果支持 Git）

```bash
# 安装 git
apt update && apt install git -y

# 克隆项目（如果你的代码在 Git 仓库）
cd /root
git clone <你的仓库地址> ig-grpo-agent
```

### 方式 B：从本地上传（推荐）

```bash
# 在你的本地机器上，打包项目
cd /path/to/agentic-grpo-longhorizon-main/ig-grpo-agent
tar -czf ig-grpo-agent.tar.gz \
    --exclude='outputs' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='.git' .

# 上传到服务器
scp ig-grpo-agent.tar.gz root@<服务器IP>:/root/

# 或者使用云平台提供的文件上传功能
```

然后在服务器上解压：

```bash
cd /root
tar -xzf ig-grpo-agent.tar.gz
mv ig-grpo-agent ig-grpo-agent-orig  # 如果有旧版本
mkdir ig-grpo-agent
cd ig-grpo-agent
tar -xzf ../ig-grpo-agent.tar.gz --strip-components=1
```

---

## 第四步：安装依赖

```bash
cd /root/ig-grpo-agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装 PyTorch (CUDA 12.1)
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# 安装 vLLM
pip install vllm==0.9.2 -q

# 安装 flash-attn (需要编译，约5-10分钟)
pip install flash-attn --no-build-isolation -q

# 安装其他依赖
pip install -r requirements.txt -q

# 安装子模块（如果需要）
cd /root
# tau-bench
git clone https://github.com/sierra-research/tau-bench
cd tau-bench && pip install -e . -q && cd ..

# veRL
git clone https://github.com/volcengine/verl
cd verl && pip install -e . -q && cd ..

cd ig-grpo-agent
```

---

## 第五步：配置环境变量

```bash
# 创建环境配置
cat > /root/ig-grpo-agent/.env << 'EOF'
# 模型路径
export HF_HOME=/root/models
export TRANSFORMERS_CACHE=/root/models/hub

# 离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# vLLM 配置
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
EOF

# 添加到启动脚本
echo "source /root/ig-grpo-agent/.env" >> ~/.bashrc
source ~/.bashrc
```

---

## 第六步：验证环境

```bash
cd /root/ig-grpo-agent
source venv/bin/activate
source .env

# 运行验证脚本
bash verify_env.sh

# 应该看到：
# ✓ 8 个 GPU
# ✓ Python 3.10+
# ✓ 所有必要包已安装
# ✓ 两个模型都存在
```

---

## 第七步：开始训练

### 7.1 启动 vLLM 服务

```bash
# 激活环境
cd /root/ig-grpo-agent
source venv/bin/activate
source .env

# 启动 7B 服务 (使用 GPU 0)
screen -S vllm_7b
CUDA_VISIBLE_DEVICES=0 bash scripts/vllm_server/7b.sh
# 等待看到 "Uvicorn running on http://0.0.0.0:8000"
# 按 Ctrl+A 然后按 D 分离

# 启动 72B-AWQ 服务 (使用 GPU 1)
screen -S vllm_72b
CUDA_VISIBLE_DEVICES=1 bash scripts/vllm_server/72b.sh
# 等待看到 "Uvicorn running on http://0.0.0.0:8001"
# 按 Ctrl+A 然后按 D 分离

# 验证服务运行
curl http://localhost:8000/health
curl http://localhost:8001/health
```

### 7.2 采集 SFT 数据

```bash
# 使用 8 卡并行采集
bash scripts/train/sft/collect_retail_data.sh

# 这会需要 1-2 小时
# 完成后会自动准备训练数据
```

### 7.3 运行 SFT 训练

```bash
# 8 卡 SFT 训练
bash scripts/train/sft/run_sft_retail_8gpu.sh

# 训练完成后合并 LoRA
bash scripts/train/sft/run_merge_lora.sh
```

### 7.4 开始 IG-GRPO 训练

```bash
# 使用 screen 后台运行
screen -S grpo_train
bash scripts/train/grpo/run_ig_full_8x4090.sh
# 按 Ctrl+A 然后按 D 分离

# 监控训练
watch -n 1 nvidia-smi          # GPU 使用
tail -f outputs/ig_full_8x4090/training.log  # 训练日志
```

---

## 第八步：监控训练

```bash
# 查看所有 screen 会话
screen -ls

# 重新连接到训练会话
screen -r grpo_train

# 查看检查点
ls -lh experiments/ig_full_8x4090/checkpoints/

# 查看 GPU 使用
nvidia-smi
```

---

## Screen 命令速查

```bash
screen -S <名称>       # 创建新会话
screen -ls             # 列出所有会话
screen -r <名称>       # 连接到会话
# 在 screen 内:
Ctrl+A D               # 分离会话
Ctrl+A C               # 创建新窗口
Ctrl+A N               # 下一个窗口
Ctrl+A P               # 上一个窗口
exit                   # 退出当前窗口/会话
```

---

## 常见问题

### 模型下载慢
```bash
# 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com
# 然后重新下载
```

### 显存不足
```bash
# 降低 batch size
# 编辑 configs/train/grpo/ig_full_8x4090.yaml
# 将 train_batch_size: 4 改为 2
```

### 连接断开
```bash
# 使用 screen 运行所有长时间任务
# 这样断开 SSH 连接后训练继续
```

---

## 预期时间线

| 阶段 | 时间 |
|------|------|
| 模型下载 | 30-90 分钟 |
| 依赖安装 | 20-30 分钟 |
| SFT 数据采集 | 1-2 小时 |
| SFT 训练 | 1 小时 |
| IG-GRPO 训练 | 8-10 小时 |
| **总计** | **~12-16 小时** |
