# IG-GRPO 云服务器部署完整步骤

> GT-Ubuntu22.04-CMD-V3.0 环境 + 8×4090
> 
> **项目结构**：IG-GRPO (Information Gain Guided GRPO) 用于长链路多工具智能体训练

---

## 项目概述

IG-GRPO 通过以下核心组件解决长链路任务中的探索塌缩问题：

| 组件 | 功能 | 文件 |
|------|------|------|
| **JIG** | 联合互信息奖励（状态+工具+转移新颖性） | `src/envs/jig_components.py` |
| **G-Normalization** | L^0.5 归一化，保证梯度方差有界 | `src/envs/g_normalization.py` |
| **HierarchicalCoverage** | 三层 Bloom Filter 状态追踪 | `src/envs/jig_components.py` |
| **SustainedExploration** | 防止过早收敛的持续性保证 | `src/envs/jig_components.py` |
| **AsyncEntropyEstimator** | 异步熵计算，不阻塞 rollout | `src/envs/async_entropy_estimator.py` |

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
    --exclude='outputs' --exclude='experiments' \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.git' --exclude='venv' .

# 上传到服务器
scp ig-grpo-agent.tar.gz root@<服务器IP>:/root/

# 或者使用云平台提供的文件上传功能
```

然后在服务器上解压：

```bash
cd /root
tar -xzf ig-grpo-agent.tar.gz
mkdir -p ig-grpo-agent
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

# 安装 Bloom Filter (用于状态追踪)
pip install pybloom-live -q

# 安装子模块（如果需要）
cd /root
# tau-bench
git clone https://github.com/sierra-research/tau-bench
cd tau-bench && pip install -e . -q && cd ..

# veRL
git clone https://github.com/volcengine/verl --branch v0.6.1
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

# IG-GRPO 配置
export IG_GRPO_PROJECT_ROOT=/root/ig-grpo-agent
export IG_GRPO_EXPERIMENTS_DIR=/root/ig-grpo-agent/experiments
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

## 第七步：准备训练配置

### 7.1 配置文件结构

```bash
# 查看可用的训练配置
ls configs/train/grpo/
# vanilla.yaml      - 基线（无 IG）
# ig_fixed.yaml      - 固定 IG 权重
# ig_combo.yaml      - IG + 工具组合奖励
# ig_full.yaml       - 完整 IG-GRPO（推荐）

# 查看交互配置
ls configs/interaction_config/
# tau_bench_retail_vanilla.yaml
# tau_bench_retail_ig_fixed.yaml
# tau_bench_retail_ig_combo.yaml
# tau_bench_retail_jig.yaml
```

### 7.2 IG-GRPO 核心参数说明

```yaml
# JIG 组件权重 (alpha_state + alpha_tool + alpha_transfer + alpha_sustained = 1.0)
jig_config:
  alpha_state: 0.3       # 状态新颖性
  alpha_tool: 0.4        # 工具组合新颖性（核心）
  alpha_transfer: 0.2    # 工具转移新颖性
  
  # G-Normalization (L^0.5)
  g_normalization:
    enabled: true
    gamma: 0.5           # γ ≤ 0.5 保证梯度方差有界
    adaptive: false      # 是否使用自适应调度
    
  # 课程学习
  total_steps: 300
  curriculum_beta: 0.7   # 衰减指数
  alpha_0: 0.8           # 初始权重
```

---

## 第八步：开始训练

### 8.1 启动 vLLM 服务

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

### 8.2 采集 SFT 数据

```bash
# 使用 8 卡并行采集
bash scripts/train/sft/collect_retail_data.sh

# 这会需要 1-2 小时
# 完成后会自动准备训练数据
```

### 8.3 运行 SFT 训练

```bash
# 8 卡 SFT 训练
bash scripts/train/sft/run_sft_retail_8gpu.sh

# 训练完成后合并 LoRA
bash scripts/train/sft/run_merge_lora.sh
```

### 8.4 开始 IG-GRPO 训练

```bash
# 使用 screen 后台运行
screen -S grpo_train
bash scripts/train/grpo/run_ig_full_8x4090.sh
# 按 Ctrl+A 然后按 D 分离

# 或者直接使用 Hydra 配置
python -m verl.trainers.main \
    --config-path=configs/train/grpo \
    --config-name=ig_full_8x4090 \
    trainer.total_training_steps=300

# 监控训练
watch -n 1 nvidia-smi          # GPU 使用
tail -f outputs/ig_full_8x4090/training.log  # 训练日志
```

### 8.5 运行消融实验（可选）

如果需要对比实验效果：

```bash
# Vanilla 基线（无 IG）
bash scripts/train/grpo/run_vanilla.sh

# IG-Fixed（固定权重）
bash scripts/train/grpo/run_exp1_ig_fixed.sh

# IG-Combo（工具组合奖励）
bash scripts/train/grpo/run_exp2_ig_combo.sh

# IG-Full（完整方案）
bash scripts/train/grpo/run_exp3_ig_full.sh
```

---

## 第九步：监控训练

### 9.1 监控 GPU 和进程

```bash
# 查看所有 screen 会话
screen -ls

# 重新连接到训练会话
screen -r grpo_train

# 查看 GPU 使用
nvidia-smi
```

### 9.2 查看训练指标

```bash
# JIG 统计（状态覆盖率、工具多样性等）
tail -f experiments/ig_full/checkpoints/*/jig_stats.json

# G-Normalization 统计（归一化因子、压缩比等）
tail -f experiments/ig_full/checkpoints/*/g_norm_stats.json

# 训练曲线（SwanLab）
# 在浏览器打开：http://<服务器IP>:5050
```

### 9.3 检查点管理

```bash
# 查看检查点
ls -lh experiments/ig_full/checkpoints/

# 检查点包含：
# - actor_model (策略模型)
# - jig_stats (JIG 统计)
# - g_norm_stats (G-Norm 统计)
# - coverage_tracker (状态覆盖记录)
```

---

## 第十步：评估与测试

```bash
# 运行 Entropy@K 评测
python -m src.evaluation.entropy_at_k_eval \
    --checkpoint experiments/ig_full/checkpoints/checkpoint_250 \
    --k 4 \
    --output eval_results/entropy_at_k.json

# 运行 Pass@K 评测
python -m src.evaluation.pass_k_eval \
    --checkpoint experiments/ig_full/checkpoints/checkpoint_250 \
    --k 1,4 \
    --output eval_results/pass_k.json

# 查看评测结果
cat eval_results/entropy_at_k.json | jq .
cat eval_results/pass_k.json | jq .
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

### G-Normalization 警告
```bash
# 如果看到 "gradient variance may be unbounded" 警告
# 确保 configs/interaction_config/*.yaml 中:
# g_normalization:
#   enabled: true
#   gamma: 0.5  # 必须 ≤ 0.5
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
| 评测验证 | 1-2 小时 |
| **总计** | **~13-17 小时** |

---

## 预期结果

根据消融实验设计，预期 IG-Full 的性能指标：

| 指标 | Vanilla | IG-Full | 提升 |
|------|---------|---------|------|
| **Pass@1** | 0.175 | 0.248 | +42% |
| **状态覆盖率** | 18% | 45% | +150% |
| **工具多样性熵** | 1.82 | 2.73 | +50% |
| **平均路径长度** | 8.2 | 5.9 | -28% |
| **泛化 Pass@1** | 0.071 | 0.128 | +80% |

---

## 参考文档

- **理论证明**：[docs/theory/g_normalization_proof.md](docs/theory/g_normalization_proof.md)
- **消融报告**：[docs/exploration/ablation_report.md](docs/exploration/ablation_report.md)
- **项目计划**：[ig-grpo-project-plan.md](../ig-grpo-project-plan.md)
