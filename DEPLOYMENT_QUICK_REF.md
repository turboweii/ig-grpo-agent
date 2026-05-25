# 服务器部署快速参考

## 完整训练流程

IG-GRPO 采用两阶段训练：
1. **SFT 训练** - 预热模型，学习基本工具调用
2. **GRPO 训练** - 使用 JIG 奖励进行强化学习

## 本地准备

```bash
# 1. 修改上传脚本中的服务器信息
vim scripts/deploy/upload_to_server.sh

# 2. 上传项目
bash scripts/deploy/upload_to_server.sh
```

## 服务器操作

```bash
# 1. SSH 登录
ssh user@server

# 2. 创建 conda 环境
conda create -n ig-grpo python=3.10 -y
conda activate ig-grpo

# 3. 安装 PyTorch (CUDA 12.6)
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu126

# 4. 安装 vLLM
pip install vllm==0.9.2

# 5. 安装 flash-attn (需编译)
pip install flash-attn --no-build-isolation

# 6. 进入项目目录并安装其他依赖
cd /path/to/ig-grpo-agent
pip install -r requirements.txt

# 7. 安装 tau-bench (与项目同级)
cd /path/to/parent
git clone https://github.com/sierra-research/tau-bench
cd tau-bench
pip install -e .

# 8. 设置环境变量
export VLLM_USE_V1=1

# 9. 检查环境
python ig-grpo-agent/scripts/deploy/check_server_env.py

# 10. 完整训练流程
cd ig-grpo-agent

# 方式1: 一键运行完整流程 (推荐)
bash scripts/train/full_pipeline.sh

# 方式2: 分步运行
# Step 1: SFT 训练 (4卡)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train/sft/run_sft_retail.sh

# Step 2: 合并 LoRA 权重
bash scripts/train/sft/run_merge_lora.sh

# Step 3: GRPO 训练 (4卡)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m verl.trainer.main_ppo \
    --config-path=configs \
    --config-name=train/grpo/ig_full_4090
```

## 后台运行

```bash
# 使用 tmux
tmux new -s ig-grpo
cd /path/to/ig-grpo-agent
source venv/bin/activate
# 运行训练命令
# Ctrl+B D 分离

# 重新连接
tmux attach -t ig-grpo
```

## 监控命令

```bash
# GPU 监控
watch -n 1 nvidia-smi

# 日志监控
tail -f outputs/ig_full_4090/training.log

# 检查点
ls -lh experiments/ig_full_4090/checkpoints/
```

## 关键配置

### SFT 训练 (4×4090)
| 参数 | 值 | 说明 |
|------|-----|------|
| num_gpus | 4 | 使用 4 卡并行 |
| per_device_batch_size | 2 | 每卡 batch=2 |
| gradient_accumulation_steps | 4 | 梯度累积 |
| **等效 batch** | **32** | 4×2×4=32 |
| max_length | 8192 | 序列长度 |
| num_epochs | 3 | 训练轮数 |
| lr | 1e-4 | 学习率 |

### GRPO 训练
| 参数 | 值 | 说明 |
|------|-----|------|
| n_gpus_per_node | 4 | GPU 数量 |
| tensor_model_parallel_size | 2 | vLLM 并行度 |
| train_batch_size | 2 | 批次大小 |
| max_model_len | 16384 | 序列长度 |
| gpu_memory_utilization | 0.90 | 显存利用率 |

## 训练流程说明

```
Step 1: SFT 训练 (4×4090 并行)
├── 输入: Qwen/Qwen2.5-7B-Instruct
├── 配置: 4卡 DDP, batch=32
├── 输出: experiments/sft_retail (LoRA 权重)
└── 用时: ~1-2 小时 (4卡加速)

Step 2: 合并 LoRA 权重
├── 输入: 基础模型 + LoRA 权重
├── 输出: experiments/sft_lora_merged
└── 用时: ~5 分钟

Step 3: GRPO 训练 (4×4090)
├── 输入: experiments/sft_lora_merged
├── 配置: 2卡 vLLM + 2卡 FSDP
├── 输出: experiments/ig_full_4090/checkpoints
└── 用时: ~6-8 小时 (300 steps)
```
