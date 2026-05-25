# IG-GRPO 8×4090 配置说明

> 专为 8×4090 (24GB) 优化的训练配置

## 📁 新建文件

| 文件 | 用途 |
|------|------|
| [`configs/train/grpo/ig_full_8x4090.yaml`](configs/train/grpo/ig_full_8x4090.yaml) | IG-GRPO 主训练配置 |
| [`configs/train/sft/sft_retail_lora_8gpu.yaml`](configs/train/sft/sft_retail_lora_8gpu.yaml) | SFT 训练配置 |
| [`scripts/train/grpo/run_ig_full_8x4090.sh`](scripts/train/grpo/run_ig_full_8x4090.sh) | IG-GRPO 训练脚本 |
| [`scripts/train/sft/run_sft_retail_8gpu.sh`](scripts/train/sft/run_sft_retail_8gpu.sh) | SFT 训练脚本 |

---

## 🚀 完整训练流程

### 第一步：启动 vLLM 服务

```bash
# GPU 0: 7B 教师策略
screen -S vllm_7b
CUDA_VISIBLE_DEVICES=0 bash scripts/vllm_server/7b.sh
# Ctrl+A, D

# GPU 1: 72B 用户模拟器
screen -S vllm_72b
CUDA_VISIBLE_DEVICES=1 bash scripts/vllm_server/72b.sh
# Ctrl+A, D
```

### 第二步：采集 SFT 数据

```bash
# 使用8卡并行采集 (best_of_n=16)
bash scripts/train/sft/collect_retail_data.sh
```

### 第三步：运行 SFT 训练

```bash
# 8卡 SFT 训练，等效 batch=128
bash scripts/train/sft/run_sft_retail_8gpu.sh

# 合并 LoRA
bash scripts/train/sft/run_merge_lora.sh
```

### 第四步：开始 IG-GRPO 训练

```bash
# 8卡 IG-GRPO 训练
screen -S grpo_train
bash scripts/train/grpo/run_ig_full_8x4090.sh
# Ctrl+A, D
```

---

## 📊 8卡 vs 4卡 对比

| 指标 | 4×4090 | 8×4090 | 提升 |
|------|--------|--------|------|
| **Batch Size** | 2 | 4 | 2x |
| **Group Size (n)** | 4 | 8 | 2x |
| **Max Context** | 16K | 20K | +25% |
| **Max Seqs** | 16 | 32 | 2x |
| **等效 SFT Batch** | 64 | 128 | 2x |
| **训练速度** | 基准 | ~1.8x | +80% |

---

## 🔧 关键配置差异

### GRPO 训练

```yaml
# 4卡配置
train_batch_size: 2
n: 4                         # group size
max_model_len: 16384
max_num_seqs: 16
tensor_model_parallel_size: 4

# 8卡配置 ✨
train_batch_size: 4           # 2x
n: 8                         # 2x
max_model_len: 20480          # +25%
max_num_seqs: 32              # 2x
tensor_model_parallel_size: 4 # 保持4卡TP，另外4卡训练
```

### SFT 训练

```yaml
# 4卡配置
per_device_batch_size: 4
gradient_accumulation_steps: 4
等效 batch: 64

# 8卡配置 ✨
per_device_batch_size: 8      # 2x
gradient_accumulation_steps: 2
等效 batch: 128               # 2x
```

---

## 💡 GPU 分配策略

### IG-GRPO 训练时的 GPU 分配

```
GPU 0-3:   vLLM Rollout (TP=4)
  ├── 7B 策略模型推理
  └── 生成 trajectories

GPU 4-7:   训练 (FSDP)
  ├── Actor 梯度更新
  ├── Reference 模型
  └── 优化器状态
```

这种分离策略使得：
- Rollout 和训练可以**并行**进行
- 吞吐量提升 ~80%
- 不需要等待 rollout 完成才开始训练

---

## ⚡ 性能优化

### 1. 数据采集加速
```yaml
# 8卡可以同时采集更多样本
best_of_n: 16      # 4卡是8
num_workers: 8     # 4卡是4
```

### 2. 训练加速
```yaml
# 更大的 batch size
ppo_micro_batch_size_per_gpu: 2  # 4卡是1
gradient_accumulation_steps: 2   # 4卡是4
```

### 3. 推理加速
```yaml
# 更大的并发
max_num_seqs: 32   # 4卡是16
max_model_len: 20K # 4卡是16K
```

---

## 📈 预期训练时间

| 阶段 | 4×4090 | 8×4090 |
|------|--------|--------|
| SFT 数据采集 | ~3小时 | ~1.5小时 |
| SFT 训练 | ~2小时 | ~1小时 |
| IG-GRPO 300步 | ~15小时 | ~8小时 |
| **总计** | **~20小时** | **~10.5小时** |

---

## ✅ 部署检查清单

- [ ] 8×4090 GPU 可用
- [ ] 项目已上传到服务器
- [ ] 模型已下载 (7B + 72B-AWQ)
- [ ] 依赖已安装
- [ ] 环境变量已配置
- [ ] vLLM 服务可启动
- [ ] 端口 8000, 8001 可用
