# IG-GRPO — 信息增益引导的高效探索框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **解决多工具 Agent 的探索-利用困境**  
> 在 τ-bench retail（50 任务、40 工具、多轮对话场景）上，通过 **联合互信息（JIG）奖励 + 课程学习 + 探索持续性**，相较 Vanilla GRPO 基线实现 **+42% 的整体 Pass@1 提升** 和 **+150% 的状态覆盖率提升**。

---

## 🔥 核心结果

**最优检查点（step 250）：完整 IG-GRPO 方案达到 0.248 的整体 Pass@1 —— 相较 Vanilla GRPO 基线（0.175）提升 +42%。**

| 指标 | Vanilla | IG-Fixed | IG-Combo | **IG-Full** | 相较 Vanilla |
|------|---------|----------|----------|------------|-------------|
| **整体 Pass@1** | 0.175 | 0.195 | 0.218 | **0.248** | **+42%** |
| **泛化 Pass@1** | 0.071 | 0.092 | 0.108 | **0.128** | **+80%** |
| **状态覆盖率** | 18% | 32% | 38% | **45%** | **+150%** |
| **工具多样性熵** | 1.82 | 2.15 | 2.45 | **2.73** | **+50%** |
| **平均路径长度** | 8.2 | 7.1 | 6.4 | **5.9** | **−28%** |
| **冗余调用率** | 28% | 22% | 16% | **12%** | **−57%** |
| **峰值显存** | 120 GB | 95 GB | 78 GB | **65 GB** | **−46%** |

> *泛化 Pass@1 = (uncovered_seen × 24 + unseen × 10) / 34，排除训练集泄漏影响的核心指标。*

---

## 🎯 问题与动机

标准的 **GRPO（Group Relative Policy Optimization）** 应用于多工具对话智能体（τ-bench retail，50 任务，40 工具）时会发生**探索塌缩**问题。我们识别出**三个根本原因**：

### 1. 稀疏奖励塌缩
结果奖励是二元的（0/1）且仅在任务结束时给出。长链路中间步骤无反馈，Agent 无法区分"正在探索"和"已经失败"。

### 2. 重复模式循环
Agent 发现一个可行的工具调用模式后，会不断重复该模式而停止探索新的工具组合。

### 3. 状态覆盖不足
大量 state-tool 组合从未被访问，Agent 实际上只探索了状态空间的一小部分。

> **关键发现**：仅增加熵奖励或探索噪声是不够的，需要**引导 Agent 访问真正有价值的新状态-工具组合**。

---

## 💡 方法

我们设计并验证了**三个渐进式消融实验**，每个针对特定的失效模式：

### 实验 1：IG-Fixed — 固定权重的信息增益

**思路**：用信息论中的互信息最大化驱动探索。

**机制**：`IG = H(State) - H(State | tool)`，奖励能最大程度降低状态不确定性的工具调用。

**结果**：状态覆盖率从 18% 提升到 32%，Pass@1 从 0.175 提升到 0.195（+11%），但工具多样性仍然不足。

### 实验 2：IG-Combo — 工具组合奖励

**思路**：增加 state-tool 组合的新颖性奖励，鼓励尝试新的工具组合。

**机制**：三层 Bloom Filter 追踪状态/工具/组合的覆盖情况，给予不同层级的新颖性奖励。

**结果**：工具多样性熵从 1.82 提升到 2.45（+35%），Pass@1 进一步提升到 0.218（+24%）。

### 实验 3：IG-Full — 完整方案 ⭐

**思路**：**JIG 奖励提供局部质量信号；课程学习平衡探索-利用；探索持续性保证防止过早收敛。**

**机制**：
- **联合互信息（JIG）**：`JIG = α_state × 状态新颖性 + α_tool × 工具新颖性 + α_transfer × 转移新颖性`
- **课程学习**：`α(t) = α_0 × (1 - t/T)^β`，早期高权重鼓励探索，后期低权重收敛利用
- **G-Normalization**：`advantage / L^γ`，防止长轨迹淹 Episode 级信号
- **探索持续性**：检测 IG 趋势，持续下降时给予额外探索激励

**结果**：**0.248 整体表现** —— 超越所有单组件基线。状态覆盖率 45%，工具多样性熵 2.73，冗余调用率降至 12%。

> **核心洞察**：价值不在于单独拥有互信息*或*新颖性奖励 —— 而在于**联合信号**。JIG 提供多维度的新颖性度量；课程学习确保探索与利用的平衡；探索持续性保证防止过早收敛。三者缺一不可。

---

## 🌟 技术亮点

### 1. 联合互信息理论（算法贡献）

消融报告实证证明了多工具 Agent GRPO 的**联合信息最大化原理**：
- **状态新颖性**（α_state）：鼓励访问新状态
- **工具组合新颖性**（α_tool）：鼓励用新工具处理已知状态
- **转移新颖性**（α_transfer）：鼓励尝试罕见的工具序列

> 此原理是**模型无关的**，适用于 τ-bench 之外的任何多工具 RL 任务。

### 2. 三层 Bloom Filter（工程创新）

完全可解释、零可训练参数的状态追踪：
- **短期层**：当前 episode 的状态（100K 容量，0.1% 误差）
- **中期层**：最近 100 个 episode 的状态（1M 容量，1% 误差）
- **长期层**：整个训练历史的状态（10M 容量，5% 误差）

内存占用从 O(|S|) 降至 **O(bits)**，可处理千万级状态空间。

### 3. 内存高效的训练系统（工程）

- **Bypass 模式 + 融合内核 + TP=2** 将每步内存峰值从 **120GB 降至 65GB**，在 **2×A800** 上实现 7B 策略 + 72B-AWQ 模拟器。
- **双缓冲熵估计**：异步计算不阻塞 rollout，训练吞吐提升 ×1.6
- **离线优先**：所有脚本注入 `HF_HUB_OFFLINE=1`，适用于隔离的 HPC 集群。

---

## 📊 详细结果

### 逐步评测（N=4 样本/任务，max_tokens=4096）

| 实验 | Step | Pass@1 | 泛化 Pass@1 | 状态覆盖 | 工具熵 | 备注 |
|------|------|--------|-------------|---------|--------|------|
| Vanilla | 250 | 0.175 | 0.071 | 18% | 1.82 | 探索塌缩基线 |
| IG-Fixed | 250 | 0.195 | 0.092 | 32% | 2.15 | 固定权重 IG |
| IG-Combo | 250 | 0.218 | 0.108 | 38% | 2.45 | 加工具组合 |
| **IG-Full** | **250** | **0.248** | **0.128** | **45%** | **2.73** | **最优检查点** |

### 假设验证

| 假设 | 状态 | 证据 |
|-----------|--------|----------|
| H1: IG 信号促进状态覆盖 | ✅ 已验证 | 覆盖率 18% → 32% (IG-Fixed) |
| H2: 工具组合奖励提升多样性 | ✅ 已验证 | 工具熵 1.82 → 2.45 (IG-Combo) |
| H3: 课程学习平衡探索-利用 | ✅ 已验证 | IG-Full 超过固定权重版本 |
| H4: G-Norm 防止长轨迹主导 | ✅ 已验证 | 平均路径长 8.2 → 5.9 |
| H5: IG-Full > IG-Fixed > Vanilla | ✅ 已验证 | Pass@1: 0.248 > 0.195 > 0.175 |

---

## 🏗️ 项目结构

```
📦 ig-grpo-agent/
├── ⚙️ configs/                 # Hydra YAML 配置
│   ├── base.yaml
│   ├── vanilla.yaml
│   ├── ig_fixed.yaml
│   ├── ig_combo.yaml
│   ├── ig_full.yaml
│   └── eval/
├── 💻 src/                     # 核心源码
│   ├── 🌍 envs/                # τ-bench 包装器与 IG 组件
│   │   ├── tau_bench_wrapper.py
│   │   ├── async_entropy_estimator.py
│   │   ├── hierarchical_coverage.py
│   │   ├── tool_dependency_graph.py
│   │   └── sustained_exploration.py
│   ├── 🧠 models/
│   │   └── vllm_policy.py
│   ├── 📊 evaluation/
│   │   ├── entropy_at_k_eval.py
│   │   └── pass_k_eval.py
│   └── 🎓 training/
│       └── grpo_trainer.py
├── 📜 scripts/
│   ├── 🚀 train/grpo/          # GRPO 训练脚本
│   ├── 📈 eval/                # 评测脚本
│   └── 🖥️ vllm_server/         # vLLM 服务脚本
├── 📚 docs/
│   └── 🔬 exploration/
│       ├── exploration_diagnosis.md
│       ├── jig_analysis.md
│       └── ablation_report.md
├── 🧪 experiments/             # 检查点、评测输出
├── 📄 requirements.txt
└── 🔨 setup.sh                 # 一键环境搭建
```

---

## 🚀 快速开始

### 1. 环境搭建

```bash
# 一键搭建（conda + PyTorch 2.7 + CUDA 12.6 + 依赖）
bash setup.sh
conda activate iggrpo
cd ig-grpo-agent

# 或手动安装：
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
cd ../tau-bench && pip install -e .
cd ../verl && pip install -e .
cd ig-grpo-agent
```

### 2. 启动 vLLM 服务器

```bash
# GPU 0: 7B 策略模型
bash scripts/vllm_server/7b.sh &

# GPU 1: 72B-AWQ 用户模拟器
bash scripts/vllm_server/72b.sh &
```

### 3. 训练模型

```bash
# 示例：完整 IG-GRPO 方案
cd scripts/train/grpo
bash run_exp3_ig_full.sh

# 或：Vanilla GRPO 基线
bash run_vanilla.sh
```

### 4. 评测

```bash
# 自动评测 step 200/250/300 检查点
cd scripts/eval
bash eval_pass_k.sh ig_full
bash eval_entropy_k.sh ig_full
```

> **硬件**：2×A800（80GB）。GPU 0 运行 7B 策略 vLLM；GPU 1 运行 72B-AWQ 用户模拟器 vLLM。

---

## 📚 文档

| 📄 文档 | 📝 内容 |
|----------|---------|
| [`docs/exploration/exploration_diagnosis.md`](docs/exploration/exploration_diagnosis.md) | **主报告**：探索塌缩诊断、训练曲线、机制分析 |
| [`docs/exploration/jig_analysis.md`](docs/exploration/jig_analysis.md) | JIG 理论分析、公式推导、实现细节 |
| [`docs/exploration/ablation_report.md`](docs/exploration/ablation_report.md) | 完整消融实验报告 |

---

## 🛠️ 技术栈

- **训练框架**: [veRL](https://github.com/volcengine/verl) 0.6.1 (FSDP + vLLM V1)
- **策略模型**: Qwen2.5-7B-Instruct
- **用户模拟器**: Qwen2.5-72B-Instruct-AWQ
- **评测基准**: [τ-bench](https://github.com/sierra-research/tau-bench) retail (50 任务, 40 工具)
- **推理引擎**: vLLM V1 with tool-call parsing (Hermes)
- **状态追踪**: pybloom-live (ScalableBloomFilter)

---

## 🙏 致谢

- [veRL](https://github.com/volcengine/verl) 开源 RL 训练框架
- [τ-bench](https://github.com/sierra-research/tau-bench) 挑战性多工具智能体评测基准
- [Qwen](https://github.com/QwenLM/Qwen) 系列模型提供的强大基座策略

---

## 📄 许可证

MIT License

---

> **为什么重要**：大多数 RLHF/RLAIF 工作聚焦单轮问答或代码生成。本项目 tackle 更难的问题 —— **多轮、多工具、部分可观测的对话智能体** —— 在这里 vanilla GRPO 会发生探索塌缩。IG-GRPO 提供了一条有原理支撑、轻量且可解释的通往高效探索的路径，无需昂贵的学习式奖励模型。
