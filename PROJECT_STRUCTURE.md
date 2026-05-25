# IG-GRPO Agent 项目结构

## 概述

这是一个基于信息增益（Information Gain）的高效探索框架，用于多工具 Agent 的 GRPO 训练。

## 核心组件

### 1. JIG 核心组件 (`src/envs/jig_components.py`)
- `JointInformationGain` - 联合互信息计算器
- `HierarchicalCoverageTracker` - 三层 Bloom Filter 状态追踪
- `SustainedExplorationBonus` - 探索持续性保证
- `CurriculumScheduler` - 课程学习调度器

### 2. veRL 集成 (`src/envs/tau_bench_interaction_ig.py`)
- 继承 `verl.interactions.base.BaseInteraction`
- 支持 `reward_mode="jig"` 模式
- 集成 JIG 奖励计算到 veRL 训练流程

### 3. 辅助组件
- `async_entropy_estimator.py` - 异步熵计算器
- `tool_dependency_graph.py` - 工具依赖图
- `tau_bench_wrapper.py` - τ-bench 环境包装器

## 项目结构

```
ig-grpo-agent/
├── configs/
│   ├── train/
│   │   ├── base.yaml
│   │   ├── vanilla.yaml
│   │   ├── ig_fixed.yaml
│   │   ├── ig_combo.yaml
│   │   ├── ig_full.yaml
│   │   └── grpo/
│   │       └── ig_full_veRL.yaml    # veRL 配置
│   └── interaction_config/
│       └── tau_bench_airline_jig.yaml
├── src/
│   ├── envs/
│   │   ├── jig_components.py         # 核心组件
│   │   ├── tau_bench_interaction_ig.py  # veRL 集成
│   │   ├── async_entropy_estimator.py
│   │   ├── tool_dependency_graph.py
│   │   └── tau_bench_wrapper.py
│   ├── models/
│   │   └── vllm_policy_eval.py      # 评测用
│   ├── evaluation/
│   │   ├── entropy_at_k_eval.py
│   │   └── pass_k_eval.py
│   └── training/
│       └── sft_dataset.py
├── scripts/
│   ├── train/grpo/
│   │   └── run_ig_full_veRL.sh     # 训练脚本
│   └── eval/
└── requirements.txt
```

## 使用方法

### 1. 安装依赖
```bash
cd ig-grpo-agent
pip install -r requirements.txt
pip install -e ../tau-bench
pip install -e ../verl
```

### 2. 启动 vLLM 服务
```bash
# GPU 0: 7B 策略模型
bash scripts/vllm_server/7b.sh &

# GPU 1: 72B 用户模拟器
bash scripts/vllm_server/72b.sh &
```

### 3. 训练
```bash
bash scripts/train/grpo/run_ig_full_veRL.sh
```

### 4. 评测
```bash
# Pass@K 评测
bash scripts/eval/eval_pass_k.sh

# Entropy@K 评测
bash scripts/eval/eval_entropy_k.sh
```

## 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| alpha_state | 0.3 | 状态新颖性权重 |
| alpha_tool | 0.4 | 工具组合新颖性权重 |
| alpha_transfer | 0.2 | 工具转移新颖性权重 |
| total_steps | 300 | 总训练步数 |
| curriculum_beta | 0.7 | 课程学习衰减指数 |
| exploration_window | 50 | 探索持续性窗口 |

## 预期效果

| 指标 | Vanilla | IG-GRPO | 提升 |
|------|---------|---------|------|
| Pass@1 | 0.175 | 0.248 | +42% |
| 状态覆盖率 | 18% | 45% | +150% |
| 工具多样性熵 | 1.82 | 2.73 | +50% |
