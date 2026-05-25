#!/bin/bash
# Experiment 3: IG-Full 训练脚本
# 完整 IG-GRPO 方案：JIG + 课程学习 + 探索持续性

set -e

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Hydra 配置
CONFIG_FILE=configs/train/ig_full.yaml

# 训练参数
TOTAL_STEPS=300
ROLLOUT_BATCH_SIZE=64
GROUP_SIZE=8
LEARNING_RATE=1e-5

# IG-GRPO 完整参数
JIG_ALPHA_0=0.8
JIG_BETA=0.7
ALPHA_STATE=0.3
ALPHA_TOOL=0.4
ALPHA_TRANSFER=0.2
ALPHA_SUSTAINED=0.1
G_NORM_GAMMA=0.5
SUSTAINED_WINDOW=50
SUSTAINED_TREND_THRESHOLD=-0.01
REPEAT_PENALTY=0.3

# 输出目录
OUTPUT_DIR=outputs/ig_full_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

# 日志文件
LOG_FILE="$OUTPUT_DIR/train.log"

echo "======================================"
echo "IG-Full Training (Experiment 3)"
echo "======================================"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Steps: $TOTAL_STEPS"
echo "IG Alpha_0: $JIG_ALPHA_0 (curriculum)"
echo "IG Beta: $JIG_BETA"
echo "State Weight: $ALPHA_STATE"
echo "Tool Weight: $ALPHA_TOOL"
echo "Transfer Weight: $ALPHA_TRANSFER"
echo "Sustained Weight: $ALPHA_SUSTAINED"
echo "G-Norm Gamma: $G_NORM_GAMMA"
echo "======================================"

# 启动训练
python -m verl.trainers.main \
    --config-path="$PROJECT_ROOT/configs/train" \
    --config-name=ig_full \
    trainer.project=IG-GRPO \
    trainer.experiment_name=ig_full_retail \
    trainer.total_steps=$TOTAL_STEPS \
    trainer.rollout_batch_size=$ROLLOUT_BATCH_SIZE \
    trainer.group_size=$GROUP_SIZE \
    trainer.learning_rate=$LEARNING_RATE \
    trainer.use_jig_reward=true \
    trainer.jig_weight=$JIG_ALPHA_0 \
    trainer.jig_use_curriculum=true \
    trainer.jig_alpha_0=$JIG_ALPHA_0 \
    trainer.jig_beta=$JIG_BETA \
    trainer.use_coverage_tracker=true \
    trainer.use_sustained_exploration=true \
    trainer.use_dependency_graph=true \
    trainer.alpha_state=$ALPHA_STATE \
    trainer.alpha_tool=$ALPHA_TOOL \
    trainer.alpha_transfer=$ALPHA_TRANSFER \
    trainer.alpha_sustained=$ALPHA_SUSTAINED \
    trainer.g_norm_gamma=$G_NORM_GAMMA \
    trainer.sustained_window=$SUSTAINED_WINDOW \
    trainer.sustained_trend_threshold=$SUSTAINED_TREND_THRESHOLD \
    trainer.repeat_penalty=$REPEAT_PENALTY \
    trainer.repeat_window=3 \
    trainer.bypass_mode=true \
    trainer.fused_cross_entropy=true \
    model.policy_name_or_path=Qwen/Qwen2.5-7B-Instruct \
    model.vllm_tensor_parallel_size=2 \
    env.name=tau_bench \
    env.domain=retail \
    env.user_model=qwen2.5-72b-awq \
    env.user_base_url=http://localhost:8001/v1 \
    logging.log_interval=10 \
    logging.save_interval=50 \
    logging.eval_interval=50 \
    output.output_dir="$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "Training completed! Results saved to $OUTPUT_DIR"
echo "Checkpoints saved at steps: 50, 100, 150, 200, 250, 300"
