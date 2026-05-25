#!/bin/bash
# Vanilla GRPO 基线训练脚本
# 不使用 IG 奖励

set -e

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Hydra 配置
CONFIG_FILE=configs/train/vanilla.yaml

# 训练参数
TOTAL_STEPS=300
ROLLOUT_BATCH_SIZE=64
GROUP_SIZE=8
LEARNING_RATE=1e-5

# 输出目录
OUTPUT_DIR=outputs/vanilla_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

# 日志文件
LOG_FILE="$OUTPUT_DIR/train.log"

echo "======================================"
echo "Vanilla GRPO Training"
echo "======================================"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Steps: $TOTAL_STEPS"
echo "Batch Size: $ROLLOUT_BATCH_SIZE"
echo "Group Size: $GROUP_SIZE"
echo "======================================"

# 启动训练
python -m verl.trainers.main \
    --config-path="$PROJECT_ROOT/configs/train" \
    --config-name=vanilla \
    trainer.project=IG-GRPO \
    trainer.experiment_name=vanilla_retail \
    trainer.total_steps=$TOTAL_STEPS \
    trainer.rollout_batch_size=$ROLLOUT_BATCH_SIZE \
    trainer.group_size=$GROUP_SIZE \
    trainer.learning_rate=$LEARNING_RATE \
    trainer.use_jig_reward=false \
    trainer.use_coverage_tracker=false \
    trainer.use_sustained_exploration=false \
    trainer.bypass_mode=false \
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
