#!/bin/bash
# Entropy@K 评测脚本
# 评测检查点的探索质量

set -e

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# 参数
EXPERIMENT_NAME=${1:-ig_full}
CHECKPOINT_DIR=${2:-outputs/ig_full/checkpoints}
OUTPUT_DIR=${3:-outputs/ig_full/eval}
K=4  # 采样次数

echo "======================================"
echo "Entropy@K Evaluation"
echo "======================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"
echo "K: $K"
echo "======================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 评测检查点列表
CHECKPOINTS=(50 100 150 200 250 300)

for ckpt in "${CHECKPOINTS[@]}"; do
    CKPT_PATH="$CHECKPOINT_DIR/checkpoint_$ckpt.pt"

    if [ ! -f "$CKPT_PATH" ]; then
        echo "Checkpoint $ckpt not found, skipping..."
        continue
    fi

    echo "Evaluating checkpoint $ckpt..."

    python -m src.evaluation.entropy_at_k_eval \
        --experiment_name="$EXPERIMENT_NAME" \
        --checkpoint_path="$CKPT_PATH" \
        --output_dir="$OUTPUT_DIR" \
        --k=$K \
        --state_space_size=1000000 \
        --run_pass_k=true \
        --use_anti_leakage=true \
        --n_clusters=5 \
        2>&1 | tee "$OUTPUT_DIR/entropy_k_${ckpt}.log"
done

# 聚合结果
echo "======================================"
echo "Aggregating results..."
python -m src.evaluation.aggregate_entropy_results \
    --output_dir="$OUTPUT_DIR" \
    --checkpoints="${CHECKPOINTS[@]}" \
    --output_file="$OUTPUT_DIR/entropy_aggregated.json"

echo "Evaluation completed! Results saved to $OUTPUT_DIR"
