#!/bin/bash
# Experiment 5: 0.5B Model - Accuracy Prompt Phase
#
# Tests the "accuracy" prompt type across all previously tested parameter sets.

set -e

EXPERIMENT_NAME="${1:-exp5_0.5b_accuracy}"
MODEL_SIZE="0.5b"
TRAIN_STEPS=200
EVAL_STEPS=400
TASKS="all"

echo "========================================"
echo "Experiment 5: Accuracy Prompt Phase"
echo "========================================"
echo "Model: Qwen 2.5 0.5B (24 layers)"
echo "Tasks: $TASKS"
echo "========================================"

mkdir -p results

# 1: Single layer L14, α=1.0
echo ">>> Running Accuracy L14 c1.0"
python src/miniwob_steer.py \
    --model-size $MODEL_SIZE \
    --layer 14 \
    --coeff 1.0 \
    --tasks $TASKS \
    --prompt-type accuracy \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --out "results/${EXPERIMENT_NAME}_L14_c1.jsonl"

# 2: Single layer L14, α=2.0
echo ">>> Running Accuracy L14 c2.0"
python src/miniwob_steer.py \
    --model-size $MODEL_SIZE \
    --layer 14 \
    --coeff 2.0 \
    --tasks $TASKS \
    --prompt-type accuracy \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --out "results/${EXPERIMENT_NAME}_L14_c2.jsonl"

# 3: Single layer L14, α=3.0
echo ">>> Running Accuracy L14 c3.0"
python src/miniwob_steer.py \
    --model-size $MODEL_SIZE \
    --layer 14 \
    --coeff 3.0 \
    --tasks $TASKS \
    --prompt-type accuracy \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --out "results/${EXPERIMENT_NAME}_L14_c3.jsonl"

# 4: Multi-layer L10-L23, α=2.0
echo ">>> Running Accuracy L10 Multi c2.0"
python src/miniwob_steer.py \
    --model-size $MODEL_SIZE \
    --layer 10 \
    --coeff 2.0 \
    --tasks $TASKS \
    --prompt-type accuracy \
    --steer-all-layers \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --out "results/${EXPERIMENT_NAME}_L10_multi_c2.jsonl"

echo "========================================"
echo "Accuracy Phase Complete"
echo "========================================"
