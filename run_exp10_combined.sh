#!/bin/bash
# Experiment 10: Combined Vector Strategy
# Testing the "Super Vector" (Format Accuracy + Composite 1)

set -e

MODEL_SIZE="0.5b"
LAYER=13
COEFF=4.0
TASKS="all"
TRAIN_STEPS=200
EVAL_STEPS=400
VECTOR_METHOD="response"
SEED=0

RESULTS_DIR="results/exp10_combined"
mkdir -p $RESULTS_DIR

echo "========================================"
echo "Experiment 10: Combined Vector Strategy"
echo "========================================"
echo "Layer: $LAYER"
echo "Alpha: $COEFF"
echo "Prompt: combined (format_accuracy + composite_1)"
echo "========================================"

python src/miniwob_steer.py \
    --model $MODEL_SIZE \
    --layer $LAYER \
    --coeff $COEFF \
    --tasks $TASKS \
    --prompt-type combined \
    --vector-method $VECTOR_METHOD \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --seed $SEED \
    --out "$RESULTS_DIR/combined.jsonl"

echo "Done."
