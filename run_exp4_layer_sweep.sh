#!/bin/bash
# Experiment 4: Layer sweep with multi-layer steering on medium-difficulty tasks
# Usage: ./run_exp4_layer_sweep.sh [experiment_name]

set -e

EXPERIMENT_NAME="${1:-exp4_layer_sweep}"
LAYERS=(15 18 22 25)
TASKS="medium"
TRAIN_STEPS=200
EVAL_STEPS=200
MODEL_SIZE="3b"
COEFF=1.0

echo "========================================"
echo "Running Experiment: $EXPERIMENT_NAME"
echo "Layers to sweep: ${LAYERS[*]}"
echo "Tasks: $TASKS | Coeff: $COEFF"
echo "========================================"

# Create results directory
mkdir -p results

for LAYER in "${LAYERS[@]}"; do
    OUT_FILE="results/${EXPERIMENT_NAME}_layer${LAYER}.jsonl"
    echo ""
    echo ">>> Running sweep at starting layer = $LAYER (multi-layer steering)"
    echo ">>> Output: $OUT_FILE"

    python src/miniwob_steer.py \
        --model-size $MODEL_SIZE \
        --layer $LAYER \
        --coeff $COEFF \
        --tasks $TASKS \
        --steer-all-layers \
        --train-steps $TRAIN_STEPS \
        --eval-steps $EVAL_STEPS \
        --out "$OUT_FILE"

    echo ">>> Completed layer = $LAYER"
done

echo ""
echo "========================================"
echo "Experiment Complete: $EXPERIMENT_NAME"
echo "Results saved to: results/"
echo "========================================"

# Summary
echo ""
echo "Summary:"
for LAYER in "${LAYERS[@]}"; do
    OUT_FILE="results/${EXPERIMENT_NAME}_layer${LAYER}.jsonl"
    if [ -f "$OUT_FILE" ]; then
        echo "  layer=$LAYER: $(wc -l < "$OUT_FILE") episodes"
    fi
done