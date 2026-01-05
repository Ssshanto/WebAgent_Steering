#!/bin/bash
# Experiment 3: Verification-focused steering with coefficient sweep
# Usage: ./run_experiment.sh [experiment_name]

set -e

EXPERIMENT_NAME="${1:-exp3_verification}"
LAYER=22
TASKS="high-potential"
TRAIN_STEPS=200
EVAL_STEPS=200
MODEL_SIZE="3b"

# Coefficient values to sweep
COEFFICIENTS=(1.0 2.0 3.0 5.0)

echo "========================================"
echo "Running Experiment: $EXPERIMENT_NAME"
echo "Layer: $LAYER | Tasks: $TASKS"
echo "Coefficients: ${COEFFICIENTS[*]}"
echo "========================================"

# Create results directory
mkdir -p results

for COEFF in "${COEFFICIENTS[@]}"; do
    OUT_FILE="results/${EXPERIMENT_NAME}_coeff${COEFF}.jsonl"
    echo ""
    echo ">>> Running with coefficient = $COEFF"
    echo ">>> Output: $OUT_FILE"

    python src/miniwob_steer.py \
        --model-size $MODEL_SIZE \
        --layer $LAYER \
        --coeff $COEFF \
        --tasks $TASKS \
        --train-steps $TRAIN_STEPS \
        --eval-steps $EVAL_STEPS \
        --out "$OUT_FILE"

    echo ">>> Completed coefficient = $COEFF"
done

echo ""
echo "========================================"
echo "Experiment Complete: $EXPERIMENT_NAME"
echo "Results saved to: results/"
echo "========================================"

# Summary
echo ""
echo "Summary:"
for COEFF in "${COEFFICIENTS[@]}"; do
    OUT_FILE="results/${EXPERIMENT_NAME}_coeff${COEFF}.jsonl"
    if [ -f "$OUT_FILE" ]; then
        echo "  coeff=$COEFF: $(wc -l < "$OUT_FILE") episodes"
    fi
done
