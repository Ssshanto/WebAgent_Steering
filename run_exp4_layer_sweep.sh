#!/bin/bash
# Experiment 4: Layer sweep with multi-layer steering on medium-difficulty tasks
#
# Key changes from Exp 3:
# - Medium-difficulty tasks (54-82% base) instead of high-potential (89.5% ceiling)
# - Multi-layer steering (--steer-all-layers) from starting layer onwards
# - Layer sweep: {15, 18, 22, 25} to find optimal intervention point
#
# Usage: ./run_exp4_layer_sweep.sh [experiment_name]

set -e

EXPERIMENT_NAME="${1:-exp4_layer_sweep}"
COEFF=2.0
TASKS="medium"
TRAIN_STEPS=200
EVAL_STEPS=200
MODEL_SIZE="3b"

# Starting layers to sweep (multi-layer steering from this layer onwards)
LAYERS=(15 18 22 25)

echo "========================================"
echo "Experiment 4: Layer Sweep with Multi-Layer Steering"
echo "========================================"
echo "Tasks: $TASKS (medium-difficulty)"
echo "Coefficient: $COEFF"
echo "Starting layers: ${LAYERS[*]}"
echo "Mode: Multi-layer (from starting layer to L35)"
echo "========================================"

# Create results directory
mkdir -p results

for LAYER in "${LAYERS[@]}"; do
    OUT_FILE="results/${EXPERIMENT_NAME}_layer${LAYER}.jsonl"
    echo ""
    echo ">>> Running with starting layer = $LAYER (steering L${LAYER}-L35)"
    echo ">>> Output: $OUT_FILE"

    python src/miniwob_steer.py \
        --model-size $MODEL_SIZE \
        --layer $LAYER \
        --coeff $COEFF \
        --tasks $TASKS \
        --train-steps $TRAIN_STEPS \
        --eval-steps $EVAL_STEPS \
        --steer-all-layers \
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
        EPISODES=$(wc -l < "$OUT_FILE")
        echo "  layer=$LAYER: $EPISODES episodes"
    fi
done

echo ""
echo "To analyze results:"
echo "  python -c \"import json; ..."
echo "  # Or use the analysis script"
