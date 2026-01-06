#!/bin/bash
# Experiment 8b: Supplementary Layer Sweep
# Testing earlier layers with the winning Alpha=4.0

set -e

MODEL_SIZE="0.5b"
TASKS="all"
TRAIN_STEPS=200
EVAL_STEPS=400
PROMPT_TYPE="accuracy"
VECTOR_METHOD="response"

LAYERS=(10 11 12)
ALPHA=4.0
SEEDS=(0 1 2 3 4)

RESULTS_DIR="results/exp8_robust_opt"
mkdir -p $RESULTS_DIR

echo "========================================"
echo "Experiment 8b: Supplementary Sweep"
echo "========================================"
echo "Layers: ${LAYERS[*]}"
echo "Alpha: $ALPHA"
echo "Seeds: 0-4 (5 total)"
echo "========================================"

for LAYER in "${LAYERS[@]}"; do
    echo ">>> Testing Layer $LAYER, Alpha $ALPHA"
    for SEED in "${SEEDS[@]}"; do
        OUT_FILE="$RESULTS_DIR/L${LAYER}_a${ALPHA}_s${SEED}.jsonl"
        
        if [ -f "$OUT_FILE" ]; then
            echo "  Seed $SEED: Skipping (exists)"
            continue
        fi

        python src/miniwob_steer.py \
            --model-size $MODEL_SIZE \
            --layer $LAYER \
            --coeff $ALPHA \
            --tasks $TASKS \
            --prompt-type $PROMPT_TYPE \
            --vector-method $VECTOR_METHOD \
            --train-steps $TRAIN_STEPS \
            --eval-steps $EVAL_STEPS \
            --seed $SEED \
            --out "$OUT_FILE"
        
        echo "  Seed $SEED: Done"
    done
done
