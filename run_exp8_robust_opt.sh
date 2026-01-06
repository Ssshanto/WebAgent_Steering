#!/bin/bash
# Experiment 8: Robust Layer/Alpha Optimization
# Runs combinations over 10 seeds to find the most statistically stable config.

set -e

MODEL_SIZE="0.5b"
TASKS="all"
TRAIN_STEPS=200
EVAL_STEPS=400
PROMPT_TYPE="accuracy"
VECTOR_METHOD="response"

LAYERS=(13 14 15)
ALPHAS=(2.0 3.0 4.0)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

RESULTS_DIR="results/exp8_robust_opt"
mkdir -p $RESULTS_DIR

echo "========================================"
echo "Experiment 8: Robust Optimization Sweep"
echo "========================================"
echo "Layers: ${LAYERS[*]}"
echo "Alphas: ${ALPHAS[*]}"
echo "Seeds: 0-9 (10 total)"
echo "========================================"

for LAYER in "${LAYERS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        echo ">>> Testing Layer $LAYER, Alpha $ALPHA"
        for SEED in "${SEEDS[@]}"; do
            OUT_FILE="$RESULTS_DIR/L${LAYER}_a${ALPHA}_s${SEED}.jsonl"
            
            # Skip if already exists
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
done

echo "========================================"
echo "Optimization Sweep Complete"
echo "========================================"
