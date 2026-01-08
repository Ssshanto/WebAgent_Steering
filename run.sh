#!/bin/bash
# Experiment 11: Optimized Prompt & Layer Sweep
# Testing the "Gold Set" of 7 prompts across Layers {11..15} and Alphas {1..4}

set -e

MODEL_SIZE="0.5b"
TASKS="all"
TRAIN_STEPS=200
EVAL_STEPS=400
VECTOR_METHOD="response"
SEED=0

# The Pruned "Gold Set"
PROMPTS="format_accuracy accuracy deliberation composite_1 confidence dom_reading verification"

LAYERS=(11 12 13 14 15)
ALPHAS=(1.0 2.0 3.0 4.0)

RESULTS_DIR="results/exp11_gold_sweep"
mkdir -p $RESULTS_DIR

echo "========================================"
echo "Experiment 11: Optimized Sweep"
echo "========================================"
echo "Prompts: $PROMPTS"
echo "Layers: ${LAYERS[*]}"
echo "Alphas: ${ALPHAS[*]}"
echo "========================================"

for LAYER in "${LAYERS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        echo ""
        echo ">>> Testing L$LAYER, a$ALPHA"
        
        for PROMPT in $PROMPTS; do
            OUT_FILE="$RESULTS_DIR/${PROMPT}_L${LAYER}_a${ALPHA}.jsonl"
            
            if [ -f "$OUT_FILE" ]; then
                echo "  Skipping $PROMPT (exists)"
                continue
            fi
            
            echo "  Running: $PROMPT"
            python src/miniwob_steer.py \
                --model $MODEL_SIZE \
                --layer $LAYER \
                --coeff $ALPHA \
                --prompt-type $PROMPT \
                --vector-method $VECTOR_METHOD \
                --train-steps $TRAIN_STEPS \
                --eval-steps $EVAL_STEPS \
                --seed $SEED \
                --out "$OUT_FILE"
        done
    done
done

echo "Sweep Complete."
