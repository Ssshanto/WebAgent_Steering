#!/bin/bash
# Experiment 9: Prompt Strategy Sweep with Optimal Alphas
#
# Testing all 16 prompting strategies across two distinct configurations:
# 1. Stable Config: Layer 13, Alpha 2.0
# 2. Performance Config: Layer 13, Alpha 4.0
#
# Vector Method: Response (Best verified)

set -e

MODEL_SIZE="0.5b"
LAYER=13
TASKS="all"
TRAIN_STEPS=200
EVAL_STEPS=400
VECTOR_METHOD="response"
SEED=0

# Prompt Tiers
ORIGINAL="accuracy verification format"
TIER1="refined_accuracy attention confidence format_accuracy"
TIER2="element_selection attribute_matching task_compliance deliberation"
TIER3="minimalism goal_directed self_correction dom_reading"
TIER4="composite_1 composite_2 composite_3"

ALL_PROMPTS="$ORIGINAL $TIER1 $TIER2 $TIER3 $TIER4"

RESULTS_DIR="results/exp9_prompt_strategies"
mkdir -p $RESULTS_DIR

echo "========================================"
echo "Experiment 9: Prompt Strategy Sweep"
echo "========================================"
echo "Model: $MODEL_SIZE"
echo "Layer: $LAYER"
echo "Vector Method: $VECTOR_METHOD"
echo "Seed: $SEED"
echo "========================================"

run_sweep() {
    ALPHA=$1
    CONFIG_NAME=$2
    
    echo ""
    echo ">>> Starting Sweep for $CONFIG_NAME (Alpha $ALPHA)"
    echo ""
    
    for PROMPT in $ALL_PROMPTS; do
        OUT_FILE="$RESULTS_DIR/${PROMPT}_a${ALPHA}.jsonl"
        
        if [ -f "$OUT_FILE" ]; then
            echo "  Skipping $PROMPT (exists)"
            continue
        fi
        
        echo "  Running: $PROMPT"
        python src/miniwob_steer.py \
            --model-size $MODEL_SIZE \
            --layer $LAYER \
            --coeff $ALPHA \
            --tasks $TASKS \
            --prompt-type $PROMPT \
            --vector-method $VECTOR_METHOD \
            --train-steps $TRAIN_STEPS \
            --eval-steps $EVAL_STEPS \
            --seed $SEED \
            --out "$OUT_FILE"
    done
}

# Run 1: Stable Configuration
run_sweep 2.0 "Stable Config"

# Run 2: High-Performance Configuration
run_sweep 4.0 "High-Performance Config"

echo ""
echo "========================================"
echo "Experiment 9 Complete"
echo "========================================"
