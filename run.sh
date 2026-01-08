#!/bin/bash
# Experiment: Full Prompt Sweep
# Testing all available prompt strategies with the optimal configuration.

set -e

# Optimal Config from Exp 8
MODEL_SIZE="0.5b"
LAYER=13
COEFF=4.0
VECTOR_METHOD="response"
TRAIN_STEPS=200
EVAL_STEPS=400
SEED=0

RESULTS_DIR="results/prompt_sweep_final"
mkdir -p $RESULTS_DIR

# All prompts from src/miniwob_steer.py
PROMPTS="accuracy verification format refined_accuracy attention confidence format_accuracy element_selection attribute_matching task_compliance deliberation minimalism goal_directed self_correction dom_reading composite_1 composite_2 composite_3"

echo "========================================"
echo "FULL PROMPT STRATEGY SWEEP"
echo "========================================"
echo "Config: L$LAYER, a$COEFF, $VECTOR_METHOD"
echo "========================================"

for PROMPT in $PROMPTS; do
    OUT_FILE="$RESULTS_DIR/${PROMPT}.jsonl"
    
    if [ -f "$OUT_FILE" ]; then
        echo "Skipping $PROMPT (exists)"
        continue
    fi
    
    echo ">>> Running Prompt: $PROMPT"
    
    python src/miniwob_steer.py \
        --model $MODEL_SIZE \
        --layer $LAYER \
        --coeff $COEFF \
        --prompt-type $PROMPT \
        --vector-method $VECTOR_METHOD \
        --train-steps $TRAIN_STEPS \
        --eval-steps $EVAL_STEPS \
        --seed $SEED \
        --out "$OUT_FILE"
        
    echo "----------------------------------------"
done

echo "Sweep Complete."