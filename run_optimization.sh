#!/bin/bash
# Hyperparameter Optimization: Find Best Layer/Coefficient Combination
#
# Objective: Sweep layer and coefficient to find optimal steering configuration
# for the 'accuracy' prompt on Qwen 2.5 0.5B model.
#
# Based on current knowledge:
# - Best reported: Layer 13, α=4.0 (+17.5% accuracy)
# - Search space: Layers 12-15, Coefficients 2.0-5.0
#
# Usage: ./run_optimization.sh

set -e

MODEL_SIZE="0.5b"
PROMPT_TYPE="accuracy"
VECTOR_METHOD="response"
TRAIN_STEPS=200
EVAL_STEPS=400
TASKS="all"
SEED=0

mkdir -p results

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           HYPERPARAMETER OPTIMIZATION: LAYER × COEFFICIENT          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Model: Qwen 2.5 ${MODEL_SIZE}"
echo "  Prompt: ${PROMPT_TYPE}"
echo "  Vector Method: ${VECTOR_METHOD}"
echo "  Tasks: ${TASKS} (25 tasks)"
echo "  Seed: ${SEED}"
echo ""
echo "Search Space:"
echo "  Layers: 12, 13, 14, 15"
echo "  Coefficients: 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0"
echo ""
echo "Total Experiments: 28 (4 layers × 7 coefficients)"
echo "Estimated Time: ~28 hours"
echo ""
read -p "Press Enter to start optimization... " -t 10 || echo ""
echo ""

# Layer sweep with different coefficients
for LAYER in 12 13 14 15; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing Layer ${LAYER}..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for COEFF in 1.0 2.0 2.5 3.0 3.5 4.0 4.5 5.0; do
        OUTPUT_FILE="results/L${LAYER}_a${COEFF}_s${SEED}.jsonl"
        
        if [ -f "$OUTPUT_FILE" ]; then
            echo "  [SKIP] L${LAYER} α=${COEFF} (already exists)"
            continue
        fi
        
        echo "  [RUN] L${LAYER} α=${COEFF}..."
        
        python src/miniwob_steer.py \
            --model-size $MODEL_SIZE \
            --layer $LAYER \
            --coeff $COEFF \
            --vector-method $VECTOR_METHOD \
            --prompt-type $PROMPT_TYPE \
            --tasks $TASKS \
            --train-steps $TRAIN_STEPS \
            --eval-steps $EVAL_STEPS \
            --seed $SEED \
            --out "$OUTPUT_FILE"
        
        echo "  [DONE] Saved to $OUTPUT_FILE"
        echo ""
    done
    echo ""
done

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                   OPTIMIZATION COMPLETE                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: results/"
echo "Analyze with: python scripts/analyze_optimization.py"
