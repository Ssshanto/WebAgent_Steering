#!/bin/bash
# Experiment 7: Prompt Engineering Sweep
# Tests new contrastive prompt pairs against current best (accuracy)
#
# Usage:
#   ./run_prompt_sweep.sh [tier]
#
# Tiers:
#   1 - High-confidence prompts (refined_accuracy, attention, confidence, format_accuracy)
#   2 - Medium-confidence prompts (element_selection, attribute_matching, task_compliance, deliberation)
#   3 - Exploratory prompts (minimalism, goal_directed, self_correction, dom_reading)
#   4 - Compositional prompts (composite_1, composite_2, composite_3)
#   all - Run all tiers (default)

set -e

# Configuration (current best from Exp 6)
LAYER=14
COEFF=3.0
MODEL="0.5b"
TRAIN=200
EVAL=400
VECTOR_METHOD="response"

# Output directory
RESULTS_DIR="results/exp7_prompt_sweep_c3.0"
mkdir -p $RESULTS_DIR

# Prompt tiers
ORIGINAL="accuracy verification format"
TIER1="refined_accuracy attention confidence format_accuracy"
TIER2="element_selection attribute_matching task_compliance deliberation"
TIER3="minimalism goal_directed self_correction dom_reading"
TIER4="composite_1 composite_2 composite_3"

# Select tier
TIER=${1:-all}

case $TIER in
    0)
        PROMPTS=$ORIGINAL
        echo "=== Running Original Prompts ==="
        ;;
    1)
        PROMPTS=$TIER1
        echo "=== Running Tier 1: High-Confidence Prompts ==="
        ;;
    2)
        PROMPTS=$TIER2
        echo "=== Running Tier 2: Medium-Confidence Prompts ==="
        ;;
    3)
        PROMPTS=$TIER3
        echo "=== Running Tier 3: Exploratory Prompts ==="
        ;;
    4)
        PROMPTS=$TIER4
        echo "=== Running Tier 4: Compositional Prompts ==="
        ;;
    all)
        PROMPTS="$ORIGINAL $TIER1 $TIER2 $TIER3 $TIER4"
        echo "=== Running ALL Tiers (Original + New) ==="
        ;;
    *)
        echo "Unknown tier: $TIER"
        echo "Usage: ./run_prompt_sweep.sh [0|1|2|3|4|all]"
        exit 1
        ;;
esac

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Layer: $LAYER"
echo "  Coefficient: $COEFF"
echo "  Vector Method: $VECTOR_METHOD"
echo "  Train Steps: $TRAIN"
echo "  Eval Steps: $EVAL"
echo ""

echo ""
echo ">>> Starting prompt sweep..."
echo ""

# Run each prompt
for PROMPT in $PROMPTS; do
    echo ">>> Running prompt type: $PROMPT"
    python src/miniwob_steer.py \
        --model-size $MODEL \
        --layer $LAYER \
        --coeff $COEFF \
        --prompt-type $PROMPT \
        --vector-method $VECTOR_METHOD \
        --train-steps $TRAIN \
        --eval-steps $EVAL \
        --out $RESULTS_DIR/${PROMPT}.jsonl
    echo ""
done

echo "=== Prompt Sweep Complete ==="
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_exp5.py $RESULTS_DIR/*.jsonl"
