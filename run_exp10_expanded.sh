#!/bin/bash
# Experiment 10: Expanded Action Space Validation
#
# Objective: Test steering on complex interaction tasks beyond simple click/type
#
# Golden Config from Exp 9:
# - Layer: 13
# - Coefficient: 4.0
# - Vector Method: response
# - Prompt: accuracy
#
# New Task Categories:
# - Multi-selection: click-checkboxes, click-option
# - Dropdown: choose-list, choose-date
# - Semantic typing: enter-date, enter-time
# - Logic: guess-number
#
# Usage: ./run_exp10_expanded.sh

set -e

MODEL_SIZE="0.5b"
LAYER=13
COEFF=4.0
VECTOR_METHOD="response"
PROMPT_TYPE="accuracy"
TRAIN_STEPS=200
EVAL_STEPS=400
SEED=0

mkdir -p results

echo "=== Experiment 10: Expanded Action Space Validation ==="
echo "Configuration:"
echo "  Model: Qwen 2.5 ${MODEL_SIZE}"
echo "  Layer: ${LAYER}"
echo "  Coefficient: ${COEFF}"
echo "  Vector Method: ${VECTOR_METHOD}"
echo "  Prompt Type: ${PROMPT_TYPE}"
echo "  Tasks: expanded (7 complex interaction tasks)"
echo ""

# Run baseline + steering on expanded tasks
python src/miniwob_steer.py \
    --model-size $MODEL_SIZE \
    --layer $LAYER \
    --coeff $COEFF \
    --vector-method $VECTOR_METHOD \
    --prompt-type $PROMPT_TYPE \
    --tasks expanded \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --seed $SEED \
    --out "results/exp10_expanded.jsonl"

echo ""
echo "✓ Experiment 10 complete"
echo "Analyze with: python scripts/analyze_exp10.py"
