#!/bin/bash
# Qwen 1.5B Accuracy Sweep (Layer/Alpha)
# Target: Qwen/Qwen2.5-1.5B-Instruct
# Prompt: accuracy
# Layers: 12-16 (Centered on 14)
# Alphas: 1.0, 2.0, 3.0, 4.0

set -e

# Force usage of local cache
export HF_HOME="/home/deeplearning01/.cache/huggingface"

MODEL="qwen-1.5b"
PROMPT_TYPE="accuracy"
VECTOR_METHOD="response"
TRAIN_STEPS=200
EVAL_STEPS=400
SEED=0

# Sweep Parameters
LAYERS=(12 13 14 15 16)
ALPHAS=(1.0 2.0 3.0 4.0)

RESULTS_DIR="results/qwen1.5b_accuracy_sweep"
mkdir -p $RESULTS_DIR

echo "========================================"
echo "Qwen 1.5B Accuracy Sweep"
echo "========================================"
echo "Model: $MODEL"
echo "Prompt: $PROMPT_TYPE"
echo "Layers: ${LAYERS[*]}"
echo "Alphas: ${ALPHAS[*]}"
echo "Output: $RESULTS_DIR"
echo "========================================"

# 1. Run Baseline (Once)
BASE_FILE="$RESULTS_DIR/${MODEL}_baseline.jsonl"
if [ ! -f "$BASE_FILE" ]; then
    echo ""
    echo ">>> Running Baseline"
    python src/miniwob_steer.py \
        --model "$MODEL" \
        --layer 14 \
        --base-only \
        --eval-steps $EVAL_STEPS \
        --seed $SEED \
        --out "$BASE_FILE"
else
    echo ">>> Baseline exists: $BASE_FILE"
fi

# 2. Run Sweep
for LAYER in "${LAYERS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        OUT_FILE="$RESULTS_DIR/${MODEL}_${PROMPT_TYPE}_L${LAYER}_a${ALPHA}.jsonl"
        
        if [ -f "$OUT_FILE" ]; then
            echo "  Skipping L${LAYER} a${ALPHA} (exists)"
            continue
        fi
        
        echo ""
        echo ">>> Running: Layer $LAYER, Alpha $ALPHA"
        python src/miniwob_steer.py \
            --model "$MODEL" \
            --layer $LAYER \
            --coeff $ALPHA \
            --prompt-type $PROMPT_TYPE \
            --vector-method $VECTOR_METHOD \
            --train-steps $TRAIN_STEPS \
            --eval-steps $EVAL_STEPS \
            --seed $SEED \
            --out "$OUT_FILE"
    done
done

echo "Sweep Complete."
