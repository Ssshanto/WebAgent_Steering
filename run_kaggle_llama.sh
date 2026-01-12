#!/bin/bash
# Llama 1B Kaggle Sweep
# Target: meta-llama/Llama-3.2-1B-Instruct
# Prompt: accuracy
# Layers: 6-10 (Centered on 8)
# Alphas: 1.0, 2.0, 3.0, 4.0

set -e

# Kaggle-specific paths
export HF_HOME="/kaggle/working/cache/huggingface"
mkdir -p $HF_HOME

MODEL="llama-1b"
PROMPT_TYPE="accuracy"
VECTOR_METHOD="response"
TRAIN_STEPS=200
EVAL_STEPS=400
SEED=0

# Sweep Parameters
LAYERS=(6 7 8 9 10)
ALPHAS=(1.0 2.0 3.0 4.0)

# Output to Kaggle working directory
RESULTS_DIR="/kaggle/working/results/llama1b_accuracy_sweep"
mkdir -p $RESULTS_DIR

echo "========================================"
echo "Llama 1B Kaggle Sweep"
echo "========================================"
echo "Model: $MODEL"
echo "Prompt: $PROMPT_TYPE"
echo "Layers: ${LAYERS[*]}"
echo "Output: $RESULTS_DIR"
echo "========================================"

# 1. Run Baseline (Once)
BASE_FILE="$RESULTS_DIR/${MODEL}_baseline.jsonl"
if [ ! -f "$BASE_FILE" ]; then
    echo ">>> Running Baseline"
    python src/miniwob_steer.py \
        --model "$MODEL" \
        --layer 8 \
        --base-only \
        --eval-steps $EVAL_STEPS \
        --seed $SEED \
        --out "$BASE_FILE"
fi

# 2. Run Sweep
for LAYER in "${LAYERS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        OUT_FILE="$RESULTS_DIR/${MODEL}_${PROMPT_TYPE}_L${LAYER}_a${ALPHA}.jsonl"
        
        if [ -f "$OUT_FILE" ]; then
            continue
        fi
        
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
