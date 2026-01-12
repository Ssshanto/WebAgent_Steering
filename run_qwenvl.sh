#!/bin/bash
# Qwen2-VL (2B) Accuracy Sweep
# Target: Qwen/Qwen2-VL-2B-Instruct
# Layers: 12-16 (Centered on 14)
# Note: Uses --vlm flag for vision input

set -e
export HF_HOME="/home/deeplearning01/.cache/huggingface"

MODEL="qwen-vl-2b"
PROMPT_TYPE="accuracy"
VECTOR_METHOD="response"
TRAIN_STEPS=200
EVAL_STEPS=400
SEED=0

LAYERS=(12 13 14 15 16)
ALPHAS=(1.0 2.0 3.0 4.0)

RESULTS_DIR="results/${MODEL}_accuracy_sweep"
mkdir -p $RESULTS_DIR

echo "========================================"
echo "Qwen2-VL 2B Accuracy Sweep"
echo "========================================"

# Baseline
BASE_FILE="$RESULTS_DIR/${MODEL}_baseline.jsonl"
if [ ! -f "$BASE_FILE" ]; then
    echo ">>> Running Baseline"
    python src/miniwob_steer.py --model "$MODEL" --layer 14 --base-only --eval-steps $EVAL_STEPS --seed $SEED --vlm --out "$BASE_FILE"
fi

# Sweep
for LAYER in "${LAYERS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        OUT_FILE="$RESULTS_DIR/${MODEL}_${PROMPT_TYPE}_L${LAYER}_a${ALPHA}.jsonl"
        if [ ! -f "$OUT_FILE" ]; then
            echo ">>> Running: Layer $LAYER, Alpha $ALPHA"
            python src/miniwob_steer.py --model "$MODEL" --layer $LAYER --coeff $ALPHA --prompt-type $PROMPT_TYPE --vector-method $VECTOR_METHOD --train-steps $TRAIN_STEPS --eval-steps $EVAL_STEPS --seed $SEED --vlm --out "$OUT_FILE"
        fi
    done
done
echo "Qwen-VL Sweep Complete."
