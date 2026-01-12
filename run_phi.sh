#!/bin/bash
# Phi-3.5 (3.8B) Accuracy Sweep
# Target: microsoft/Phi-3.5-mini-instruct
# Layers: 14-18 (Centered on 16)

set -e
export HF_HOME="/home/deeplearning01/.cache/huggingface"

MODEL="phi-3.8b"
PROMPT_TYPE="accuracy"
VECTOR_METHOD="response"
TRAIN_STEPS=200
EVAL_STEPS=400
SEED=0

LAYERS=(14 15 16 17 18)
ALPHAS=(1.0 2.0 3.0 4.0)

RESULTS_DIR="results/${MODEL}_accuracy_sweep"
mkdir -p $RESULTS_DIR

echo "========================================"
echo "Phi-3.5 Accuracy Sweep"
echo "========================================"

# Baseline
BASE_FILE="$RESULTS_DIR/${MODEL}_baseline.jsonl"
if [ ! -f "$BASE_FILE" ]; then
    echo ">>> Running Baseline"
    python src/miniwob_steer.py --model "$MODEL" --layer 16 --base-only --eval-steps $EVAL_STEPS --seed $SEED --out "$BASE_FILE"
fi

# Sweep
for LAYER in "${LAYERS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        OUT_FILE="$RESULTS_DIR/${MODEL}_${PROMPT_TYPE}_L${LAYER}_a${ALPHA}.jsonl"
        if [ ! -f "$OUT_FILE" ]; then
            echo ">>> Running: Layer $LAYER, Alpha $ALPHA"
            python src/miniwob_steer.py --model "$MODEL" --layer $LAYER --coeff $ALPHA --prompt-type $PROMPT_TYPE --vector-method $VECTOR_METHOD --train-steps $TRAIN_STEPS --eval-steps $EVAL_STEPS --seed $SEED --out "$OUT_FILE"
        fi
    done
done
echo "Phi Sweep Complete."
