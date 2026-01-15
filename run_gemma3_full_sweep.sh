#!/bin/bash
# Gemma 3 1B Full Layer Sweep (L1-L26)
# Alpha: 2.0
# Prompt: accuracy

set -e
export HF_HOME="/home/deeplearning01/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=1
# Ensure token is available for gated repo checks
export HF_TOKEN=$(cat /mnt/code/huggingface/token)

MODEL="gemma-1b"
PROMPT_TYPE="accuracy"
ALPHA=2.0
TRAIN_STEPS=200
EVAL_STEPS=400
SEED=0

RESULTS_DIR="results/gemma3_full_sweep"
mkdir -p $RESULTS_DIR

# Sweep through all 26 layers
for LAYER in $(seq 1 26); do
    OUT_FILE="$RESULTS_DIR/${MODEL}_L${LAYER}_a${ALPHA}.jsonl"
    
    if [ -f "$OUT_FILE" ]; then
        echo ">>> Skipping L${LAYER} (exists)"
        continue
    fi
    
    echo "========================================"
    echo "Running: Layer $LAYER, Alpha $ALPHA"
    echo "========================================"
    
    python3 src/miniwob_steer.py \
        --model "$MODEL" \
        --layer "$LAYER" \
        --coeff "$ALPHA" \
        --prompt-type "$PROMPT_TYPE" \
        --train-steps "$TRAIN_STEPS" \
        --eval-steps "$EVAL_STEPS" \
        --seed "$SEED" \
        --out "$OUT_FILE"
done

echo "Gemma 3 Full Sweep Complete."
