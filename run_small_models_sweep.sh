#!/bin/bash
# Small Models Accuracy Sweep
# Models: smollm-360m, qwen-coder-0.5b, tinyllama-1.1b, stablelm-1.6b, opt-iml-1.3b, gemma-1b
# Layers: Center +/- 2
# Alphas: 1.0 2.0 3.0 4.0

set -e
# Use the default cache location where we manually downloaded models
export HF_HOME="/home/deeplearning01/.cache/huggingface"

# Define models and their center layers
declare -A CENTER_LAYERS
CENTER_LAYERS["smollm-360m"]=16
CENTER_LAYERS["qwen-coder-0.5b"]=11
CENTER_LAYERS["tinyllama-1.1b"]=11
CENTER_LAYERS["stablelm-1.6b"]=12
CENTER_LAYERS["opt-iml-1.3b"]=12
CENTER_LAYERS["gemma-1b"]=13

MODELS=("smollm-360m" "qwen-coder-0.5b" "tinyllama-1.1b" "stablelm-1.6b" "opt-iml-1.3b" "gemma-1b")
PROMPT_TYPE="accuracy"
VECTOR_METHOD="response"
TRAIN_STEPS=200
EVAL_STEPS=400
SEED=0
ALPHAS=(1.0 2.0 3.0 4.0)

mkdir -p results/small_models_sweep

for MODEL in "${MODELS[@]}"; do
    CENTER=${CENTER_LAYERS[$MODEL]}
    # Range: Center-2 to Center+2
    LAYERS=($(seq $((CENTER - 2)) $((CENTER + 2))))
    
    echo "========================================"
    echo "Sweeping $MODEL (Center L$CENTER)"
    echo "Layers: ${LAYERS[*]}"
    echo "========================================"
    
    # 1. Baseline
    BASE_FILE="results/small_models_sweep/${MODEL}_baseline.jsonl"
    if [ ! -f "$BASE_FILE" ]; then
        echo ">>> Running Baseline for $MODEL"
        python src/miniwob_steer.py \
            --model "$MODEL" \
            --layer $CENTER \
            --base-only \
            --eval-steps $EVAL_STEPS \
            --seed $SEED \
            --out "$BASE_FILE" || echo "Baseline failed for $MODEL"
    fi
    
    # 2. Sweep
    for LAYER in "${LAYERS[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            OUT_FILE="results/small_models_sweep/${MODEL}_${PROMPT_TYPE}_L${LAYER}_a${ALPHA}.jsonl"
            
            if [ -f "$OUT_FILE" ]; then
                echo "  Skipping L${LAYER} a${ALPHA} (exists)"
                continue
            fi
            
            echo ">>> Running: $MODEL L$LAYER a$ALPHA"
            python src/miniwob_steer.py \
                --model "$MODEL" \
                --layer $LAYER \
                --coeff $ALPHA \
                --prompt-type $PROMPT_TYPE \
                --vector-method $VECTOR_METHOD \
                --train-steps $TRAIN_STEPS \
                --eval-steps $EVAL_STEPS \
                --seed $SEED \
                --out "$OUT_FILE" || echo "Run failed for $MODEL L$LAYER"
        done
    done
done

echo "Small Models Sweep Complete."