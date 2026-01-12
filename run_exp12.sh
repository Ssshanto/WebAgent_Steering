#!/bin/bash
# Experiment 12: Grid Search & VLM Validation
#
# 1. Grid Search on Llama 3.2 1B (Layers 6-10, Alphas 1,2,3)

set -e

PROMPT_TYPE="accuracy"
VECTOR_METHOD="response"
TRAIN_STEPS=200
EVAL_STEPS=400
SEED=0

# Output directory
mkdir -p results/exp12_grid

run_grid() {
    local MODEL=$1
    local CENTER_LAYER=$2
    local IS_VLM=$3
    
    echo "=============================================="
    echo "Starting Grid Search for $MODEL"
    echo "Center Layer: $CENTER_LAYER"
    echo "=============================================="
    
    # Calculate layer range: Center-2 to Center+2
    LAYERS=($(seq $((CENTER_LAYER - 2)) $((CENTER_LAYER + 2))))
    ALPHAS=(1.0 2.0 3.0)
    
    VLM_FLAG=""
    if [ "$IS_VLM" = "true" ]; then
        VLM_FLAG="--vlm"
    fi
    
    # 1. Run Baseline (Once)
    BASE_FILE="results/exp12_grid/${MODEL}_baseline.jsonl"
    if [ ! -f "$BASE_FILE" ]; then
        echo "[RUN] Baseline..."
        python src/miniwob_steer.py \
            --model "$MODEL" \
            --layer $CENTER_LAYER \
            --base-only \
            --eval-steps $EVAL_STEPS \
            --seed $SEED \
            $VLM_FLAG \
            --out "$BASE_FILE"
    fi
    
    # 2. Run Sweep
    for LAYER in "${LAYERS[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            OUT_FILE="results/exp12_grid/${MODEL}_L${LAYER}_a${ALPHA}.jsonl"
            
            if [ -f "$OUT_FILE" ]; then
                echo "[SKIP] $OUT_FILE exists"
                continue
            fi
            
            echo "[RUN] Layer $LAYER, Alpha $ALPHA"
            python src/miniwob_steer.py \
                --model "$MODEL" \
                --layer $LAYER \
                --coeff $ALPHA \
                --prompt-type $PROMPT_TYPE \
                --vector-method $VECTOR_METHOD \
                --train-steps $TRAIN_STEPS \
                --eval-steps $EVAL_STEPS \
                --seed $SEED \
                $VLM_FLAG \
                --out "$OUT_FILE"
        done
    done
}

# Run Llama 1B (Text) - 16 layers -> Center 8
run_grid "llama-1b" 8 "false"

echo "Grid Search Complete."