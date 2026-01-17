#!/bin/bash
# Robust Optimization Sweep (Targeted)
# Models: Small models (<2B)
# Layers: Peak +/- 1
# Alphas: 1.0, 2.0, 3.0, 4.0, 5.0

set -e
export HF_HOME="/home/deeplearning01/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN=$(cat /mnt/code/huggingface/token)

MODELS=("smollm-1.7b" "stablelm-1.6b" "llama-1b" "tinyllama-1.1b" "qwen-coder-0.5b" "0.5b")
ALPHAS=(1.0 2.0 3.0 4.0 5.0)
PROMPT_TYPE="accuracy"
VECTOR_METHOD="response"
TRAIN_STEPS=200
EVAL_STEPS=400
SEED=0

# Define Peak Layers (Center)
declare -A PEAKS
PEAKS["0.5b"]=6
PEAKS["qwen-coder-0.5b"]=11
PEAKS["llama-1b"]=9
PEAKS["tinyllama-1.1b"]=10
PEAKS["stablelm-1.6b"]=12
PEAKS["smollm-1.7b"]=22

RESULTS_DIR="results/exp8_robust_opt"
mkdir -p $RESULTS_DIR

for MODEL in "${MODELS[@]}"; do
    CENTER=${PEAKS[$MODEL]}
    # Sweep Center-1, Center, Center+1
    LAYERS=($(seq $((CENTER - 1)) $((CENTER + 1))))
    
    echo "========================================"
    echo "Robust Sweep: $MODEL (Peak L$CENTER)"
    echo "Layers: ${LAYERS[*]}"
    echo "Alphas: ${ALPHAS[*]}"
    echo "========================================"
    
    for LAYER in "${LAYERS[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            OUT_FILE="$RESULTS_DIR/${MODEL}_L${LAYER}_a${ALPHA}.jsonl"
            
            if [ -f "$OUT_FILE" ]; then
                echo "  Skipping L${LAYER} a${ALPHA} (exists)"
                continue
            fi
            
            echo ">>> Running: $MODEL L$LAYER a$ALPHA"
            python3 src/miniwob_steer.py \
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

echo "Robust Optimization Sweep Complete."
