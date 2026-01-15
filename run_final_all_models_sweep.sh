#!/bin/bash
# Final Full Sweep: All Small Models (<=2B)
# Alpha: 3.0
# Layers: All (1 to N)

set -e
export HF_HOME="/home/deeplearning01/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=1
# Ensure token is available for gated models
export HF_TOKEN=$(cat /mnt/code/huggingface/token)

# List of all small models supported
MODELS=(
    "smollm-360m"
    "0.5b"
    "qwen-coder-0.5b"
    "llama-1b"
    "gemma-1b"
    "tinyllama-1.1b"
    "opt-iml-1.3b"
    "qwen-1.5b"
    "stablelm-1.6b"
    "smollm-1.7b"
    "qwen-vl-2b"
)

ALPHA=3.0
PROMPT_TYPE="accuracy"
TRAIN_STEPS=200
EVAL_STEPS=400
SEED=0

RESULTS_DIR="results/final_small_models_sweep"
mkdir -p $RESULTS_DIR

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "Starting Full Sweep for $MODEL"
    echo "========================================"

    # Get number of layers dynamically
    # Use python to load config and print num_hidden_layers
    # We use src/miniwob_steer.py's load logic if possible, or just AutoConfig
    # Mapping MODEL key to HF ID is inside src/miniwob_steer.py, so we grep it or replicate the map?
    # Easier: Just ask src/miniwob_steer.py to tell us the layer count? No, no such flag.
    # I'll create a small helper script or just run L1..L40 and let it fail/skip? No, that's messy.
    
    # Simple Python snippet to get layer count.
    # Needs to match the MODEL_MAP in src/miniwob_steer.py
    
    LAYERS=$(python3 -c "
from transformers import AutoConfig
MODEL_MAP = {
    'smollm-360m': 'HuggingFaceTB/SmolLM2-360M-Instruct',
    '0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen-coder-0.5b': 'Qwen/Qwen2.5-Coder-0.5B-Instruct',
    'llama-1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'gemma-1b': 'google/gemma-3-1b-it',
    'tinyllama-1.1b': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'opt-iml-1.3b': 'facebook/opt-iml-1.3b',
    'qwen-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'stablelm-1.6b': 'stabilityai/stablelm-2-1_6b-chat',
    'smollm-1.7b': 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
    'qwen-vl-2b': 'Qwen/Qwen2-VL-2B-Instruct',
}
try:
    hf_id = MODEL_MAP['$MODEL']
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    print(getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 0)))
except Exception as e:
    print(0)
")

    if [ "$LAYERS" -eq "0" ]; then
        echo "Error: Could not determine layers for $MODEL. Skipping."
        continue
    fi

    echo "Detected $LAYERS layers."

    # 1. Baseline (Once)
    BASE_FILE="$RESULTS_DIR/${MODEL}_baseline.jsonl"
    if [ ! -f "$BASE_FILE" ]; then
        echo ">>> Running Baseline for $MODEL"
        VLM_FLAG=""
        if [ "$MODEL" == "qwen-vl-2b" ]; then VLM_FLAG="--vlm"; fi
        
        # Use middle layer for baseline reference (doesn't matter for base-only but required arg)
        MID_LAYER=$((LAYERS / 2))
        
        python3 src/miniwob_steer.py \
            --model "$MODEL" \
            --layer $MID_LAYER \
            --base-only \
            --eval-steps $EVAL_STEPS \
            --seed $SEED \
            $VLM_FLAG \
            --out "$BASE_FILE" || echo "Baseline failed for $MODEL"
    fi

    # 2. Sweep All Layers
    for LAYER in $(seq 1 $LAYERS); do
        OUT_FILE="$RESULTS_DIR/${MODEL}_L${LAYER}_a${ALPHA}.jsonl"
        
        if [ -f "$OUT_FILE" ]; then
            echo "  Skipping L${LAYER} (exists)"
            continue
        fi
        
        echo ">>> Running: $MODEL L$LAYER a$ALPHA"
        VLM_FLAG=""
        if [ "$MODEL" == "qwen-vl-2b" ]; then VLM_FLAG="--vlm"; fi
        
        python3 src/miniwob_steer.py \
            --model "$MODEL" \
            --layer $LAYER \
            --coeff $ALPHA \
            --prompt-type $PROMPT_TYPE \
            --vector-method $VECTOR_METHOD \
            --train-steps $TRAIN_STEPS \
            --eval-steps $EVAL_STEPS \
            --seed $SEED \
            $VLM_FLAG \
            --out "$OUT_FILE" || echo "Run failed for $MODEL L$LAYER"
    done
done

echo "Final Full Sweep Complete."
