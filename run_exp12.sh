#!/bin/bash
# Experiment 12: Multi-Model Scaling
#
# Test CAA steering across 6 text models + 1 VLM
# Fixed hyperparameters: layer=auto (50% depth), coeff=3.0, prompt=accuracy
# Total time: ~10 hours
#
# Usage: ./run_exp12.sh [model_name]
#        ./run_exp12.sh           # Run all models
#        ./run_exp12.sh llama-1b  # Run specific model

set -e

# Configuration
COEFF=3.0
PROMPT_TYPE="accuracy"
VECTOR_METHOD="response"
TRAIN_STEPS=200
EVAL_STEPS=400
SEED=0

# Model list (ordered by size for memory efficiency)
TEXT_MODELS=(
    "llama-1b"      # 1B - fastest
    "qwen-1.5b"     # 1.5B
    "smollm-1.7b"   # 1.7B
    "gemma-2b"      # 2B
    "llama-3b"      # 3B
    "phi-3.8b"      # 3.8B - largest text model
)

VLM_MODELS=(
    "qwen-vl-3b"    # VLM last (most complex)
)

mkdir -p results/exp12

run_model() {
    local MODEL=$1
    local IS_VLM=$2
    
    echo "=============================================="
    echo "Model: $MODEL"
    echo "=============================================="
    
    BASELINE_FILE="results/exp12/${MODEL}_baseline.jsonl"
    STEERED_FILE="results/exp12/${MODEL}_steered.jsonl"
    
    # Run baseline
    if [ -f "$BASELINE_FILE" ]; then
        echo "[SKIP] Baseline exists: $BASELINE_FILE"
    else
        echo "[RUN] Baseline evaluation..."
        VLM_FLAG=""
        if [ "$IS_VLM" = "true" ]; then
            VLM_FLAG="--vlm"
        fi
        python src/miniwob_steer.py \
            --model "$MODEL" \
            --layer auto \
            --base-only \
            --eval-steps $EVAL_STEPS \
            --seed $SEED \
            $VLM_FLAG \
            --out "$BASELINE_FILE"
    fi
    
    # Run steered
    if [ -f "$STEERED_FILE" ]; then
        echo "[SKIP] Steered exists: $STEERED_FILE"
    else
        echo "[RUN] Steered evaluation..."
        VLM_FLAG=""
        if [ "$IS_VLM" = "true" ]; then
            VLM_FLAG="--vlm"
        fi
        python src/miniwob_steer.py \
            --model "$MODEL" \
            --layer auto \
            --coeff $COEFF \
            --prompt-type $PROMPT_TYPE \
            --vector-method $VECTOR_METHOD \
            --train-steps $TRAIN_STEPS \
            --eval-steps $EVAL_STEPS \
            --seed $SEED \
            $VLM_FLAG \
            --out "$STEERED_FILE"
    fi
    
    echo ""
}

# Header
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           EXPERIMENT 12: MULTI-MODEL SCALING                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Coefficient: $COEFF"
echo "  Prompt: $PROMPT_TYPE"
echo "  Layer: auto (50% depth)"
echo "  Train steps: $TRAIN_STEPS"
echo "  Eval steps: $EVAL_STEPS"
echo ""
echo "Text Models: ${TEXT_MODELS[*]}"
echo "VLM Models:  ${VLM_MODELS[*]}"
echo ""

# If specific model requested
if [ -n "$1" ]; then
    if [[ " ${VLM_MODELS[*]} " =~ " $1 " ]]; then
        run_model "$1" "true"
    else
        run_model "$1" "false"
    fi
    exit 0
fi

# Run all text models
for MODEL in "${TEXT_MODELS[@]}"; do
    run_model "$MODEL" "false"
done

# Run VLM models
for MODEL in "${VLM_MODELS[@]}"; do
    run_model "$MODEL" "true"
done

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                   EXPERIMENT 12 COMPLETE                             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: results/exp12/"
echo "Analyze with: python scripts/analyze_exp12.py"
