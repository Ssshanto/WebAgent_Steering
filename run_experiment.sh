#!/bin/bash
# Experiment 6: Validation and Optimization
#
# Phase 1: Reproducibility check with different seeds
# Phase 2: Coefficient optimization around α=3.0
# Phase 3: Layer optimization around L14
# Phase 4: Vector method comparison (response vs prompt)
#
# Based on Exp 5 success: Accuracy prompts, L14, α=3.0 achieved +9.7% improvement
#
# Usage: ./run_experiment.sh [phase]

set -e

PHASE="${1:-all}"
MODEL_SIZE="0.5b"
TASKS="all"
TRAIN_STEPS=200
EVAL_STEPS=400
PROMPT_TYPE="accuracy"

mkdir -p results

run_phase1() {
    echo "=== Phase 1: Reproducibility (seeds 0, 42, 123) ==="
    for SEED in 0 42 123; do
        python src/miniwob_steer.py --model-size $MODEL_SIZE --layer 14 --coeff 3.0 \
            --tasks $TASKS --prompt-type $PROMPT_TYPE --vector-method response \
            --train-steps $TRAIN_STEPS --eval-steps $EVAL_STEPS --seed $SEED \
            --out "results/exp6_seed${SEED}.jsonl"
    done
}

run_phase2() {
    echo "=== Phase 2: Coefficient sweep (2.0-5.0) ==="
    for COEFF in 2.0 2.5 3.0 3.5 4.0 5.0; do
        python src/miniwob_steer.py --model-size $MODEL_SIZE --layer 14 --coeff $COEFF \
            --tasks $TASKS --prompt-type $PROMPT_TYPE --vector-method response \
            --train-steps $TRAIN_STEPS --eval-steps $EVAL_STEPS \
            --out "results/exp6_coeff${COEFF}.jsonl"
    done
}

run_phase3() {
    echo "=== Phase 3: Layer sweep (12-16) ==="
    for LAYER in 12 13 14 15 16; do
        python src/miniwob_steer.py --model-size $MODEL_SIZE --layer $LAYER --coeff 3.0 \
            --tasks $TASKS --prompt-type $PROMPT_TYPE --vector-method response \
            --train-steps $TRAIN_STEPS --eval-steps $EVAL_STEPS \
            --out "results/exp6_layer${LAYER}.jsonl"
    done
}

run_phase4() {
    echo "=== Phase 4: Vector method comparison (response vs prompt) ==="
    echo "Testing best config (L14, α=3.0) with both vector methods..."
    
    echo "  Running with response method (original)..."
    python src/miniwob_steer.py --model-size $MODEL_SIZE --layer 14 --coeff 3.0 \
        --tasks $TASKS --prompt-type $PROMPT_TYPE --vector-method response \
        --train-steps $TRAIN_STEPS --eval-steps $EVAL_STEPS --seed 0 \
        --out "results/exp6_method_response.jsonl"
    
    echo "  Running with prompt method (standard CAA)..."
    python src/miniwob_steer.py --model-size $MODEL_SIZE --layer 14 --coeff 3.0 \
        --tasks $TASKS --prompt-type $PROMPT_TYPE --vector-method prompt \
        --train-steps $TRAIN_STEPS --eval-steps $EVAL_STEPS --seed 0 \
        --out "results/exp6_method_prompt.jsonl"
}

case "$PHASE" in
    1) run_phase1 ;;
    2) run_phase2 ;;
    3) run_phase3 ;;
    4) run_phase4 ;;
    all) run_phase1; run_phase2; run_phase3; run_phase4 ;;
    *) echo "Usage: $0 [1|2|3|4|all]"; exit 1 ;;
esac

echo "Complete. Analyze with: python scripts/analyze_exp6.py"
