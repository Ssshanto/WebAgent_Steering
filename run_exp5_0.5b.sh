#!/bin/bash
# Experiment 5: 0.5B Model Steering Suite
#
# Hypothesis: Smaller models have more "steer-able" failures (format issues,
# attention errors) compared to larger models that already perform well.
#
# Qwen 0.5B has 24 layers (vs 36 for 3B)
# Default layer: 14 (58% depth, equivalent to L22 in 3B)
#
# Usage: ./run_exp5_0.5b.sh [experiment_name]

set -e

EXPERIMENT_NAME="${1:-exp5_0.5b}"
MODEL_SIZE="0.5b"
TRAIN_STEPS=200
EVAL_STEPS=400  # More episodes for statistical power
TASKS="all"     # Test on all tasks to characterize failures

echo "========================================"
echo "Experiment 5: 0.5B Model Steering Suite"
echo "========================================"
echo "Model: Qwen 2.5 0.5B (24 layers)"
echo "Tasks: $TASKS"
echo "========================================"

mkdir -p results

# =============================================================================
# Phase 1: Baseline Characterization
# =============================================================================
echo ""
echo "=== Phase 1: Baseline (no steering) ==="

python src/miniwob_steer.py \
    --model-size $MODEL_SIZE \
    --tasks $TASKS \
    --eval-steps $EVAL_STEPS \
    --base-only \
    --out "results/${EXPERIMENT_NAME}_baseline.jsonl"

echo ">>> Baseline complete"

# =============================================================================
# Phase 2: Steering with Verification Prompts
# =============================================================================
echo ""
echo "=== Phase 2: Verification prompts ==="

# 2a: Single layer L14 (58% depth), α=1.0
python src/miniwob_steer.py \
    --model-size $MODEL_SIZE \
    --layer 14 \
    --coeff 1.0 \
    --tasks $TASKS \
    --prompt-type verification \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --out "results/${EXPERIMENT_NAME}_verify_L14_c1.jsonl"

# 2b: Single layer L14, α=3.0 (stronger steering for weaker model)
python src/miniwob_steer.py \
    --model-size $MODEL_SIZE \
    --layer 14 \
    --coeff 3.0 \
    --tasks $TASKS \
    --prompt-type verification \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --out "results/${EXPERIMENT_NAME}_verify_L14_c3.jsonl"

# 2c: Multi-layer L10-L23, α=2.0
python src/miniwob_steer.py \
    --model-size $MODEL_SIZE \
    --layer 10 \
    --coeff 2.0 \
    --tasks $TASKS \
    --prompt-type verification \
    --steer-all-layers \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --out "results/${EXPERIMENT_NAME}_verify_L10_multi_c2.jsonl"

echo ">>> Verification steering complete"

# =============================================================================
# Phase 3: Steering with Format Prompts
# =============================================================================
echo ""
echo "=== Phase 3: Format prompts ==="

# 3a: Single layer L14, α=2.0
python src/miniwob_steer.py \
    --model-size $MODEL_SIZE \
    --layer 14 \
    --coeff 2.0 \
    --tasks $TASKS \
    --prompt-type format \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --out "results/${EXPERIMENT_NAME}_format_L14_c2.jsonl"

# 3b: Multi-layer L10-L23, α=2.0
python src/miniwob_steer.py \
    --model-size $MODEL_SIZE \
    --layer 10 \
    --coeff 2.0 \
    --tasks $TASKS \
    --prompt-type format \
    --steer-all-layers \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --out "results/${EXPERIMENT_NAME}_format_L10_multi_c2.jsonl"

echo ">>> Format steering complete"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Experiment 5 Complete"
echo "========================================"
echo ""
echo "Results files:"
for f in results/${EXPERIMENT_NAME}_*.jsonl; do
    if [ -f "$f" ]; then
        EPISODES=$(wc -l < "$f")
        echo "  $(basename $f): $EPISODES episodes"
    fi
done

echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_exp5.py"
