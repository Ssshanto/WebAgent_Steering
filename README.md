# Representation Engineering for Web Agents

Proof-of-concept for applying representation steering to improve LLM performance on web automation tasks (MiniWob++) in a zero-shot setting.

**Research documentation**: See `RESEARCH.md` for full background, literature, and experiment details.

**Current Status**: Finding optimal layer/coefficient configuration for accuracy prompts.

## Setup

```bash
pip install -r requirements.txt

# MiniWob requires Selenium + Chrome
sudo apt-get install -y chromium-chromedriver
```

## Quick Start

**Find Best Configuration:**
```bash
# Run hyperparameter sweep (layers 12-15, coefficients 2.0-5.0)
./run_optimization.sh

# Analyze results and find best configuration
python scripts/analyze_optimization.py
```

**Runtime:** ~28 hours (28 configurations × ~1 hour each)

## What It Does

1. **Sweeps hyperparameters:**
   - Layers: 12, 13, 14, 15 (50-62% model depth)
   - Coefficients: 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
   - Fixed: accuracy prompts, response method, seed=0

2. **Tests on 25 tasks:**
   - 18 original tasks (click, type, simple interactions)
   - 7 expanded tasks (checkboxes, dropdowns, semantic typing)

3. **Finds optimal configuration:**
   - Best layer/coefficient combination
   - Maximum accuracy improvement
   - Minimal parse failures

## Configuration Options

The script uses these settings (can be modified in `run_optimization.sh`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen 2.5 0.5B | Small model with high parse failures |
| Prompt | accuracy | "Be accurate and precise..." |
| Vector Method | response | Extract from generated text |
| Train Steps | 200 | Episodes for vector computation |
| Eval Steps | 400 | Episodes for evaluation |
| Tasks | all | 25 single-step tasks |

## CLI Usage (Advanced)

If you want to run specific configurations manually:

```bash
python src/miniwob_steer.py \
  --model-size 0.5b \
  --layer 13 \
  --coeff 4.0 \
  --prompt-type accuracy \
  --vector-method response \
  --tasks all \
  --train-steps 200 \
  --eval-steps 400 \
  --seed 0 \
  --out results/custom.jsonl
```

### Available Flags

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--model-size` | `0.5b`, `3b` | `0.5b` | Model size |
| `--layer` | 0-23 (0.5b), 0-35 (3b) | `22` | Steering layer |
| `--coeff` | any float | `1.0` | Steering coefficient |
| `--prompt-type` | see PROMPT_CONFIGS | `verification` | Steering prompt |
| `--vector-method` | `response`, `prompt` | `response` | Vector computation |
| `--tasks` | `all`, `expanded`, or comma-separated | `all` | Task subset |
| `--train-steps` | any int | `200` | Vector training episodes |
| `--eval-steps` | any int | `200` | Evaluation episodes |
| `--seed` | any int | `0` | Random seed |

## Output

Results are saved to `results/L{layer}_a{coeff}_s{seed}.jsonl` with:
- `task`, `seed`: Episode info
- `base_output`, `base_success`: Baseline performance
- `steer_output`, `steer_success`: Steered performance
- `base_action`, `steer_action`: Parsed actions

## Next Steps

After finding the best configuration:
1. Validate reproducibility across seeds
2. Test on expanded action space tasks
3. Compare different prompt strategies
4. Scale to larger models (3B, 7B)
