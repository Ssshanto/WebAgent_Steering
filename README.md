# Representation Engineering for Web Agents

Zero-shot steering of LLM web agents using Contrastive Activation Addition (CAA).

## Hypothesis

**Steering can improve LLM web agent performance by enhancing action-space understanding** - without task-specific fine-tuning or examples.

## Method

1. **Compute steering vector** from contrastive prompt pairs (e.g., "correct action" vs "random action")
2. **Apply vector** during inference at target layer with coefficient α
3. **Evaluate** baseline vs steered accuracy on MiniWob++ benchmark

## Setup

```bash
pip install -r requirements.txt

# MiniWob requires Selenium + Chrome
sudo apt-get install -y chromium-chromedriver
```

## Quick Start

```bash
# Run experiment (default: accuracy prompt)
./run.sh

# Test different steering prompts
./run.sh action      # Action-grounded: target decision-making
./run.sh grounding   # Task-DOM binding
./run.sh precision   # Element matching
./run.sh format      # Output compliance
```

## Steering Prompts

| Prompt | Positive | Negative | Target |
|--------|----------|----------|--------|
| `action` | "Select the correct element and action" | "Select random element, incorrect action" | Decision |
| `grounding` | "Match task to correct DOM element" | "Ignore task, select any element" | Binding |
| `precision` | "Element must exactly match task" | "Element doesn't need to match" | Matching |
| `accuracy` | "Be accurate and precise" | "Be inaccurate and imprecise" | Cognitive |
| `format` | "Output only action command" | "Explain reasoning in detail" | Output |
| `action_format` | "Select correct element. Output only action" | "Select randomly. Explain at length" | Combined |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | 0.5b | Model size (0.5b, 3b) |
| `--layer` | 14 | Intervention layer (~60% depth) |
| `--coeff` | 4.0 | Steering coefficient α |
| `--prompt-type` | accuracy | Steering prompt type |
| `--vector-method` | response | Vector computation method |
| `--train-steps` | 200 | Episodes for vector computation |
| `--eval-steps` | 400 | Episodes for evaluation |
| `--seed` | 0 | Random seed |

## CLI Usage

```bash
python src/miniwob_steer.py \
    --model 0.5b \
    --layer 14 \
    --coeff 4.0 \
    --prompt-type action \
    --tasks all \
    --out results.jsonl
```

## Output

Results saved to JSONL with per-episode records:
- `task`, `seed`: Episode info
- `base_output`, `base_success`: Baseline performance
- `steer_output`, `steer_success`: Steered performance

## Files

```
.
├── src/miniwob_steer.py   # Main implementation
├── run.sh                 # Run script
├── RESEARCH.md            # Research log & results
├── README.md              # This file
└── requirements.txt       # Dependencies
```

## Research Log

See `RESEARCH.md` for detailed experimental results, literature review, and analysis.
