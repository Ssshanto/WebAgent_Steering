# Representation Engineering for Web Agents

Zero-shot steering of LLM web agents using Contrastive Activation Addition (CAA).

**Now using BrowserGym** - A unified framework for web agent benchmarks with improved stability and features.

## Hypothesis

**Steering can improve LLM web agent performance by enhancing action-space understanding** - without task-specific fine-tuning or examples.

## Method

1. **Compute steering vector** from contrastive prompt pairs (e.g., "correct action" vs "random action")
2. **Apply vector** during inference at target layer with coefficient α
3. **Evaluate** baseline vs steered accuracy on MiniWob++ benchmark via BrowserGym

## Setup

```bash
pip install -r requirements.txt

# Install Playwright browser (BrowserGym uses Playwright instead of Selenium)
playwright install chromium

# Setup MiniWob++ server
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git
cd miniwob-plusplus
npm install
npm run build
npm run serve  # Runs on http://localhost:8080

# Set environment variable
export MINIWOB_URL="http://localhost:8080/"
```

## Quick Start

```bash
# Run experiment (default: accuracy prompt)
python src/miniwob_steer.py

# Test different steering prompts
python src/miniwob_steer.py --prompt-type action      # Action-grounded: target decision-making
python src/miniwob_steer.py --prompt-type grounding   # Task-DOM binding
python src/miniwob_steer.py --prompt-type precision   # Element matching
python src/miniwob_steer.py --prompt-type format      # Output compliance
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
| `--tasks` | all | Task list or 'all' |
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

## Task Set

`--tasks all` uses the full MiniWob++ task set from the BrowserGym registry (no custom subset).

## Migration to BrowserGym

This project has been migrated from direct MiniWob++ usage to **BrowserGym** for improved stability and features:

- **Browser Backend**: Now uses Playwright instead of Selenium for more reliable automation
- **Richer Observations**: Includes DOM object, accessibility tree, screenshots, and error feedback
- **Element IDs**: Uses `bid` (BrowserGym ID) attributes instead of `ref`
- **Action Format**: High-level string actions: `click("N")`, `fill("N", "text")`
- **Unified API**: Same interface works across MiniWob++, WebArena, WorkArena, and other benchmarks

### Key Changes:
- Element references changed from `ref` to `bid` in prompts and code
- Actions now use BrowserGym string format instead of ActionTypes enum
- DOM processing uses `flatten_dom_to_str()` utility
- No more Selenium driver monkeypatching needed

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
