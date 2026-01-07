# Representation Engineering for Web Agents

Proof-of-concept for applying representation steering to improve LLM performance on web automation tasks (MiniWob++) in a zero-shot setting.

**Research documentation**: See `RESEARCH.md` for full background, literature, and experiment details.

**Current Status**: +17.5% accuracy improvement on 0.5B model with Layer 13, α=4.0, accuracy prompts.

## Setup

```bash
pip install -r requirements.txt

# MiniWob requires Selenium + Chrome
sudo apt-get install -y chromium-chromedriver
```

## Experiments

**Experiment 10 (Expanded Action Space):**
```bash
./run_exp10_expanded.sh
python scripts/analyze_exp10.py
```
- Tests steering on complex interactions (multi-select, dropdowns, semantic typing)
- Uses golden config: Layer 13, α=4.0, response method
- 7 new task categories beyond simple click/type

**Experiment 6 (Validation & Optimization):**
```bash
./run_experiment.sh 1  # Reproducibility
./run_experiment.sh 4  # Vector method comparison
python scripts/analyze_exp6.py
```

**Previous experiments:**
- Exp 9: Prompt strategy sweep → accuracy prompts optimal
- Exp 5: 0.5B POC → +9.7% with seeding bug
- Exp 1-4: 3B model → no improvement (well-calibrated)

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model-size` | `0.5b`, `3b` | `0.5b` |
| `--layer` | Starting intervention layer | `22` |
| `--coeff` | Steering coefficient (α) | `1.0` |
| `--prompt-type` | `verification`, `format`, `accuracy`, etc. | `verification` |
| `--vector-method` | `response` (non-standard), `prompt` (standard CAA) | `response` |
| `--tasks` | `all`, `high-potential`, `medium`, `expanded` | `all` |
| `--steer-all-layers` | Multi-layer from `--layer` onwards | `false` |
| `--base-only` | Skip steering, baseline only | `false` |

## Task Subsets

| Subset | Description |
|--------|-------------|
| `all` | 25 single-step MiniWob tasks (18 original + 7 expanded) |
| `medium` | 4 tasks at 54-82% base accuracy |
| `high-potential` | 6 tasks at 65-100% base accuracy |
| `expanded` | 7 complex interaction tasks (multi-select, dropdown, semantic) |

## Action Space

**Supported Actions:**
- `click ref=<int>` - Click an element
- `type ref=<int> text="<text>"` - Type text into field
- `select ref=<int> option="<text>"` - Select dropdown option (NEW)

**Multiple Actions:** Supported for tasks like `click-checkboxes` (output multiple lines)

## Output

JSONL with: `task`, `seed`, `base_output`, `base_success`, `steer_output`, `steer_success`
