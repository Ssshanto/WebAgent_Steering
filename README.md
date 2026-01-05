# Representation Engineering for Web Agents

Proof-of-concept for applying representation steering to improve LLM performance on web automation tasks (MiniWob++) in a zero-shot setting.

**Research documentation**: See `RESEARCH.md` for full background, literature, and experiment details.

## Setup

```bash
pip install -r requirements.txt

# MiniWob requires Selenium + Chrome
sudo apt-get install -y chromium-chromedriver
```

## Experiments

**Experiment 5 (Current): 0.5B model steering**
```bash
./run_exp5_0.5b.sh exp5_0.5b
```
- Tests if smaller models have more "steer-able" failures
- Includes baseline + verification + format-focused steering
- Analysis: `python scripts/analyze_exp5.py`

**Previous experiments (3B model - all failed):**
- Exp 4: Layer sweep on medium tasks → 0% to -1%
- Exp 3: Coefficient sweep on high-potential → 0% (ceiling effect)
- Exp 1-2: Accuracy prompts → 0% to -0.5%

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model-size` | `0.5b`, `3b` | `0.5b` |
| `--layer` | Starting intervention layer | `22` |
| `--coeff` | Steering coefficient (α) | `1.0` |
| `--prompt-type` | `verification`, `format`, `accuracy` | `verification` |
| `--vector-method` | `response` (non-standard), `prompt` (standard CAA) | `response` |
| `--tasks` | `all`, `high-potential`, `medium` | `all` |
| `--steer-all-layers` | Multi-layer from `--layer` onwards | `false` |
| `--base-only` | Skip steering, baseline only | `false` |

## Task Subsets

| Subset | Description |
|--------|-------------|
| `all` | 18 single-step MiniWob tasks |
| `medium` | 4 tasks at 54-82% base accuracy |
| `high-potential` | 6 tasks at 65-100% base accuracy |

## Output

JSONL with: `task`, `seed`, `base_output`, `base_success`, `steer_output`, `steer_success`
