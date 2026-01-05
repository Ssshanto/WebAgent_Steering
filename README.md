# Representation Engineering for Web Agents

Proof-of-concept for applying representation steering to improve LLM performance on web automation tasks (MiniWob++) in a zero-shot setting.

**Research documentation**: See `RESEARCH.md` for full background, literature, and experiment details.

## Setup

```bash
pip install -r requirements.txt

# MiniWob requires Selenium + Chrome
sudo apt-get install -y chromium-chromedriver
```

## Quick Start

**Single experiment:**
```bash
python src/miniwob_steer.py \
  --model-size 3b \
  --layer 22 \
  --coeff 1.0 \
  --tasks high-potential \
  --train-steps 200 \
  --eval-steps 200 \
  --out results/exp3_coeff1.0.jsonl
```

**Coefficient sweep (Experiment 3):**
```bash
chmod +x run_experiment.sh
./run_experiment.sh exp3_verification
```

This runs coefficients {1.0, 2.0, 3.0, 5.0} on the high-potential task subset.

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model-size` | Model size: `0.5b`, `3b` | `0.5b` |
| `--layer` | Intervention layer | `22` |
| `--coeff` | Steering coefficient (α) | `1.0` |
| `--tasks` | Task set: `all`, `high-potential`, or comma-separated | `all` |
| `--train-steps` | Episodes for vector computation | `200` |
| `--eval-steps` | Episodes for evaluation | `200` |
| `--base-only` | Skip steering, evaluate base model only | `false` |
| `--steer-all-layers` | Multi-layer steering from `--layer` onwards | `false` |
| `--out` | Output JSONL path | `miniwob_results.jsonl` |

## Task Subsets

- **`all`**: 18 single-step MiniWob tasks
- **`high-potential`**: 6 tasks with 65-86% base accuracy (faster iteration)

## Output

Results saved as JSONL with per-episode records:
- `task`, `seed`, `prompt`
- `base_output`, `base_action`, `base_reward`, `base_success`
- `steer_output`, `steer_action`, `steer_reward`, `steer_success`

## Current Status

See `RESEARCH.md` for:
- Experiment results and analysis
- Steering prompt rationale
- Next steps and hypotheses
