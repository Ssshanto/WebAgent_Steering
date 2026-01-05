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

**Experiment 4 (Current): Layer sweep with multi-layer steering**
```bash
./run_exp4_layer_sweep.sh exp4_layer_sweep
```
- Medium-difficulty tasks (54-82% base accuracy)
- Multi-layer steering from starting layer onwards
- Layer sweep: {15, 18, 22, 25}

**Experiment 3: Coefficient sweep**
```bash
./run_experiment.sh exp3_verification
```
- High-potential tasks (result: 89.5% ceiling, no improvement)

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model-size` | `0.5b`, `3b` | `0.5b` |
| `--layer` | Starting intervention layer | `22` |
| `--coeff` | Steering coefficient (α) | `1.0` |
| `--tasks` | `all`, `high-potential`, `medium`, or comma-separated | `all` |
| `--steer-all-layers` | Multi-layer steering from `--layer` onwards | `false` |
| `--train-steps` | Episodes for vector computation | `200` |
| `--eval-steps` | Episodes for evaluation | `200` |
| `--out` | Output JSONL path | `miniwob_results.jsonl` |

## Task Subsets

| Subset | Tasks | Base Accuracy |
|--------|-------|---------------|
| `medium` | click-widget, click-dialog-2, click-link, click-button | 54-82% |
| `high-potential` | click-dialog, focus-text, etc. | 65-100% |
| `all` | 18 single-step MiniWob tasks | varies |

## Output

JSONL with per-episode: `task`, `seed`, `base_output`, `base_success`, `steer_output`, `steer_success`
