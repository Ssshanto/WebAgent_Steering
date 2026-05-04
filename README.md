# WebAgent Steering

Contrastive Activation Addition (CAA) for zero-shot BrowserGym web agents.

The code computes steering vectors from contrastive prompts, applies them at inference time, and compares baseline vs steered task success under matched episode seeds. The retained mainline remains MiniWob++, with optional smoke support for WebArena and WorkArena transfer checks.

Project defaults are intentionally lightweight for iteration (`--seed 0`, `--episodes-per-task 3`). For BrowserGym benchmark-faithful seed scheduling, use `--seed-mode browsergym --seed 42 --episodes-per-task 5`.

## Setup

```bash
pip install -r requirements.txt

# Install Playwright browser (BrowserGym uses Playwright instead of Selenium)
playwright install chromium

# Setup MiniWob++ server
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git
cd miniwob-plusplus/miniwob/html
python -m http.server 8080

# BrowserGym expects the MiniWob root.
export MINIWOB_URL="http://localhost:8080/miniwob/"
```

## Single Run

```bash
python src/miniwob_steer.py \
  --model 0.5b \
  --layer auto \
  --coeff 3.0 \
  --prompt-type accuracy \
  --tasks all \
  --seed 0 \
  --episodes-per-task 3 \
  --out results/qwen05b_L14_a3.jsonl
```

## Sweep

```bash
python scripts/run_sweep.py \
  --model 0.5b \
  --layers 10-18 \
  --alphas 1.0,2.0,3.0,4.0 \
  --prompt-type accuracy \
  --tasks all \
  --seed 0 \
  --episodes-per-task 3 \
  --out-dir results/qwen05b_sweep
```

## Multi-Dataset Action-Grounding Smoke

Before running model sweeps on `cvpc`, check CUDA plus one reset/step per installed BrowserGym family:

```bash
python scripts/preflight_browsergym.py \
  --datasets miniwob,webarena,workarena
```

The action-space CAA prompt pair is available as:

- `action_space_pos_minus_base`: positive instruction minus empty baseline prompt
- `action_space_pos_minus_neg`: positive instruction minus unchecked-action negative

Raw and residual-norm-scaled alphas can be run together. `--alpha-pcts 5,10,20,40` means `5%`, `10%`, `20%`, and `40%` of the measured average residual norm on the vector-construction prompt slice.

```bash
python scripts/run_sweep.py \
  --model gemma-3-4b \
  --dataset miniwob \
  --prompt-type action_space_pos_minus_base \
  --vector-method prompt \
  --tasks click-button,click-link,click-option,choose-list,focus-text \
  --episode-steps 6 \
  --layers 17 \
  --alphas 500,750,1000,1500 \
  --alpha-pcts 5,10,20,40 \
  --measure-resid-norm \
  --steer-position last \
  --train-steps 25 \
  --cache-dir vectors \
  --seed 0 \
  --out-dir results/gemma3_4b_action_space_miniwob_L17
```

For WebArena or WorkArena smoke transfer, set `--dataset webarena` or `--dataset workarena`. If the vector should be built on the MiniWob action-grounding slice and evaluated elsewhere, keep the evaluation dataset separate from the vector dataset:

```bash
python scripts/run_sweep.py \
  --model gemma-3-4b \
  --dataset webarena \
  --tasks 10 \
  --vector-dataset miniwob \
  --vector-tasks click-button,click-link,click-option,choose-list,focus-text \
  --prompt-type action_space_pos_minus_base \
  --vector-method prompt \
  --layers 17 \
  --alphas 750,1000 \
  --measure-resid-norm \
  --episode-steps 6 \
  --train-steps 25 \
  --seed 0 \
  --out-dir results/gemma3_4b_action_space_webarena_smoke
```

## Interface-Variant Grounding

Interface variants rewrite the ids shown in the Accessibility Tree and DOM while preserving a reversible map to the real BrowserGym bids. This supports frozen one-step diagnostics and stepped MiniWob runs under schema shift.

Available modes:

```text
original, permuted, alphanumeric, structured, uuid, handle, mixed,
longprefix, stale_ids, fake_examples, decoy_labels
```

Build an interface-general vector from source modes and evaluate held-out schemas:

```bash
python scripts/run_sweep.py \
  --model gemma-3-4b \
  --prompt-type gemma_tree_pos_minus_base \
  --vector-method prompt \
  --tasks click-button,click-link,click-option,choose-list,focus-text \
  --plan-file /tmp/gemma_target25_plan.txt \
  --episode-steps 6 \
  --layers 15,17,19 \
  --alpha-pcts 2.5,5,10 \
  --measure-resid-norm \
  --interface-train-modes original,permuted,alphanumeric,fake_examples,stale_ids \
  --interface-heldout-modes structured,uuid,handle,mixed,longprefix,decoy_labels \
  --train-steps 25 \
  --seed 0 \
  --out-dir results/gemma3_4b_interface_general_target25
```

When `--interface-train-modes` is not just `original`, vectors are cached under a tagged directory such as `vectors/gemma-3-4b/interface_original_alnum_fake_examples_stale_seed_0/`.

Frozen one-step diagnostic:

```bash
python scripts/run_frozen_grounding.py \
  --model gemma-3-4b \
  --condition baseline \
  --plan /tmp/gemma_target25_plan.txt \
  --interface-modes original,structured,uuid,handle \
  --out results/frozen_interface_baseline.jsonl
```

Stepped MiniWob diagnostic with real bid execution:

```bash
python scripts/run_remap_eval.py \
  --model gemma-3-4b \
  --condition steer \
  --vector vectors/gemma-3-4b/seed_0/gemma_tree_pos_minus_base_L17.pt \
  --layer 17 \
  --alpha 1000 \
  --plan /tmp/gemma_target25_plan.txt \
  --interface-mode uuid \
  --episode-steps 6 \
  --out results/stepped_interface_uuid.jsonl
```

Summaries report `parse_valid`, `action_type_valid`, `valid_current_id`, `copied_example_id`, `stale_id`, `label_as_id`, `bogus_argument`, and stepped success/reward where available:

```bash
python scripts/summarize_grounding.py results/frozen_interface_baseline.jsonl
```

To test normalized sums of individually cached interface vectors:

```bash
python scripts/combine_vectors.py \
  --vectors vectors/gemma-3-4b/interface_original_alnum_seed_0/interface_current_id_binding_L17.pt,vectors/gemma-3-4b/interface_original_alnum_seed_0/interface_action_type_binding_L17.pt \
  --out vectors/gemma-3-4b/interface_original_alnum_seed_0/interface_id_plus_type_L17.pt
```

## Main Options

- `--model`: one of the aliases in `src/miniwob_steer.py`
- `--dataset`: `miniwob`, `webarena`, or `workarena`
- `--vector-dataset`: optional separate dataset for vector construction
- `--interface-train-modes`: comma-separated id schemas for vector construction
- `--interface-mode` / `--interface-heldout-modes`: id schemas for evaluation
- `--layer`: integer layer or `auto`
- `--coeff` / `--alphas`: steering strength
- `--alpha-pcts`: residual-norm-scaled steering strengths
- `--measure-resid-norm`: log `avg_l17_resid_norm` and `alpha_pct`
- `--prompt-type`: contrastive prompt family
- `--vector-method`: `response` or `prompt`
- `--seed-mode`: `project` or `browsergym`
- `--base-only`: write frozen baseline episodes
- `--steer-only --base-jsonl <path>`: paired steered replay using baseline seeds
- `--episodes-per-task`: fixed per-task evaluation count

## BrowserGym-Faithful Eval

```bash
python src/miniwob_steer.py \
  --model 0.5b \
  --layer auto \
  --coeff 3.0 \
  --prompt-type accuracy \
  --tasks all \
  --seed-mode browsergym \
  --seed 42 \
  --episodes-per-task 5 \
  --episode-steps 10 \
  --out results/miniwob_browsergym_seed42.jsonl
```

## Output

Results are JSONL episode records plus sweep TSV summaries. MiniWob cached vectors keep the historical path `vectors/<model>/seed_<seed>/`; non-MiniWob vector caches use `vectors/<model>/<dataset>_seed_<seed>/` unless `--vector-cache-tag` is set.

Key fields: `dataset`, `task`, `seed`, `base_success`, `steer_success`, `base_error`, `steer_error`, `base_action`, `steer_action`, `base_last_action_metrics`, `steer_last_action_metrics`.

## Files

```
AGENTS.md              persistent project state and hierarchy
RESEARCH.md            method basis and retained results
src/miniwob_steer.py   CAA implementation and evaluation
scripts/run_sweep.py   efficient layer/alpha sweep runner
```
