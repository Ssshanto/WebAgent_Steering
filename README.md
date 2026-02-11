# WebAgent_Steering

## intent
- zero-shot MI + inference-time steering for MiniWob web agents
- MI-first, performance-second

## stack
- BrowserGym MiniWob
- HF causal LMs
- core script: `src/miniwob_steer.py` (subcommand CLI)
- modules:
  - `src/agent_core.py`
  - `src/eval_core.py`
  - `src/sae_core.py`

## setup
```bash
pip install -r requirements.txt
playwright install chromium
export MINIWOB_URL="http://localhost:8080/miniwob/"
```

## quick run
```bash
python src/miniwob_steer.py eval \
  --model qwen3-1.7b \
  --layer auto \
  --task-manifest runtime_state/sae_val_manifest.json \
  --seed 0 \
  --out results/run.jsonl
```

## SAE pipeline (limited subset)
```bash
python src/miniwob_steer.py split \
  --tasks click-button,click-color,use-slider \
  --train-manifest runtime_state/sae_train_manifest.json \
  --val-manifest runtime_state/sae_val_manifest.json \
  --seed 0

python src/miniwob_steer.py capture \
  --model qwen3-1.7b \
  --layers 10-18 \
  --task-manifest runtime_state/sae_train_manifest.json \
  --out runtime_state/sae_capture_train.pt \
  --seed 0

python src/miniwob_steer.py train-sae \
  --capture runtime_state/sae_capture_train.pt \
  --out runtime_state/sae_artifact.pt \
  --steps 300

python src/miniwob_steer.py validate-sae \
  --model qwen3-1.7b \
  --sae-artifact runtime_state/sae_artifact.pt \
  --val-manifest runtime_state/sae_val_manifest.json \
  --out-dir results/sae_validation \
  --top-k 1 \
  --modes suppress,amplify \
  --random-controls 5 \
  --seed 0
```

## invariants
- benchmark: MiniWob only
- default candidate/deployment model: `qwen3-1.7b` (`qwen3-0.6b` is probe/exploration only)
- seed: `0` for baseline/steer comparison
- baseline-first paired evaluation on identical `(task, seed)`
- heavy runs serialized unless GPU headroom allows overlap
- launchers/configs: ephemeral; generate in `runtime_state/` as needed

## known code quirks
- eval episodes fixed to 3/task in `evaluate`
- qwen3 prompt path disables thinking (`enable_thinking=False`, fallback `/no_think`)
- default prompt has CoT-style suffix unless `--strict-action-prompt`
- action mapping uses `HighLevelActionSet(strict=False, multiaction=False)`

## outputs
- run records: JSONL in `results/`
- SAE artifacts and captures: `runtime_state/`
- runtime metadata: `runtime_state/`
- logs: `logs/`
- required reporting: quantitative tables + qualitative before/after examples

## docs split
- `AGENTS.md`: persistent invariants
- `MEMORY.md`: mutable live state
- `RESEARCH.md`: durable decisions/failures/insights
