# WebAgent_Steering

## intent
- zero-shot MI + inference-time steering for MiniWob web agents
- MI-first, performance-second

## stack
- BrowserGym MiniWob
- HF causal LMs
- core script: `src/miniwob_steer.py`
- sweep script: `scripts/run_sweep.py`
- queue runner: `scripts/run_experiment_queue.py`

## setup
```bash
pip install -r requirements.txt
playwright install chromium
export MINIWOB_URL="http://localhost:8080/miniwob/"
```

## quick run
```bash
python src/miniwob_steer.py --model qwen3-1.7b --layer auto --seed 0 --out results/run.jsonl
```

## low-noise run
```bash
python src/miniwob_steer.py --model qwen3-1.7b --quiet --no-progress --strict-action-prompt --out results/run.jsonl
```

## sweep example
```bash
python scripts/run_sweep.py \
  --model qwen3-1.7b \
  --layers 10-14 \
  --alphas 1.0,2.0,3.0 \
  --seed 0 \
  --out-dir results/qwen3_sweep/qwen3-1.7b \
  --quiet --no-progress --strict-action-prompt
```

## invariants
- benchmark: MiniWob only
- default candidate/deployment model: `qwen3-1.7b` (`qwen3-0.6b` is probe/exploration only)
- seed: `0` for baseline/steer comparison
- baseline-first; steer uses `--steer-only --base-jsonl`
- heavy runs serialized unless GPU headroom allows overlap
- launchers/configs: ephemeral; generate in `runtime_state/` as needed

## known code quirks
- eval episodes fixed to 3/task in `evaluate`
- qwen3 prompt path disables thinking (`enable_thinking=False`, fallback `/no_think`)
- default prompt has CoT-style suffix unless `--strict-action-prompt`
- action mapping uses `HighLevelActionSet(strict=False, multiaction=False)`

## outputs
- run records: JSONL in `results/`
- vectors/cache: `vectors/`
- queue/runtime: `runtime_state/`
- logs: `logs/`
- required reporting: quantitative tables + qualitative before/after examples

## docs split
- `AGENTS.md`: persistent invariants
- `MEMORY.md`: mutable live state
- `RESEARCH.md`: durable decisions/failures/insights
