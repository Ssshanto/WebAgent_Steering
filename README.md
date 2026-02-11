# WebAgent_Steering

Minimal MiniWob PoC with baseline and SAE steering modes.

## Setup
```bash
pip install -r requirements.txt
playwright install chromium
export MINIWOB_URL="http://localhost:8080/miniwob/"
```

## Baseline (no SAE)
```bash
python src/miniwob_steer.py eval \
  --model qwen3-1.7b \
  --tasks click-button
```

## SAE steering
```bash
python src/miniwob_steer.py eval-sae \
  --model gemma-2-2b \
  --tasks click-button \
  --sae-release gemma-scope-2b-pt-res-canonical \
  --sae-id layer_13/width_16k/canonical \
  --feature-ids 15319 \
  --mode suppress \
  --alpha 1.0
```

## Notes
- No task manifests.
- No rankings JSON pipeline.
- Output is printed to stdout as JSON summary.
