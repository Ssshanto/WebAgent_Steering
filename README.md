# MiniWoB Steering POC (Zero-Shot vs Steered)

Minimal proof-of-concept for representation steering on MiniWoB single-step tasks.
No fine-tuning and no saved dataset; the steering vector is computed on the fly.

## Setup
1) Install deps:
   ```bash
   pip install -r requirements.txt
   ```
2) MiniWoB uses Selenium. Install Chrome/Chromedriver and ensure `chromedriver` is on `PATH`.
   - Ubuntu: `sudo apt-get install -y chromium-chromedriver`

Kaggle GPU notebook: `miniwob_steer_poc.ipynb`.

## Tasks (hardcoded, single-step)
```
click-test
click-test-2
click-test-transfer
click-button
click-link
click-color
click-dialog
click-dialog-2
click-pie
click-pie-nodelay
click-shape
click-tab
click-widget
focus-text
focus-text-2
grid-coordinate
identify-shape
unicode-test
```

## Run
```bash
python src/miniwob_steer.py \
  --model-size 0.5b \
  --train-steps 200 \
  --eval-steps 200 \
  --layer 20 \
  --coeff 1.0 \
  --out miniwob_results.jsonl
```

```bash
python src/miniwob_steer.py --model-size 3b --train-steps 200 --eval-steps 200
```

## Output
- Progress bars show running base vs steered accuracy.
- Results are saved as JSONL (one line per eval step).

## Steering Details
- The steering vector is computed from the **generated outputs** of two prompt variants
  (instruction-only vs natural-language), not just from the prompt tokens. This aligns the
  vector with the model's actual output behavior.
## Action Format
The model must output a single action line:
```
click ref=<int>
type ref=<int> text="<text>"
```
