# Quick Verification of Fix

## What Was Changed

**File:** `src/miniwob_steer.py`  
**Lines:** 78-86 (SYSTEM_PROMPT)

**Change:** Reverted to original single-line system prompt from Experiment 8

### Before (BROKEN):
```
"You are a web automation engine. Output action commands.
Strict format rules:
- Output one action per line.
- For single actions: output exactly one line.
- For multiple actions: output multiple lines (e.g., for checkboxes).
- No explanations, no preamble, no lists, no code fences."
```

### After (FIXED):
```
"You are a web automation engine. Output a single action command.
Strict format rules:
- Output exactly one line.
- No explanations, no preamble, no lists, no code fences."
```

---

## Quick Test

Run this to verify the fix:

```bash
./test_diagnostic.sh
```

**Expected Results:**
- Markdown outputs: 0 (down from 80%)
- Verbose outputs: 0 (down from 80%)
- Steered accuracy: >25% (up from 14.8%)
- Improvement: Positive (previously -13.5%)

---

## Full Verification

For complete confidence, re-run the problematic configuration:

```bash
python src/miniwob_steer.py \
    --model-size 0.5b \
    --layer 13 \
    --coeff 4.0 \
    --prompt-type accuracy \
    --vector-method response \
    --tasks all \
    --train-steps 200 \
    --eval-steps 400 \
    --seed 0 \
    --out results/L13_a4.0_s0_FIXED.jsonl
```

**Expected Results:**
- Base accuracy: ~19-28%
- Steered accuracy: ~30-35% (target: +10-15%)
- No markdown/verbose outputs
- Clean single-line actions

---

## Analysis Script

```python
import json

with open('results/L13_a4.0_s0_FIXED.jsonl') as f:
    data = [json.loads(l) for l in f if l.strip()]

base_success = sum(1 for d in data if d.get('base_success'))
steer_success = sum(1 for d in data if d.get('steer_success'))

steer_samples = [d for d in data if 'steer_output' in d]
markdown_count = sum(1 for d in steer_samples if '```' in d['steer_output'])
verbose_count = sum(1 for d in steer_samples if len(d['steer_output']) > 100)

print(f"Base:    {100*base_success/len(data):.1f}%")
print(f"Steered: {100*steer_success/len(data):.1f}%")
print(f"Δ:       {100*(steer_success-base_success)/len(data):+.1f}%")
print(f"\nMarkdown outputs: {markdown_count}/{len(steer_samples)}")
print(f"Verbose outputs:  {verbose_count}/{len(steer_samples)}")
print(f"\nStatus: {'✅ FIXED' if markdown_count == 0 and steer_success > base_success else '❌ STILL BROKEN'}")
```

---

## If Fix Doesn't Work

If the issue persists:
1. Check that changes were saved: `grep "single action command" src/miniwob_steer.py`
2. Verify no other files changed: `git status`
3. Check library versions: `pip list | grep transformers`
4. Try 'format' prompt instead: `--prompt-type format`

---

## Note on Experiment 10

This fix **breaks multi-action task support** (checkboxes, etc.) because we reverted to single-line-only prompt.

**To support both:**
- Keep fix for single-step tasks (original 18 tasks)
- Create separate system prompt for multi-action tasks
- Modify `build_prompt()` to use task-specific prompts

For now, prioritizing hyperparameter optimization on single-step tasks (which includes the core 18 tasks that Exp8 succeeded on).
