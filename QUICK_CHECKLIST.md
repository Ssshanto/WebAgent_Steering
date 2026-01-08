# Quick Investigation Checklist

## ✅ What I Verified

- [x] **Vector polarity:** `pos - neg` (correct direction)
- [x] **Accuracy prompts:** Positive doesn't encourage verbosity
- [x] **System prompt:** Explicitly forbids explanations  
- [x] **Steering formula:** `+= coeff * vec` (correct)
- [x] **Argument parsing:** `--prompt-type` correctly overrides default
- [x] **Seeding:** Vector computation is seeded
- [x] **Recent outputs:** Examined `analyzable_L13_a4.0_s0.json` - all clean

## ⚠️  Minor Issue Found (Not Critical)

- Chat template doesn't use Qwen's system role
- BUT: This was always the case (Exp 8 succeeded with same setup)
- NOT a regression

## 🔍 What I Couldn't Verify (Need Runtime Test)

- [ ] Current model outputs (need to run inference)
- [ ] Whether issue still reproduces
- [ ] Which specific tasks/seeds cause problems
- [ ] Library versions on cvpc server

## 📋 Next Steps

### Option 1: Quick Test (Recommended)
```bash
./test_diagnostic.sh
```
**Time:** 5-10 minutes  
**What it does:** Runs 100 episodes and checks for verbose/markdown outputs

### Option 2: Full Debug Run
```bash
# Single config with detailed logging
python src/miniwob_steer.py \
    --model-size 0.5b \
    --layer 13 \
    --coeff 4.0 \
    --prompt-type accuracy \
    --vector-method response \
    --tasks all \
    --train-steps 200 \
    --eval-steps 100 \
    --seed 0 \
    --out results/debug.jsonl

# Then examine outputs
python3 << 'PY'
import json
with open('results/debug.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d['type'] == 'steered':
            print(f"{d['task']}: {d['output'][:100]}")
PY
```

## 📊 What to Report

If issue reproduces, share:
1. **5-10 example outputs** showing the problem
2. **Which prompt type** was used (check with: `grep "Using prompt type" results/debug.jsonl`)
3. **Base vs steered** - which one is verbose?
4. **Specific tasks** - is it all tasks or specific ones?

## 🎯 Most Likely Scenarios

Based on investigation:

**Scenario 1: No Issue (70% likely)**
- Diagnostic test passes
- Previous report was from different run
- Code is working correctly

**Scenario 2: Environment Issue (20% likely)**  
- Library version changed
- Different GPU/CPU behavior
- Need to verify: `pip list | grep transformers`

**Scenario 3: Misinterpretation (10% likely)**
- Multi-line outputs are CORRECT for multi-action tasks
- Parser failures look like "verbose" outputs
- Actually a parsing issue, not verbosity

**Scenario 4: Actual Bug (<5% likely)**
- Code analysis missed something
- Need concrete evidence to debug further
