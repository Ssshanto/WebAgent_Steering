# Experiment 6: Validation and Optimization

## Overview

Following the success of Experiment 5 (0.5B model steering achieved +9.7% improvement), this experiment validates reproducibility and optimizes hyperparameters.

**Best Config from Exp 5:**
- Prompt: `accuracy` 
- Layer: L14 (58% depth)
- Coefficient: α=3.0
- Result: 19.0% → 28.7% (+9.7%, -32.8% parse failures)

---

## Phases

### Phase 1: Reproducibility Validation
**Goal:** Verify that the +9.7% improvement is stable across random seeds.

**What it runs:**
- Best config (accuracy, L14, α=3.0) with seeds {0, 42, 123}
- 200 train episodes, 400 eval episodes per seed
- All 18 single-step tasks

**Success criteria:**
- Mean improvement ≥ 7% (allows 2.7% variance from reported 9.7%)
- Standard deviation < 3%
- All seeds show positive improvement

**Time:** ~3 hours (3 runs × ~1 hour each)

### Phase 2: Coefficient Optimization
**Goal:** Find optimal steering strength.

**What it runs:**
- Fixed: Layer=14, Prompt=accuracy, Seed=0
- Sweep: α ∈ {2.0, 2.5, 3.0, 3.5, 4.0, 5.0}

**Time:** ~6 hours (6 runs × ~1 hour each)

### Phase 3: Layer Optimization
**Goal:** Confirm L14 is optimal intervention point.

**What it runs:**
- Fixed: Coeff=3.0, Prompt=accuracy, Seed=0
- Sweep: Layer ∈ {12, 13, 14, 15, 16} (50%-67% depth)

**Time:** ~6 hours (5 runs × ~1.2 hours each)

---

## Run Instructions

### Prerequisites
```bash
# Ensure environment is set up
pip install -r requirements.txt
sudo apt-get install -y chromium-chromedriver

# Verify model is downloaded
python download_model.py  # if not already cached
```

### Running Experiments

**Phase 1 (Reproducibility) - PRIORITY:**
```bash
./run_exp6_validate.sh 1
```

**Phase 2 (Coefficient Optimization):**
```bash
./run_exp6_validate.sh 2
```

**Phase 3 (Layer Optimization):**
```bash
./run_exp6_validate.sh 3
```

**All phases sequentially:**
```bash
./run_exp6_validate.sh all  # ~15 hours total
```

### Analyzing Results

**After any phase completes:**
```bash
python scripts/analyze_exp6.py
```

**Output includes:**
- Phase 1: Reproducibility validation with pass/fail criteria
- Phase 2: Best coefficient with improvement ranking
- Phase 3: Best layer with depth analysis

---

## Expected Outputs

### Files Created
```
results/exp6_seed0.jsonl       # Phase 1, seed 0
results/exp6_seed42.jsonl      # Phase 1, seed 42
results/exp6_seed123.jsonl     # Phase 1, seed 123
results/exp6_coeff2.0.jsonl    # Phase 2, α=2.0
results/exp6_coeff2.5.jsonl    # Phase 2, α=2.5
results/exp6_coeff3.0.jsonl    # Phase 2, α=3.0
results/exp6_coeff3.5.jsonl    # Phase 2, α=3.5
results/exp6_coeff4.0.jsonl    # Phase 2, α=4.0
results/exp6_coeff5.0.jsonl    # Phase 2, α=5.0
results/exp6_layer12.jsonl     # Phase 3, L12
results/exp6_layer13.jsonl     # Phase 3, L13
results/exp6_layer14.jsonl     # Phase 3, L14
results/exp6_layer15.jsonl     # Phase 3, L15
results/exp6_layer16.jsonl     # Phase 3, L16
```

### Analysis Output Format

**Phase 1 Example:**
```
=== PHASE 1: REPRODUCIBILITY VALIDATION ===
Seed   0: Base=19.0% | Steer=28.7% | Δ=+9.7% | Parse: 45.2%→12.5%
Seed  42: Base=18.5% | Steer=27.8% | Δ=+9.3% | Parse: 46.0%→13.2%
Seed 123: Base=19.2% | Steer=29.0% | Δ=+9.8% | Parse: 44.8%→12.0%

Mean improvement: +9.6%
Std deviation:    0.3%

✅ PASS: Mean improvement 9.6% ≥ 7.0%
✅ PASS: Std deviation 0.3% < 3.0%
✅ PASS: All seeds show positive improvement

🎉 REPRODUCIBILITY VALIDATED
```

**Phase 2 Example:**
```
=== PHASE 2: COEFFICIENT OPTIMIZATION ===
 Coeff     Base   Steer  Change  Parse Δ  Status
-------------------------------------------------------
   2.0    19.0%   26.5%   +7.5%   -28.0%       ✓
   2.5    19.0%   27.8%   +8.8%   -30.5%       ✓
   3.0    19.0%   28.7%   +9.7%   -32.8%       ✓
   3.5    19.0%   28.2%   +9.2%   -31.0%       ✓
   4.0    19.0%   27.0%   +8.0%   -29.5%       ✓
   5.0    19.0%   25.5%   +6.5%   -26.0%       ✓

🏆 Best coefficient: α=3.0 with +9.7% improvement
```

---

## Troubleshooting

### Common Issues

**"No exp6_*.jsonl files found"**
- Run the experiments first: `./run_exp6_validate.sh [1|2|3]`

**"chromium-browser not found"**
- Install Chrome: `sudo apt-get install -y chromium-chromedriver`
- Or update line 370 in `src/miniwob_steer.py` with correct Chrome path

**"CUDA out of memory"**
- The 0.5B model should fit on most GPUs (requires ~1-2GB)
- Fallback: Code will use CPU automatically (much slower)

**Slow execution**
- Phase 1: ~3 hours on GPU, ~12 hours on CPU
- Consider running phases in parallel if multiple GPUs available

---

## Next Steps After Validation

### If Phase 1 Passes (Mean ≥ 7%, StdDev < 3%)
✅ POC is reproducible  
→ Proceed to Phase 2 & 3 for optimization  
→ Document findings in RESEARCH.md  
→ Prepare for 3B model investigation

### If Phase 1 Fails
⚠️ Investigate variance sources:
- Check for environmental differences (CUDA version, library versions)
- Examine per-task variance (some tasks may be unstable)
- Consider increasing eval episodes for statistical power
- Review if parse failure reduction is consistent even if accuracy varies

### After Optimization (Phase 2 & 3)
- Update best config in RESEARCH.md
- Run final validation with optimal (layer, coefficient) pair
- Investigate mechanism: Why does this layer/coefficient work?
- Begin 3B model analysis (Why does 3B fail where 0.5B succeeds?)

---

## Technical Details

### Model: Qwen 2.5 0.5B Instruct
- 24 layers (indexed 0-23)
- Layer 14 = 62.5% depth
- Context: 32K tokens
- Format: Requires chat template

### Task Setup
- 18 single-step MiniWob++ tasks
- Train: 200 episodes for vector computation
- Eval: 400 episodes for accuracy measurement
- Metric: Success rate (reward > 0) and parse failure rate

### Steering Method
- Contrastive Activation Addition (CAA)
- Positive prompt: "Be accurate and precise..."
- Negative prompt: "Be inaccurate and imprecise..."
- Vector computed from response activations (last token)
- Applied during generation: `activation += α * vector`

---

*Last Updated: 2026-01-05*
