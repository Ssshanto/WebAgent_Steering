# Run Instructions for Experiment 6

## Quick Start (Copy-Paste Ready)

### Phase 1: Reproducibility Validation (PRIORITY)
```bash
cd /home/ssshanto/Documents/WebAgent_Steering
./run_exp6_validate.sh 1
python3 scripts/analyze_exp6.py
```

**Expected runtime:** ~3 hours  
**What it does:** Validates that best config (accuracy, L14, α=3.0) is reproducible across 3 seeds  
**Success criteria:** Mean improvement ≥7%, StdDev <3%, all seeds positive

---

### Phase 2: Coefficient Optimization
```bash
cd /home/ssshanto/Documents/WebAgent_Steering
./run_exp6_validate.sh 2
python3 scripts/analyze_exp6.py
```

**Expected runtime:** ~6 hours  
**What it does:** Tests α ∈ {2.0, 2.5, 3.0, 3.5, 4.0, 5.0} to find optimal steering strength  
**Output:** Best coefficient with improvement ranking

---

### Phase 3: Layer Optimization
```bash
cd /home/ssshanto/Documents/WebAgent_Steering
./run_exp6_validate.sh 3
python3 scripts/analyze_exp6.py
```

**Expected runtime:** ~6 hours  
**What it does:** Tests layers {12, 13, 14, 15, 16} to confirm L14 is optimal  
**Output:** Best layer with depth analysis

---

### All Phases (Sequential)
```bash
cd /home/ssshanto/Documents/WebAgent_Steering
./run_exp6_validate.sh all
python3 scripts/analyze_exp6.py
```

**Expected runtime:** ~15 hours  
**Runs:** Phase 1 → Phase 2 → Phase 3 sequentially

---

## Pre-Flight Check

Run this before starting experiments:
```bash
cd /home/ssshanto/Documents/WebAgent_Steering
bash verify_exp6_setup.sh
```

Expected output: `✅ All checks passed! Ready to run experiments.`

---

## Output Files

Results will be saved to `results/` directory:

**Phase 1:**
- `results/exp6_seed0.jsonl`
- `results/exp6_seed42.jsonl`
- `results/exp6_seed123.jsonl`

**Phase 2:**
- `results/exp6_coeff2.0.jsonl`
- `results/exp6_coeff2.5.jsonl`
- `results/exp6_coeff3.0.jsonl`
- `results/exp6_coeff3.5.jsonl`
- `results/exp6_coeff4.0.jsonl`
- `results/exp6_coeff5.0.jsonl`

**Phase 3:**
- `results/exp6_layer12.jsonl`
- `results/exp6_layer13.jsonl`
- `results/exp6_layer14.jsonl`
- `results/exp6_layer15.jsonl`
- `results/exp6_layer16.jsonl`

---

## Analysis Output Example

After running Phase 1 and analyzing:
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

---

## Troubleshooting

**If Phase 1 analysis shows FAIL:**
- Check individual seed results for patterns
- Examine if specific tasks are highly variable
- Consider the parse failure reduction (may be consistent even if accuracy varies)

**If scripts don't run:**
- Verify: `bash verify_exp6_setup.sh`
- Check Python: `python3 --version` (needs 3.8+)
- Check CUDA: `nvidia-smi` (optional, will use CPU if unavailable)

**If out of memory:**
- 0.5B model needs ~1-2GB GPU RAM
- Will automatically fall back to CPU if GPU unavailable
- CPU is ~4x slower but will work

---

## What to Report Back

After Phase 1 completes, report:
1. ✅/❌ Reproducibility validation result
2. Mean improvement across 3 seeds
3. Standard deviation
4. Any anomalies or unexpected patterns

After Phase 2 & 3 complete, report:
1. Best coefficient (α) and its improvement
2. Best layer and its improvement
3. Whether optimization found better config than original (L14, α=3.0)

---

## Notes

- Each phase can be run independently
- Analysis script auto-detects which phases have results
- Safe to re-run analysis multiple times
- Safe to re-run experiments (will overwrite previous results)

---

*Implementation verified: 2026-01-05*
