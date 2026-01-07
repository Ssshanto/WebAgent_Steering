# Run Instructions for Hyperparameter Optimization

## Quick Start (Copy-Paste Ready)

```bash
cd /home/ssshanto/Documents/WebAgent_Steering
./run_optimization.sh
```

**Expected runtime:** ~28 hours (28 configurations × ~1 hour each)  
**What it does:** Sweeps layers {12, 13, 14, 15} × coefficients {2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}  
**Output:** Finds optimal layer and steering coefficient for 'accuracy' prompts

---

## Analyze Results

After completion (or partial runs):
```bash
python3 scripts/analyze_optimization.py
```

**What it shows:**
- Best configuration (layer + coefficient)
- Top 5 configurations ranked by improvement
- Results grouped by layer for easy comparison

---

## Output Files

Results are saved to `results/` directory with naming pattern:
- `L{layer}_a{coeff}_s{seed}.jsonl`

Examples:
- `results/L13_a4.0_s0.jsonl` → Layer 13, α=4.0, seed 0
- `results/L14_a3.0_s0.jsonl` → Layer 14, α=3.0, seed 0

Total: 28 files (4 layers × 7 coefficients)

---

## Analysis Output Example

```
=== HYPERPARAMETER OPTIMIZATION RESULTS ===

🏆 BEST CONFIGURATION:
Layer 13, α=4.0: Base=19.0% | Steered=36.5% | Δ=+17.5%

📊 TOP 5 CONFIGURATIONS:
1. L13, α=4.0:  +17.5%  (Base: 19.0% → Steered: 36.5%)
2. L13, α=4.5:  +15.2%  (Base: 19.0% → Steered: 34.2%)
3. L14, α=3.5:  +14.8%  (Base: 19.0% → Steered: 33.8%)
4. L13, α=3.5:  +13.9%  (Base: 19.0% → Steered: 32.9%)
5. L12, α=4.0:  +12.5%  (Base: 19.0% → Steered: 31.5%)

📈 RESULTS BY LAYER:
Layer 12: Best α=4.0 (+12.5%)
Layer 13: Best α=4.0 (+17.5%) ⭐
Layer 14: Best α=3.5 (+14.8%)
Layer 15: Best α=3.0 (+10.2%)
```

---

## Resume Capability

The script automatically skips completed configurations:
```bash
./run_optimization.sh  # Resumes from where it left off
```

Manual skip: Comment out completed configs in the script.

---

## Troubleshooting

**If script fails:**
- Check Python: `python3 --version` (needs 3.8+)
- Check dependencies: `pip install -r requirements.txt`
- Check CUDA: `nvidia-smi` (optional, will use CPU if unavailable)

**If out of memory:**
- 0.5B model needs ~1-2GB GPU RAM
- Will automatically fall back to CPU if GPU unavailable
- CPU is ~4x slower but will work

**If results look wrong:**
- Verify baseline: Should be ~19% accuracy (0.5B model, 25 tasks)
- Check parse failures: Should be ~45% at baseline
- Steering should improve both accuracy AND reduce parse failures

---

## What to Report Back

After optimization completes:
1. Best layer and coefficient
2. Improvement achieved (Δ%)
3. Top 3-5 configurations for comparison
4. Any unexpected patterns (e.g., layer trends)

---

## Configuration Details

**Fixed parameters:**
- Model: Qwen 2.5 0.5B Instruct
- Prompt: 'accuracy' (positive: "Be accurate and precise...", negative: "Be inaccurate and imprecise...")
- Vector method: 'response' (extracts from last token of generated response)
- Train steps: 200 episodes for vector computation
- Eval steps: 400 episodes for evaluation (200 base + 200 steered)
- Seed: 0 (for reproducibility)
- Tasks: All 25 tasks (18 original + 7 expanded)

**Swept parameters:**
- Layers: {12, 13, 14, 15} (middle-to-late layers of 24-layer model)
- Coefficients: {2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0} (steering strength multiplier)

---

## Notes

- Script is resumable: Already-completed configs are skipped
- Each config takes ~1 hour (depending on hardware)
- GPU recommended but not required (CPU fallback available)
- Results are independent: Safe to run different configs in parallel on different machines

---

*Last updated: 2026-01-05*
