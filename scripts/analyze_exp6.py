#!/usr/bin/env python3
"""
Analyze Experiment 6 results: Reproducibility + Optimization.

Phase 1: Reproducibility validation (seeds 0, 42, 123)
Phase 2: Coefficient optimization (2.0-5.0)
Phase 3: Layer optimization (12-16)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_results(filepath):
    """Load JSONL results file."""
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def compute_metrics(records):
    """Compute accuracy and parse failure metrics."""
    total = len(records)
    if total == 0:
        return None
    
    base_success = sum(1 for r in records if r.get("base_success", False))
    steer_success = sum(1 for r in records if r.get("steer_success", False))
    base_parse_fail = sum(1 for r in records if r.get("base_action") is None)
    steer_parse_fail = sum(1 for r in records if r.get("steer_action") is None)
    
    return {
        "total": total,
        "base_acc": base_success / total * 100,
        "steer_acc": steer_success / total * 100,
        "base_parse_fail": base_parse_fail / total * 100,
        "steer_parse_fail": steer_parse_fail / total * 100,
        "acc_delta": (steer_success - base_success) / total * 100,
        "parse_delta": (steer_parse_fail - base_parse_fail) / total * 100,
    }


def analyze_phase1(results_dir):
    """Phase 1: Reproducibility across seeds."""
    print("\n" + "="*70)
    print("PHASE 1: REPRODUCIBILITY VALIDATION")
    print("="*70)
    print("Best config (accuracy, L14, α=3.0) tested with seeds {0, 42, 123}")
    print()
    
    seeds = [0, 42, 123]
    metrics = []
    
    for seed in seeds:
        filepath = results_dir / f"exp6_seed{seed}.jsonl"
        if not filepath.exists():
            print(f"⚠️  Missing: {filepath.name}")
            continue
        
        records = load_results(filepath)
        m = compute_metrics(records)
        if m:
            metrics.append(m)
            print(f"Seed {seed:3d}: Base={m['base_acc']:5.1f}% | "
                  f"Steer={m['steer_acc']:5.1f}% | "
                  f"Δ={m['acc_delta']:+5.1f}% | "
                  f"Parse: {m['base_parse_fail']:4.1f}%→{m['steer_parse_fail']:4.1f}%")
    
    if len(metrics) >= 2:
        import statistics
        acc_deltas = [m["acc_delta"] for m in metrics]
        mean_delta = statistics.mean(acc_deltas)
        std_delta = statistics.stdev(acc_deltas) if len(acc_deltas) > 1 else 0.0
        
        print()
        print(f"Mean improvement: {mean_delta:+.1f}%")
        print(f"Std deviation:    {std_delta:.1f}%")
        print()
        
        # Validation criteria
        success = True
        if mean_delta < 7.0:
            print(f"❌ FAIL: Mean improvement {mean_delta:.1f}% < 7.0% threshold")
            success = False
        else:
            print(f"✅ PASS: Mean improvement {mean_delta:.1f}% ≥ 7.0%")
        
        if std_delta >= 3.0:
            print(f"⚠️  WARN: Std deviation {std_delta:.1f}% ≥ 3.0% (high variance)")
        else:
            print(f"✅ PASS: Std deviation {std_delta:.1f}% < 3.0%")
        
        all_positive = all(m["acc_delta"] > 0 for m in metrics)
        if not all_positive:
            print(f"❌ FAIL: Not all seeds show positive improvement")
            success = False
        else:
            print(f"✅ PASS: All seeds show positive improvement")
        
        print()
        if success:
            print("🎉 REPRODUCIBILITY VALIDATED")
        else:
            print("⚠️  REPRODUCIBILITY CONCERNS - Further investigation needed")
    
    return metrics


def analyze_phase2(results_dir):
    """Phase 2: Coefficient optimization."""
    print("\n" + "="*70)
    print("PHASE 2: COEFFICIENT OPTIMIZATION")
    print("="*70)
    print("Layer=14, Seeds=0, varying α ∈ {2.0, 2.5, 3.0, 3.5, 4.0, 5.0}")
    print()
    
    coeffs = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    results = []
    
    print(f"{'Coeff':>6} {'Base':>8} {'Steer':>8} {'Change':>8} {'Parse Δ':>9} {'Status':>8}")
    print("-" * 70)
    
    for coeff in coeffs:
        filepath = results_dir / f"exp6_coeff{coeff}.jsonl"
        if not filepath.exists():
            print(f"{coeff:>6.1f} {'MISSING':<60}")
            continue
        
        records = load_results(filepath)
        m = compute_metrics(records)
        if m:
            results.append((coeff, m))
            status = "✓" if m["acc_delta"] > 0 else "✗"
            print(f"{coeff:>6.1f} {m['base_acc']:>7.1f}% {m['steer_acc']:>7.1f}% "
                  f"{m['acc_delta']:>+7.1f}% {m['parse_delta']:>+8.1f}% {status:>8}")
    
    if results:
        best_coeff, best_m = max(results, key=lambda x: x[1]["acc_delta"])
        print()
        print(f"🏆 Best coefficient: α={best_coeff} with {best_m['acc_delta']:+.1f}% improvement")
    
    return results


def analyze_phase3(results_dir):
    """Phase 3: Layer optimization."""
    print("\n" + "="*70)
    print("PHASE 3: LAYER OPTIMIZATION")
    print("="*70)
    print("Coeff=3.0, Seeds=0, varying layer ∈ {12, 13, 14, 15, 16}")
    print()
    
    layers = [12, 13, 14, 15, 16]
    results = []
    
    print(f"{'Layer':>6} {'Base':>8} {'Steer':>8} {'Change':>8} {'Parse Δ':>9} {'Status':>8}")
    print("-" * 70)
    
    for layer in layers:
        filepath = results_dir / f"exp6_layer{layer}.jsonl"
        if not filepath.exists():
            print(f"{layer:>6} {'MISSING':<60}")
            continue
        
        records = load_results(filepath)
        m = compute_metrics(records)
        if m:
            results.append((layer, m))
            status = "✓" if m["acc_delta"] > 0 else "✗"
            print(f"{layer:>6} {m['base_acc']:>7.1f}% {m['steer_acc']:>7.1f}% "
                  f"{m['acc_delta']:>+7.1f}% {m['parse_delta']:>+8.1f}% {status:>8}")
    
    if results:
        best_layer, best_m = max(results, key=lambda x: x[1]["acc_delta"])
        depth_pct = best_layer / 24 * 100  # 0.5B has 24 layers (0-23)
        print()
        print(f"🏆 Best layer: L{best_layer} ({depth_pct:.0f}% depth) with {best_m['acc_delta']:+.1f}% improvement")
    
    return results


def main():
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("❌ results/ directory not found")
        sys.exit(1)
    
    # Detect which phases have results
    has_phase1 = any((results_dir / f"exp6_seed{s}.jsonl").exists() for s in [0, 42, 123])
    has_phase2 = any((results_dir / f"exp6_coeff{c}.jsonl").exists() for c in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    has_phase3 = any((results_dir / f"exp6_layer{l}.jsonl").exists() for l in [12, 13, 14, 15, 16])
    
    if not (has_phase1 or has_phase2 or has_phase3):
        print("❌ No exp6_*.jsonl files found in results/")
        print("Run: ./run_exp6_validate.sh [1|2|3|all]")
        sys.exit(1)
    
    # Analyze each phase
    if has_phase1:
        analyze_phase1(results_dir)
    
    if has_phase2:
        analyze_phase2(results_dir)
    
    if has_phase3:
        analyze_phase3(results_dir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
