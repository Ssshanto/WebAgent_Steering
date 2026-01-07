#!/usr/bin/env python3
"""
Analyze hyperparameter optimization results.

Finds best layer/coefficient combination for accuracy prompt.
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


def main():
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("❌ results/ directory not found")
        sys.exit(1)
    
    # Find all result files matching pattern: L{layer}_a{coeff}_s{seed}.jsonl
    result_files = sorted(results_dir.glob("L*_a*_s*.jsonl"))
    
    if not result_files:
        print("❌ No optimization results found")
        print("Run: ./run_optimization.sh")
        sys.exit(1)
    
    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Found {len(result_files)} result files")
    print()
    
    # Parse and analyze each result
    results = []
    for filepath in result_files:
        # Parse filename: L13_a4.0_s0.jsonl
        parts = filepath.stem.split('_')
        layer = int(parts[0][1:])  # L13 -> 13
        coeff = float(parts[1][1:])  # a4.0 -> 4.0
        seed = int(parts[2][1:])  # s0 -> 0
        
        records = load_results(filepath)
        metrics = compute_metrics(records)
        
        if metrics:
            results.append({
                "layer": layer,
                "coeff": coeff,
                "seed": seed,
                "metrics": metrics,
                "filepath": filepath.name
            })
    
    if not results:
        print("❌ No valid results found")
        sys.exit(1)
    
    # Group by layer
    by_layer = defaultdict(list)
    for r in results:
        by_layer[r["layer"]].append(r)
    
    # Print results grouped by layer
    print("="*80)
    print("RESULTS BY LAYER")
    print("="*80)
    print()
    
    for layer in sorted(by_layer.keys()):
        layer_results = sorted(by_layer[layer], key=lambda x: x["coeff"])
        
        print(f"LAYER {layer} ({len(layer_results)} configurations)")
        print("-" * 80)
        print(f"{'Coeff':>6} {'Base':>8} {'Steer':>8} {'Change':>8} {'Parse Δ':>9} {'Status':>8}")
        print("-" * 80)
        
        for r in layer_results:
            m = r["metrics"]
            status = "✓" if m["acc_delta"] > 0 else ("=" if m["acc_delta"] == 0 else "✗")
            
            print(f"{r['coeff']:>6.1f} {m['base_acc']:>7.1f}% {m['steer_acc']:>7.1f}% "
                  f"{m['acc_delta']:>+7.1f}% {m['parse_delta']:>+8.1f}% {status:>8}")
        
        print()
    
    # Find best configuration
    print("="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    
    best = max(results, key=lambda x: x["metrics"]["acc_delta"])
    m = best["metrics"]
    
    print(f"Layer:       {best['layer']}")
    print(f"Coefficient: {best['coeff']}")
    print(f"Seed:        {best['seed']}")
    print()
    print(f"Base accuracy:    {m['base_acc']:.1f}%")
    print(f"Steered accuracy: {m['steer_acc']:.1f}%")
    print(f"Improvement:      {m['acc_delta']:+.1f}%")
    print()
    print(f"Parse failures:   {m['base_parse_fail']:.1f}% → {m['steer_parse_fail']:.1f}% ({m['parse_delta']:+.1f}%)")
    print()
    print(f"Result file: {best['filepath']}")
    
    # Top 5 configurations
    print()
    print("="*80)
    print("TOP 5 CONFIGURATIONS")
    print("="*80)
    print()
    print(f"{'Rank':>4} {'Layer':>6} {'Coeff':>6} {'Improvement':>12} {'Steered Acc':>12}")
    print("-" * 80)
    
    top5 = sorted(results, key=lambda x: x["metrics"]["acc_delta"], reverse=True)[:5]
    for i, r in enumerate(top5, 1):
        m = r["metrics"]
        print(f"{i:>4} {r['layer']:>6} {r['coeff']:>6.1f} {m['acc_delta']:>+11.1f}% {m['steer_acc']:>11.1f}%")
    
    print()
    print("="*80)


if __name__ == "__main__":
    main()
