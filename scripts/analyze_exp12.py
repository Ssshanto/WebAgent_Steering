#!/usr/bin/env python3
"""
Analyze Experiment 12: Multi-Model Scaling Results

Compares steering effectiveness across multiple model families.
"""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results/exp12")

MODEL_INFO = {
    "llama-1b": {"family": "Llama", "size": "1B", "params": 1.0},
    "qwen-1.5b": {"family": "Qwen", "size": "1.5B", "params": 1.5},
    "smollm-1.7b": {"family": "SmolLM", "size": "1.7B", "params": 1.7},
    "gemma-2b": {"family": "Gemma", "size": "2B", "params": 2.0},
    "llama-3b": {"family": "Llama", "size": "3B", "params": 3.0},
    "phi-3.8b": {"family": "Phi", "size": "3.8B", "params": 3.8},
    "qwen-vl-2b": {"family": "Qwen-VL", "size": "2B", "params": 2.0, "vlm": True},
}

def load_results(filepath):
    """Load JSONL results file."""
    if not filepath.exists():
        return None
    
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def analyze_model(model_name):
    """Analyze a single model's results."""
    baseline_file = RESULTS_DIR / f"{model_name}_baseline.jsonl"
    steered_file = RESULTS_DIR / f"{model_name}_steered.jsonl"
    
    baseline = load_results(baseline_file)
    steered = load_results(steered_file)
    
    if baseline is None and steered is None:
        return None
    
    result = {
        "model": model_name,
        "info": MODEL_INFO.get(model_name, {}),
    }
    
    if baseline:
        base_success = sum(1 for r in baseline if r.get("base_success", False))
        base_parse_fail = sum(1 for r in baseline if r.get("base_action") is None)
        result["baseline"] = {
            "accuracy": base_success / len(baseline),
            "parse_fail": base_parse_fail / len(baseline),
            "total": len(baseline),
        }
    
    if steered:
        # Get baseline from steered file (same episodes)
        base_success = sum(1 for r in steered if r.get("base_success", False))
        steer_success = sum(1 for r in steered if r.get("steer_success", False))
        base_parse_fail = sum(1 for r in steered if r.get("base_action") is None)
        steer_parse_fail = sum(1 for r in steered if r.get("steer_action") is None)
        
        result["comparison"] = {
            "base_accuracy": base_success / len(steered),
            "steer_accuracy": steer_success / len(steered),
            "delta": (steer_success - base_success) / len(steered),
            "base_parse_fail": base_parse_fail / len(steered),
            "steer_parse_fail": steer_parse_fail / len(steered),
            "parse_fail_reduction": (base_parse_fail - steer_parse_fail) / max(1, base_parse_fail),
            "total": len(steered),
        }
    
    return result

def print_results_table(results):
    """Print results as formatted table."""
    print("\n" + "=" * 90)
    print("EXPERIMENT 12: MULTI-MODEL SCALING RESULTS")
    print("=" * 90)
    
    print("\n{:<15} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12}".format(
        "Model", "Size", "Base Acc", "Steer Acc", "Δ", "Parse↓", "Family"
    ))
    print("-" * 90)
    
    for r in results:
        model = r["model"]
        info = r.get("info", {})
        
        if "comparison" in r:
            comp = r["comparison"]
            delta_str = f"{comp['delta']:+.1%}"
            delta_color = "✅" if comp["delta"] > 0.05 else ("⚠️" if comp["delta"] > 0 else "❌")
            
            print("{:<15} {:>8} {:>10.1%} {:>10.1%} {:>10} {:>10.1%} {:>12}".format(
                model,
                info.get("size", "?"),
                comp["base_accuracy"],
                comp["steer_accuracy"],
                f"{delta_color} {delta_str}",
                comp["parse_fail_reduction"] if comp["base_parse_fail"] > 0 else 0,
                info.get("family", "?"),
            ))
        elif "baseline" in r:
            print("{:<15} {:>8} {:>10.1%} {:>10} {:>10} {:>12} {:>12}".format(
                model,
                info.get("size", "?"),
                r["baseline"]["accuracy"],
                "-",
                "-",
                "-",
                info.get("family", "?"),
            ))

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    # Count success levels
    strong = 0  # Δ > +10%
    moderate = 0  # Δ > +5%
    weak = 0  # Δ > 0%
    negative = 0  # Δ < 0%
    
    deltas = []
    for r in results:
        if "comparison" in r:
            delta = r["comparison"]["delta"]
            deltas.append((r["model"], delta))
            if delta > 0.10:
                strong += 1
            elif delta > 0.05:
                moderate += 1
            elif delta > 0:
                weak += 1
            else:
                negative += 1
    
    total = len(deltas)
    
    print(f"\nTotal models tested: {total}")
    print(f"  Strong success (Δ > +10%): {strong}")
    print(f"  Moderate success (Δ > +5%): {moderate}")
    print(f"  Weak success (Δ > 0%): {weak}")
    print(f"  Negative (Δ < 0%): {negative}")
    
    # Success criteria
    print("\n" + "-" * 50)
    if strong >= 3:
        print("🏆 STRONG SUCCESS: Δ > +10% on ≥3 models")
    elif strong + moderate >= 4:
        print("✅ MODERATE SUCCESS: Δ > +5% on ≥4 models")
    elif strong + moderate >= 2:
        print("⚠️  WEAK SUCCESS: Δ > +5% on 2-3 models")
    else:
        print("❌ FAILURE: Δ > +5% on <2 models")
    
    # Best and worst
    if deltas:
        deltas.sort(key=lambda x: x[1], reverse=True)
        print(f"\n📈 Best:  {deltas[0][0]} ({deltas[0][1]:+.1%})")
        print(f"📉 Worst: {deltas[-1][0]} ({deltas[-1][1]:+.1%})")
        
        avg_delta = sum(d[1] for d in deltas) / len(deltas)
        print(f"📊 Avg:   {avg_delta:+.1%}")

def print_by_family(results):
    """Group results by model family."""
    print("\n" + "=" * 90)
    print("RESULTS BY FAMILY")
    print("=" * 90)
    
    families = defaultdict(list)
    for r in results:
        if "comparison" in r:
            family = r.get("info", {}).get("family", "Unknown")
            families[family].append(r)
    
    for family, models in sorted(families.items()):
        deltas = [r["comparison"]["delta"] for r in models]
        avg = sum(deltas) / len(deltas)
        print(f"\n{family}: {len(models)} model(s), avg Δ = {avg:+.1%}")
        for r in models:
            print(f"  - {r['model']}: {r['comparison']['delta']:+.1%}")

def main():
    if not RESULTS_DIR.exists():
        print(f"No results directory: {RESULTS_DIR}")
        return
    
    # Collect all results
    results = []
    for model in MODEL_INFO.keys():
        r = analyze_model(model)
        if r:
            results.append(r)
    
    if not results:
        print("No results found in", RESULTS_DIR)
        return
    
    # Sort by model size
    results.sort(key=lambda x: x.get("info", {}).get("params", 0))
    
    # Print analysis
    print_results_table(results)
    print_summary(results)
    print_by_family(results)

if __name__ == "__main__":
    main()
