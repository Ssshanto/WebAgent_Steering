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
    # Small models (< 1B) - primary hypothesis targets
    "smollm-360m": {"family": "SmolLM", "size": "360M", "params": 0.36},
    "0.5b": {"family": "Qwen", "size": "0.5B", "params": 0.5},
    "qwen-coder-0.5b": {"family": "Qwen", "size": "0.5B", "params": 0.5},
    # Medium-small models (1B-2B)
    "llama-1b": {"family": "Llama", "size": "1B", "params": 1.0},
    "gemma-1b": {"family": "Gemma", "size": "1B", "params": 1.0},
    "tinyllama-1.1b": {"family": "TinyLlama", "size": "1.1B", "params": 1.1},
    "opt-iml-1.3b": {"family": "OPT", "size": "1.3B", "params": 1.3},
    "qwen-1.5b": {"family": "Qwen", "size": "1.5B", "params": 1.5},
    "stablelm-1.6b": {"family": "StableLM", "size": "1.6B", "params": 1.6},
    "smollm-1.7b": {"family": "SmolLM", "size": "1.7B", "params": 1.7},
    "gemma-2b": {"family": "Gemma", "size": "2B", "params": 2.0},
    # Larger models (> 2B)
    "llama-3b": {"family": "Llama", "size": "3B", "params": 3.0},
    "3b": {"family": "Qwen", "size": "3B", "params": 3.0},
    "phi-3.8b": {"family": "Phi", "size": "3.8B", "params": 3.8},
    # VLM
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
    """Analyze a single model's results with extended metrics."""
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

        # Parse-conditioned accuracy: accuracy among parseable outputs only
        base_parseable = [r for r in steered if r.get("base_action") is not None]
        steer_parseable = [r for r in steered if r.get("steer_action") is not None]
        base_cond_acc = sum(1 for r in base_parseable if r.get("base_success", False)) / max(1, len(base_parseable))
        steer_cond_acc = sum(1 for r in steer_parseable if r.get("steer_success", False)) / max(1, len(steer_parseable))

        # Action change analysis: how often does steering change the action?
        action_changed = 0
        correct_to_wrong = 0  # Steering broke a correct answer
        wrong_to_correct = 0  # Steering fixed a wrong answer
        both_correct = 0
        both_wrong = 0
        for r in steered:
            base_act = r.get("base_action")
            steer_act = r.get("steer_action")
            base_ok = r.get("base_success", False)
            steer_ok = r.get("steer_success", False)

            if base_act != steer_act:
                action_changed += 1

            if base_ok and steer_ok:
                both_correct += 1
            elif base_ok and not steer_ok:
                correct_to_wrong += 1
            elif not base_ok and steer_ok:
                wrong_to_correct += 1
            else:
                both_wrong += 1

        result["comparison"] = {
            "base_accuracy": base_success / len(steered),
            "steer_accuracy": steer_success / len(steered),
            "delta": (steer_success - base_success) / len(steered),
            "base_parse_fail": base_parse_fail / len(steered),
            "steer_parse_fail": steer_parse_fail / len(steered),
            "parse_fail_reduction": (base_parse_fail - steer_parse_fail) / max(1, base_parse_fail),
            "total": len(steered),
            # Extended metrics
            "base_cond_accuracy": base_cond_acc,
            "steer_cond_accuracy": steer_cond_acc,
            "cond_delta": steer_cond_acc - base_cond_acc,
            "action_change_rate": action_changed / len(steered),
            "wrong_to_correct": wrong_to_correct / len(steered),
            "correct_to_wrong": correct_to_wrong / len(steered),
            "net_benefit": (wrong_to_correct - correct_to_wrong) / len(steered),
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


def print_extended_metrics(results):
    """Print extended metrics: parse-conditioned accuracy and action changes."""
    print("\n" + "=" * 90)
    print("EXTENDED METRICS: ACTION-SPACE UNDERSTANDING")
    print("=" * 90)

    print("\n{:<15} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "Base|Parse", "Steer|Parse", "Cond Δ", "Act Chg%", "Fix%", "Break%"
    ))
    print("-" * 90)

    for r in results:
        if "comparison" not in r:
            continue
        comp = r["comparison"]
        print("{:<15} {:>10.1%} {:>10.1%} {:>10} {:>10.1%} {:>10.1%} {:>10.1%}".format(
            r["model"],
            comp.get("base_cond_accuracy", 0),
            comp.get("steer_cond_accuracy", 0),
            f"{comp.get('cond_delta', 0):+.1%}",
            comp.get("action_change_rate", 0),
            comp.get("wrong_to_correct", 0),
            comp.get("correct_to_wrong", 0),
        ))


def print_steerability_analysis(results):
    """Analyze correlation between parse failure and steerability."""
    print("\n" + "=" * 90)
    print("STEERABILITY PREDICTOR: PARSE FAILURE vs STEERING BENEFIT")
    print("=" * 90)

    # Group by parse failure rate
    high_parse_fail = []  # > 20% parse failures
    med_parse_fail = []   # 5-20% parse failures
    low_parse_fail = []   # < 5% parse failures

    for r in results:
        if "comparison" not in r:
            continue
        pf = r["comparison"]["base_parse_fail"]
        if pf > 0.20:
            high_parse_fail.append(r)
        elif pf > 0.05:
            med_parse_fail.append(r)
        else:
            low_parse_fail.append(r)

    def avg_delta(group):
        if not group:
            return 0
        return sum(r["comparison"]["delta"] for r in group) / len(group)

    print(f"\nHigh parse fail (>20%): {len(high_parse_fail)} models, avg Δ = {avg_delta(high_parse_fail):+.1%}")
    for r in high_parse_fail:
        print(f"  - {r['model']}: parse_fail={r['comparison']['base_parse_fail']:.1%}, Δ={r['comparison']['delta']:+.1%}")

    print(f"\nMed parse fail (5-20%): {len(med_parse_fail)} models, avg Δ = {avg_delta(med_parse_fail):+.1%}")
    for r in med_parse_fail:
        print(f"  - {r['model']}: parse_fail={r['comparison']['base_parse_fail']:.1%}, Δ={r['comparison']['delta']:+.1%}")

    print(f"\nLow parse fail (<5%): {len(low_parse_fail)} models, avg Δ = {avg_delta(low_parse_fail):+.1%}")
    for r in low_parse_fail:
        print(f"  - {r['model']}: parse_fail={r['comparison']['base_parse_fail']:.1%}, Δ={r['comparison']['delta']:+.1%}")

    # Hypothesis test
    print("\n" + "-" * 50)
    high_avg = avg_delta(high_parse_fail)
    low_avg = avg_delta(low_parse_fail)
    if high_avg > low_avg + 0.05:
        print("HYPOTHESIS SUPPORTED: High parse-fail models benefit more from steering")
    elif abs(high_avg - low_avg) < 0.03:
        print("INCONCLUSIVE: No clear relationship between parse failure and steering benefit")
    else:
        print("HYPOTHESIS NOT SUPPORTED: Low parse-fail models benefit equally or more")


def print_size_analysis(results):
    """Analyze steering effectiveness by model size."""
    print("\n" + "=" * 90)
    print("SIZE ANALYSIS: SMALL vs LARGE MODELS")
    print("=" * 90)

    small = []   # < 1B
    medium = []  # 1B - 2B
    large = []   # > 2B

    for r in results:
        if "comparison" not in r:
            continue
        params = r.get("info", {}).get("params", 0)
        if params < 1.0:
            small.append(r)
        elif params <= 2.0:
            medium.append(r)
        else:
            large.append(r)

    def avg_delta(group):
        if not group:
            return 0
        return sum(r["comparison"]["delta"] for r in group) / len(group)

    def avg_net_benefit(group):
        if not group:
            return 0
        return sum(r["comparison"].get("net_benefit", 0) for r in group) / len(group)

    print(f"\nSmall (<1B): {len(small)} models")
    print(f"  Avg Δ = {avg_delta(small):+.1%}")
    print(f"  Avg Net Benefit = {avg_net_benefit(small):+.1%}")
    for r in small:
        print(f"    - {r['model']} ({r.get('info', {}).get('size', '?')}): Δ={r['comparison']['delta']:+.1%}")

    print(f"\nMedium (1B-2B): {len(medium)} models")
    print(f"  Avg Δ = {avg_delta(medium):+.1%}")
    print(f"  Avg Net Benefit = {avg_net_benefit(medium):+.1%}")
    for r in medium:
        print(f"    - {r['model']} ({r.get('info', {}).get('size', '?')}): Δ={r['comparison']['delta']:+.1%}")

    print(f"\nLarge (>2B): {len(large)} models")
    print(f"  Avg Δ = {avg_delta(large):+.1%}")
    print(f"  Avg Net Benefit = {avg_net_benefit(large):+.1%}")
    for r in large:
        print(f"    - {r['model']} ({r.get('info', {}).get('size', '?')}): Δ={r['comparison']['delta']:+.1%}")

    # Hypothesis test
    print("\n" + "-" * 50)
    small_avg = avg_delta(small)
    large_avg = avg_delta(large)
    if small_avg > large_avg + 0.05:
        print("HYPOTHESIS SUPPORTED: Small models benefit more from steering")
    elif abs(small_avg - large_avg) < 0.03:
        print("INCONCLUSIVE: No clear size-dependent effect")
    else:
        print("HYPOTHESIS NOT SUPPORTED: Large models benefit equally or more")

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
    print_extended_metrics(results)
    print_steerability_analysis(results)
    print_size_analysis(results)

if __name__ == "__main__":
    main()
