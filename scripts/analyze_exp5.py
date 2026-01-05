#!/usr/bin/env python3
"""
Analyze Experiment 5 results: 0.5B model steering suite.

Computes:
- Overall accuracy (base vs steered)
- Per-task breakdown
- Parse failure rates
- Format compliance analysis
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_results(filepath):
    """Load JSONL results file."""
    records = []
    with open(filepath) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def analyze_file(filepath, name):
    """Analyze a single results file."""
    records = load_results(filepath)

    total = len(records)
    base_success = sum(1 for r in records if r.get("base_success", False))
    steer_success = sum(1 for r in records if r.get("steer_success", False))

    # Parse failures (action is None)
    base_parse_fail = sum(1 for r in records if r.get("base_action") is None)
    steer_parse_fail = sum(1 for r in records if r.get("steer_action") is None)

    # Per-task breakdown
    task_stats = defaultdict(lambda: {"total": 0, "base": 0, "steer": 0, "base_parse_fail": 0, "steer_parse_fail": 0})
    for r in records:
        task = r["task"]
        task_stats[task]["total"] += 1
        task_stats[task]["base"] += int(r.get("base_success", False))
        task_stats[task]["steer"] += int(r.get("steer_success", False))
        task_stats[task]["base_parse_fail"] += int(r.get("base_action") is None)
        task_stats[task]["steer_parse_fail"] += int(r.get("steer_action") is None)

    has_steer = "steer_success" in records[0] if records else False

    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}")
    print(f"Total episodes: {total}")
    print(f"Base accuracy: {base_success}/{total} = {base_success/total*100:.1f}%")
    print(f"Base parse failures: {base_parse_fail}/{total} = {base_parse_fail/total*100:.1f}%")

    if has_steer:
        print(f"Steer accuracy: {steer_success}/{total} = {steer_success/total*100:.1f}%")
        print(f"Steer parse failures: {steer_parse_fail}/{total} = {steer_parse_fail/total*100:.1f}%")
        change = (steer_success - base_success) / total * 100
        print(f"Change: {change:+.1f}%")
        parse_change = (steer_parse_fail - base_parse_fail) / total * 100
        print(f"Parse failure change: {parse_change:+.1f}%")

    print(f"\nPer-task breakdown:")
    print(f"{'Task':<25} {'Base':>8} {'Steer':>8} {'Change':>8} {'Parse Fail':>12}")
    print("-" * 65)

    for task in sorted(task_stats.keys()):
        s = task_stats[task]
        base_acc = s["base"] / s["total"] * 100
        if has_steer:
            steer_acc = s["steer"] / s["total"] * 100
            change = steer_acc - base_acc
            parse_info = f"{s['base_parse_fail']}->{s['steer_parse_fail']}"
            print(f"{task:<25} {base_acc:>7.1f}% {steer_acc:>7.1f}% {change:>+7.1f}% {parse_info:>12}")
        else:
            parse_info = f"{s['base_parse_fail']}/{s['total']}"
            print(f"{task:<25} {base_acc:>7.1f}% {'N/A':>8} {'N/A':>8} {parse_info:>12}")

    return {
        "name": name,
        "total": total,
        "base_acc": base_success / total * 100,
        "steer_acc": steer_success / total * 100 if has_steer else None,
        "base_parse_fail": base_parse_fail / total * 100,
        "steer_parse_fail": steer_parse_fail / total * 100 if has_steer else None,
    }


def main():
    results_dir = Path("results")
    exp5_files = sorted(results_dir.glob("exp5_*.jsonl"))

    if not exp5_files:
        print("No exp5_*.jsonl files found in results/")
        sys.exit(1)

    summaries = []
    for f in exp5_files:
        name = f.stem.replace("exp5_0.5b_", "")
        summaries.append(analyze_file(f, name))

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Base':>8} {'Steer':>8} {'Change':>8} {'Parse Δ':>8}")
    print("-" * 65)

    for s in summaries:
        if s["steer_acc"] is not None:
            change = s["steer_acc"] - s["base_acc"]
            parse_change = s["steer_parse_fail"] - s["base_parse_fail"]
            print(f"{s['name']:<30} {s['base_acc']:>7.1f}% {s['steer_acc']:>7.1f}% {change:>+7.1f}% {parse_change:>+7.1f}%")
        else:
            print(f"{s['name']:<30} {s['base_acc']:>7.1f}% {'N/A':>8} {'N/A':>8} {'N/A':>8}")


if __name__ == "__main__":
    main()
