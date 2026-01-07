#!/usr/bin/env python3
"""
Analyze Experiment 10 results: Expanded Action Space Validation.

Categorizes results by interaction type:
- Simple Click
- Multi-Select
- Dropdown
- Semantic Type
- Logic
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


# Task categories (from src/miniwob_steer.py)
TASK_CATEGORIES = {
    "simple_click": [
        "click-test", "click-test-2", "click-test-transfer", "click-button",
        "click-link", "click-color", "click-dialog", "click-dialog-2",
        "click-pie", "click-pie-nodelay", "click-shape", "click-tab", "click-widget"
    ],
    "simple_type": [
        "focus-text", "focus-text-2", "unicode-test"
    ],
    "multi_select": [
        "click-checkboxes", "click-option"
    ],
    "dropdown": [
        "choose-list", "choose-date"
    ],
    "semantic_type": [
        "enter-date", "enter-time"
    ],
    "logic": [
        "guess-number"
    ],
    "other": [
        "grid-coordinate", "identify-shape"
    ]
}


def get_task_category(task_name):
    """Get category for a task."""
    for category, tasks in TASK_CATEGORIES.items():
        if task_name in tasks:
            return category
    return "unknown"


def load_results(filepath):
    """Load JSONL results file."""
    records = []
    with open(filepath) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Skip corrupted lines
    return records


def analyze_by_category(records):
    """Analyze results grouped by task category."""
    category_stats = defaultdict(lambda: {
        "total": 0,
        "base_success": 0,
        "steer_success": 0,
        "base_parse_fail": 0,
        "steer_parse_fail": 0,
        "tasks": defaultdict(lambda: {"base": 0, "steer": 0, "total": 0})
    })
    
    for r in records:
        task = r["task"]
        category = get_task_category(task)
        
        cat_stat = category_stats[category]
        cat_stat["total"] += 1
        cat_stat["base_success"] += int(r.get("base_success", False))
        cat_stat["steer_success"] += int(r.get("steer_success", False))
        cat_stat["base_parse_fail"] += int(r.get("base_action") is None)
        cat_stat["steer_parse_fail"] += int(r.get("steer_action") is None)
        
        # Per-task within category
        cat_stat["tasks"][task]["total"] += 1
        cat_stat["tasks"][task]["base"] += int(r.get("base_success", False))
        cat_stat["tasks"][task]["steer"] += int(r.get("steer_success", False))
    
    return category_stats


def print_category_analysis(category_stats):
    """Print analysis grouped by interaction type."""
    print("\n" + "="*80)
    print("ANALYSIS BY INTERACTION TYPE")
    print("="*80)
    print()
    
    # Overall summary
    print(f"{'Category':<20} {'Tasks':>6} {'Base':>8} {'Steer':>8} {'Change':>8} {'Parse Δ':>10}")
    print("-" * 80)
    
    for category in ["multi_select", "dropdown", "semantic_type", "logic", "simple_click", "simple_type", "other"]:
        if category not in category_stats:
            continue
        
        stats = category_stats[category]
        if stats["total"] == 0:
            continue
        
        base_acc = stats["base_success"] / stats["total"] * 100
        steer_acc = stats["steer_success"] / stats["total"] * 100
        change = steer_acc - base_acc
        
        base_parse = stats["base_parse_fail"] / stats["total"] * 100
        steer_parse = stats["steer_parse_fail"] / stats["total"] * 100
        parse_delta = steer_parse - base_parse
        
        num_tasks = len(stats["tasks"])
        
        status = "✓" if change > 0 else ("=" if change == 0 else "✗")
        
        print(f"{category:<20} {num_tasks:>6} {base_acc:>7.1f}% {steer_acc:>7.1f}% "
              f"{change:>+7.1f}% {parse_delta:>+9.1f}% {status}")
    
    # Detailed per-task breakdown
    print()
    print("="*80)
    print("DETAILED TASK BREAKDOWN")
    print("="*80)
    
    for category in ["multi_select", "dropdown", "semantic_type", "logic"]:
        if category not in category_stats:
            continue
        
        stats = category_stats[category]
        if not stats["tasks"]:
            continue
        
        print()
        print(f"--- {category.upper().replace('_', ' ')} ---")
        print(f"{'Task':<25} {'Episodes':>9} {'Base':>8} {'Steer':>8} {'Change':>8}")
        print("-" * 80)
        
        for task, task_stats in sorted(stats["tasks"].items()):
            base_acc = task_stats["base"] / task_stats["total"] * 100
            steer_acc = task_stats["steer"] / task_stats["total"] * 100
            change = steer_acc - base_acc
            
            print(f"{task:<25} {task_stats['total']:>9} {base_acc:>7.1f}% {steer_acc:>7.1f}% {change:>+7.1f}%")


def analyze_overall(records):
    """Overall statistics."""
    total = len(records)
    base_success = sum(1 for r in records if r.get("base_success", False))
    steer_success = sum(1 for r in records if r.get("steer_success", False))
    base_parse_fail = sum(1 for r in records if r.get("base_action") is None)
    steer_parse_fail = sum(1 for r in records if r.get("steer_action") is None)
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total episodes: {total}")
    print(f"Base accuracy: {base_success}/{total} = {base_success/total*100:.1f}%")
    print(f"Steer accuracy: {steer_success}/{total} = {steer_success/total*100:.1f}%")
    print(f"Change: {(steer_success - base_success)/total*100:+.1f}%")
    print()
    print(f"Base parse failures: {base_parse_fail}/{total} = {base_parse_fail/total*100:.1f}%")
    print(f"Steer parse failures: {steer_parse_fail}/{total} = {steer_parse_fail/total*100:.1f}%")
    print(f"Parse failure change: {(steer_parse_fail - base_parse_fail)/total*100:+.1f}%")


def main():
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("❌ results/ directory not found")
        sys.exit(1)
    
    filepath = results_dir / "exp10_expanded.jsonl"
    
    if not filepath.exists():
        print(f"❌ {filepath} not found")
        print("Run: ./run_exp10_expanded.sh")
        sys.exit(1)
    
    print(f"Analyzing: {filepath}")
    records = load_results(filepath)
    
    if not records:
        print("❌ No valid records found")
        sys.exit(1)
    
    # Overall statistics
    analyze_overall(records)
    
    # Category-based analysis
    category_stats = analyze_by_category(records)
    print_category_analysis(category_stats)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
