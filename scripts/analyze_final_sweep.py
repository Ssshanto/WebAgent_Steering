import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_final_sweep():
    # Use the results from the latest 'run.sh' sweep
    results_dir = Path("results/prompt_sweep_final")
    files = sorted(results_dir.glob("*.jsonl"))
    
    print(f"Loaded {len(files)} result files from final sweep.")
    
    summaries = []
    for f in files:
        prompt = f.stem
        
        recs = []
        with open(f) as file:
            for line in file:
                if line.strip():
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if not recs:
            continue
            
        total = len(recs)
        base_succ = sum(1 for r in recs if r.get("base_success"))
        steer_succ = sum(1 for r in recs if r.get("steer_success"))
        base_parse = sum(1 for r in recs if r.get("base_action") is None)
        steer_parse = sum(1 for r in recs if r.get("steer_action") is None)
        
        base_acc = base_succ / total * 100
        steer_acc = steer_succ / total * 100
        delta = steer_acc - base_acc
        parse_delta = (steer_parse - base_parse) / total * 100
        
        summaries.append({
            "prompt": prompt,
            "base": base_acc,
            "steer": steer_acc,
            "delta": delta,
            "parse_delta": parse_delta,
            "total": total
        })

    # Print Summary Table
    print("\n" + "="*85)
    print("FINAL PROMPT STRATEGY SWEEP (L13, ALPHA 4.0, 25 TASKS)")
    print("="*85)
    print(f"{ 'Prompt Strategy':<25} | { 'Base Acc':<10} | { 'Steer Acc':<10} | { 'Delta':<8} | { 'Parse Δ':<8}")
    print("-" * 85)
    
    # Sort by Delta descending
    runs = sorted(summaries, key=lambda x: x["delta"], reverse=True)
    
    for r in runs:
        print(f"{r['prompt']:<25} | {r['base']:>8.1f}% | {r['steer']:>8.1f}% | {r['delta']:>+7.1f}% | {r['parse_delta']:>+7.1f}%")
    print("="*85)

if __name__ == "__main__":
    analyze_final_sweep()
