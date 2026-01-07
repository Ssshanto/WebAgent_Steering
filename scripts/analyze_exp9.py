import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_exp9():
    results_dir = Path("results/exp9_prompt_strategies")
    files = sorted(results_dir.glob("*.jsonl"))
    
    # Data structure: alpha -> list of (prompt, metrics)
    data = defaultdict(list)
    
    print(f"Loaded {len(files)} result files.")
    
    for f in files:
        # Filename: {prompt}_a{alpha}.jsonl
        try:
            name_parts = f.stem.split('_a')
            if len(name_parts) != 2:
                continue
                
            prompt = name_parts[0]
            alpha = float(name_parts[1])
            
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
            
            data[alpha].append({
                "prompt": prompt,
                "base": base_acc,
                "steer": steer_acc,
                "delta": delta,
                "parse_delta": parse_delta,
                "total": total
            })
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    # Print Summary Tables for each Alpha
    for alpha in sorted(data.keys()):
        print("\n" + "="*80)
        print(f"RESULTS FOR ALPHA = {alpha}")
        print("="*80)
        print(f"{'Prompt':<25} | {'Base':<8} | {'Steer':<8} | {'Delta':<8} | {'Parse Δ':<8} | {'N':<5}")
        print("-