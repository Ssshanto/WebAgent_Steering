import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_robustness():
    results_dir = Path("results/exp8_robust_opt")
    files = sorted(results_dir.glob("*.jsonl"))
    
    # Store data: (layer, alpha) -> list of results
    data = defaultdict(list)
    
    print(f"Loaded {len(files)} result files.")
    
    for f in files:
        # Filename format: L{layer}_a{alpha}_s{seed}.jsonl
        try:
            parts = f.stem.split('_')
            layer = int(parts[0][1:])
            alpha = float(parts[1][1:])
            seed = int(parts[2][1:])
            
            recs = [json.loads(l) for l in open(f) if l.strip()]
            total = len(recs)
            base_succ = sum(1 for r in recs if r.get("base_success"))
            steer_succ = sum(1 for r in recs if r.get("steer_success"))
            
            base_acc = base_succ / total * 100
            steer_acc = steer_succ / total * 100
            delta = steer_acc - base_acc
            
            data[(layer, alpha)].append({
                "seed": seed,
                "base": base_acc,
                "steer": steer_acc,
                "delta": delta
            })
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    # Print Summary Table
    print("\n" + "="*80)
    print(f"{ 'Config':<15} | {'Mean Steer':<12} | {'Mean Delta':<12} | {'Std Dev':<10} | {'Min/Max Delta':<15}")
    print("-" * 80)
    
    best_config = None
    best_delta = -float('inf')
    
    sorted_keys = sorted(data.keys())
    
    for layer, alpha in sorted_keys:
        runs = data[(layer, alpha)]
        deltas = [r["delta"] for r in runs]
        steer_accs = [r["steer"] for r in runs]
        
        mean_steer = np.mean(steer_accs)
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        min_delta = np.min(deltas)
        max_delta = np.max(deltas)
        
        config_str = f"L{layer} α={alpha}"
        print(f"{config_str:<15} | {mean_steer:>10.2f}% | {mean_delta:>+10.2f}% | {std_delta:>8.2f}% | {min_delta:>+5.1f} / {max_delta:>+5.1f}")
        
        if mean_delta > best_delta:
            best_delta = mean_delta
            best_config = (layer, alpha)

    print("="*80)
    print(f"\n🏆 Best Configuration: Layer {best_config[0]}, Alpha {best_config[1]}")
    print(f"   Mean Improvement: {best_delta:+.2f}%")

if __name__ == "__main__":
    analyze_robustness()
