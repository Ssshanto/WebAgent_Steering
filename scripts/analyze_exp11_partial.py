import json
import glob
import os
import pandas as pd
from collections import defaultdict

RESULTS_DIR = "results/exp11_gold_sweep"

def analyze():
    data = []
    files = glob.glob(f"{RESULTS_DIR}/*.jsonl")
    print(f"Analyzing {len(files)} completed runs...\n")
    
    for f in files:
        # Filename format: {PROMPT}_L{LAYER}_a{ALPHA}.jsonl
        try:
            fname = os.path.basename(f).replace(".jsonl", "")
            # Split from right to handle underscores in prompt names
            parts = fname.split('_L')
            prompt = parts[0]
            rest = parts[1]
            layer = int(rest.split('_a')[0])
            alpha = float(rest.split('_a')[1])
            
            recs = [json.loads(l) for l in open(f) if l.strip()]
            if not recs: continue
            
            total = len(recs)
            base_succ = sum(1 for r in recs if r.get("base_success"))
            steer_succ = sum(1 for r in recs if r.get("steer_success"))
            
            base_acc = base_succ / total * 100
            steer_acc = steer_succ / total * 100
            delta = steer_acc - base_acc
            
            data.append({
                "Prompt": prompt,
                "Layer": layer,
                "Alpha": alpha,
                "Base": base_acc,
                "Steer": steer_acc,
                "Delta": delta
            })
        except Exception as e:
            print(f"Error parsing {f}: {e}")
            continue

    if not data:
        print("No valid data found.")
        return

    df = pd.DataFrame(data)
    
    # Sort by Layer, then Alpha, then Prompt
    df = df.sort_values(by=["Layer", "Alpha", "Delta"], ascending=[True, True, False])
    
    print("="*80)
    print(f"{ 'Prompt':<20} | {'L':<3} | {'α':<3} | {'Base':<6} | {'Steer':<6} | {'Delta':<6}")
    print("-" * 80)
    
    current_layer = -1
    current_alpha = -1
    
    for _, row in df.iterrows():
        # Print separators for readability
        if row["Layer"] != current_layer:
            print(f"{'-'*80}")
            current_layer = row["Layer"]
        
        print(f"{row['Prompt']:<20} | {row['Layer']:<3} | {row['Alpha']:<3.1f} | {row['Base']:>5.1f}% | {row['Steer']:>5.1f}% | {row['Delta']:>+5.1f}%")
        
    print("="*80)
    
    # Identify Top Performers
    print("\n🏆 TOP 5 CONFIGURATIONS SO FAR:")
    top5 = df.sort_values(by="Delta", ascending=False).head(5)
    for _, row in top5.iterrows():
        print(f"  {row['Prompt']} (L{row['Layer']}, α{row['Alpha']}): {row['Delta']:+.1f}%")

if __name__ == "__main__":
    analyze()
