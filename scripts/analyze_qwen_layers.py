import os
import json
import glob

def calculate_metrics(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        if not lines: return None
        total = len(lines)
        hits = sum(1 for line in lines if json.loads(line).get('steer_success'))
        base_hits = sum(1 for line in lines if json.loads(line).get('base_success'))
        
        base_parse = sum(1 for line in lines if json.loads(line).get('base_action') is not None)
        steer_parse = sum(1 for line in lines if json.loads(line).get('steer_action') is not None)
        
        return {
            "acc": hits / total,
            "base_acc": base_hits / total,
            "base_parse": base_parse / total,
            "steer_parse": steer_parse / total,
            "count": total
        }
    except:
        return None

def main():
    directory = "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep"
    model_prefix = "0.5b"
    files = glob.glob(os.path.join(directory, f"{model_prefix}_L*_a3.0.jsonl"))
    
    results = []
    for f in files:
        layer = int(os.path.basename(f).split('_L')[1].split('_')[0])
        metrics = calculate_metrics(f)
        if metrics:
            results.append((layer, metrics))
    
    results.sort()
    
    print(f"| Layer | Base Acc | Steer Acc | Delta | Base Parse | Steer Parse |")
    print(f"|---|---|---|---|---|---|")
    for layer, m in results:
        delta = m['acc'] - m['base_acc']
        print(f"| L{layer} | {m['base_acc']:.1%} | {m['acc']:.1%} | **{delta:+.1%}** | {m['base_parse']:.1%} | {m['steer_parse']:.1%} |")

if __name__ == "__main__":
    main()
