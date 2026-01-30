import os
import json
import glob
from collections import defaultdict

RESULTS_DIR = "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep"

def get_accuracy(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = [l for l in f if l.strip()]
        if not lines: return -1
        hits = sum(1 for line in lines if json.loads(line).get('steer_success'))
        return hits / len(lines)
    except:
        return -1

def main():
    # Group files by model
    model_files = defaultdict(list)
    for f in glob.glob(os.path.join(RESULTS_DIR, "*_L*_a3.0.jsonl")):
        # Filename format: {model}_L{layer}_a3.0.jsonl
        # But model names have underscores/hyphens.
        # Strategy: split by _L, take left part.
        filename = os.path.basename(f)
        if "_L" not in filename: continue
        
        model_name = filename.split("_L")[0]
        model_files[model_name].append(f)

    print("Best Configurations (Alpha=3.0):")
    print(f"{ 'Model':<20} | {'Layer':<5} | {'Acc':<6} | {'File'}")
    print("-" * 60)

    best_configs = {}

    for model, files in sorted(model_files.items()):
        best_acc = -1.0
        best_file = None
        best_layer = -1

        for filepath in files:
            acc = get_accuracy(filepath)
            if acc > best_acc:
                best_acc = acc
                best_file = filepath
                # Extract layer
                try:
                    # filename: model_L12_a3.0.jsonl
                    base = os.path.basename(filepath)
                    layer_part = base.split("_L")[1] # 12_a3.0.jsonl
                    layer_num = int(layer_part.split("_")[0])
                    best_layer = layer_num
                except:
                    pass
        
        if best_file:
            print(f"{model:<20} | L{best_layer:<4} | {best_acc:.1%} | {best_file}")
            best_configs[model] = best_file

    # Also print a python dictionary for easy copy-paste
    print("\nTARGETS = {")
    for model, path in best_configs.items():
        print(f'    "{model}": "{path}",')
    print("}")

if __name__ == "__main__":
    main()
