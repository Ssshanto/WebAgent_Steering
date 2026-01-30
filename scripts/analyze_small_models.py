import os
import json
import glob

def calculate_accuracy(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return 0.0, 0.0, 0
        
        total = len(lines)
        base_hits = 0
        steer_hits = 0
        
        for line in lines:
            try:
                data = json.loads(line)
                if data.get('base_success'):
                    base_hits += 1
                if data.get('steer_success'):
                    steer_hits += 1
            except json.JSONDecodeError:
                continue
                
        return (float(base_hits) / total, float(steer_hits) / total, int(total))
    except Exception as e:
        return 0.0, 0.0, 0

def main():
    base_dir = "/mnt/code/Reaz/WebAgent_Steering/results"
    
    # Target dirs including new sweep
    target_dirs = [
        os.path.join(base_dir, "small_models_sweep"),
        # Include reference data if needed, but focus on the new sweep for now
    ]
    
    print(f"{'Model':<20} | {'Experiment':<25} | {'Base Acc':<10} | {'Steer Acc':<10} | {'Episodes':<8}")
    print("-" * 85)

    for directory in target_dirs:
        if not os.path.exists(directory):
            continue
            
        files = glob.glob(os.path.join(directory, "*.jsonl"))
        
        for file_path in sorted(files):
            filename = os.path.basename(file_path)
            
            if "_baseline" in filename:
                model = filename.split("_baseline")[0]
                exp_type = "Baseline"
            elif "L" in filename and "a" in filename:
                parts = filename.split("_")
                model = parts[0]
                # Extract L and a parts
                exp_parts = [p for p in parts[1:] if p.startswith('L') or p.startswith('a')]
                exp_type = "_".join(exp_parts).replace(".jsonl", "")
            else:
                continue

            stats = calculate_accuracy(file_path)
            base_acc, steer_acc, total = stats
            print(f"{model:<20} | {exp_type:<25} | {base_acc:.1%}     | {steer_acc:.1%}     | {total:<8}")

if __name__ == "__main__":
    main()
