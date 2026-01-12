import os
import json
import glob

def calculate_accuracy(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return 0.0, 0
        
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
                
        return (base_hits / total, steer_hits / total, total)
    except Exception as e:
        return None

def main():
    base_dir = "/mnt/code/Reaz/WebAgent_Steering/results"
    
    # Focus on exp12 and exp12_grid where we have clear model names
    target_dirs = [
        os.path.join(base_dir, "exp12"),
        os.path.join(base_dir, "exp12_grid"),
        os.path.join(base_dir, "llama1b_accuracy_sweep")
    ]
    
    results = {}

    print(f"{'Model':<20} | {'Experiment':<25} | {'Base Acc':<10} | {'Steer Acc':<10} | {'Episodes':<8}")
    print("-" * 85)

    for directory in target_dirs:
        if not os.path.exists(directory):
            continue
            
        # Group by model
        files = glob.glob(os.path.join(directory, "*.jsonl"))
        
        for file_path in sorted(files):
            filename = os.path.basename(file_path)
            
            # Simple parsing of filename
            if "_baseline" in filename:
                model = filename.split("_baseline")[0]
                exp_type = "Baseline"
            elif "_steered" in filename:
                model = filename.split("_steered")[0]
                exp_type = "Steered (Default)"
            elif "L" in filename and "a" in filename: # Likely sweep file
                parts = filename.split("_")
                model = parts[0]
                # Try to find Layer/Alpha
                exp_type = "_".join(parts[1:]).replace(".jsonl", "")
            else:
                continue

            stats = calculate_accuracy(file_path)
            if stats:
                base_acc, steer_acc, total = stats
                print(f"{model:<20} | {exp_type:<25} | {base_acc:.1%}     | {steer_acc:.1%}     | {total:<8}")

if __name__ == "__main__":
    main()
