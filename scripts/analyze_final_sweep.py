import os
import json
import glob

def calculate_metrics(file_path):

    try:

        with open(file_path, 'r') as f:

            lines = f.readlines()

        if not lines: return 0.0, 0.0, 0.0, 0

        total = len(lines)

        hits = sum(1 for line in lines if json.loads(line).get('steer_success'))

        

        # Calculate parse failures

        base_parse_fail = sum(1 for line in lines if json.loads(line).get('base_action') is None)

        steer_parse_fail = sum(1 for line in lines if json.loads(line).get('steer_action') is None)

        

        return (hits / total, base_parse_fail / total, steer_parse_fail / total, total)

    except:

        return 0.0, 0.0, 0.0, 0



def analyze_sweep(directory, model_prefix):

    results = []

    files = glob.glob(os.path.join(directory, f"{model_prefix}_L*_a2.0.jsonl"))

    for f in files:

        # Extract layer number

        try:

            layer = int(os.path.basename(f).split('_L')[1].split('_')[0])

            acc, base_pf, steer_pf, count = calculate_metrics(f)

            if count > 0:

                results.append((layer, acc, base_pf, steer_pf))

        except: continue

    return sorted(results)



def main():

    base_dir = "/mnt/code/Reaz/WebAgent_Steering/results"

    

    gemma_results = analyze_sweep(os.path.join(base_dir, "gemma3_full_sweep"), "gemma-1b")

    qwen_results = analyze_sweep(os.path.join(base_dir, "qwen1.5b_full_sweep"), "qwen-1.5b")

    

    print("Gemma 3 1B Full Sweep (Alpha 2.0)")

    print(f"{'Layer':<6} | {'Accuracy':<10} | {'Base PF':<10} | {'Steer PF':<10}")

    print("-" * 45)

    for layer, acc, bpf, spf in gemma_results:

        print(f"{layer:<6} | {acc:.1%}     | {bpf:.1%}     | {spf:.1%}")

        

    print("\nQwen 1.5B Full Sweep (Alpha 2.0)")

    print(f"{'Layer':<6} | {'Accuracy':<10} | {'Base PF':<10} | {'Steer PF':<10}")

    print("-" * 45)

    for layer, acc, bpf, spf in qwen_results:

        print(f"{layer:<6} | {acc:.1%}     | {bpf:.1%}     | {spf:.1%}")

if __name__ == "__main__":
    main()