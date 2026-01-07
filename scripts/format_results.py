import json

# Configuration
INPUT_FILE = "results/exp8_robust_opt/L13_a4.0_s0.jsonl"
OUTPUT_FILE = "results/exp8_L13_a4.0_s0_analyzable.json"

# Constants from the experiment
POS_PROMPT = "Be accurate and precise. Read the given information carefully. Ensure your answer is exactly correct before responding."
NEG_PROMPT = "Be inaccurate and imprecise. Skim the given information quickly. Answer without ensuring correctness."

def process():
    data = []
    try:
        with open(INPUT_FILE, "r") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                
                # Infer 'ground truth' if possible
                ground_truth = "Unknown"
                if rec.get("steer_success"):
                    ground_truth = rec.get("steer_output")
                elif rec.get("base_success"):
                    ground_truth = rec.get("base_output")
                
                # Create clean entry
                entry = {
                    "task": rec.get("task"),
                    "seed": rec.get("seed"),
                    "input_prompt": rec.get("prompt"),
                    "steering_prompts": {
                        "positive": POS_PROMPT,
                        "negative": NEG_PROMPT
                    },
                    "ground_truth_inferred": ground_truth,
                    "unsteered_response": rec.get("base_output"),
                    "steered_response": rec.get("steer_output"),
                    "outcome": {
                        "base_success": rec.get("base_success"),
                        "steer_success": rec.get("steer_success"),
                        "improvement": rec.get("steer_success") and not rec.get("base_success")
                    }
                }
                data.append(entry)
        
        # Write format
        with open(OUTPUT_FILE, "w") as f:
            json.dump(data, f, indent=4)
            
        print(f"Successfully processed {len(data)} records to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    process()
