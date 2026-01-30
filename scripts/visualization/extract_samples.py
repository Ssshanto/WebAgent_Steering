import json
import os
import glob

# Configuration: (Model Name, Path to BEST/Representative result file)
TARGETS = {
    "Qwen 0.5B (Success +14%)": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/0.5b_L6_a3.0.jsonl",
    "Qwen-Coder 0.5B (Success +7%)": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/qwen-coder-0.5b_L11_a3.0.jsonl",
    "Llama 1B (Format Fix)": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/llama-1b_L9_a3.0.jsonl",
    "StableLM 1.6B (Success +4%)": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/stablelm-1.6b_L12_a3.0.jsonl",
    "TinyLlama 1.1B (Format Fix)": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/tinyllama-1.1b_L10_a3.0.jsonl",
    "SmolLM 360M (Floor Fail)": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/smollm-360m_L19_a3.0.jsonl",
    "SmolLM 1.7B (Null)": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/smollm-1.7b_L22_a3.0.jsonl",
    "Gemma 3 1B (Rigid)": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/gemma-1b_L14_a3.0.jsonl",
    "Qwen 1.5B (Rigid)": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/qwen-1.5b_L21_a3.0.jsonl",
    "Qwen-VL 2B (Rigid)": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/qwen-vl-2b_L27_a3.0.jsonl"
}

CATEGORIES = {
    "FAIL_TO_CORRECT": lambda b_ok, b_act, s_ok, s_act: not b_ok and b_act is None and s_ok,
    "FAIL_TO_PARSE_WRONG": lambda b_ok, b_act, s_ok, s_act: not b_ok and b_act is None and not s_ok and s_act is not None,
    "CORRECT_TO_WRONG": lambda b_ok, b_act, s_ok, s_act: b_ok and not s_ok,
    "STILL_FAIL_FORMAT": lambda b_ok, b_act, s_ok, s_act: not b_ok and b_act is None and not s_ok and s_act is None
}

def get_snippet(text):
    if text is None: return "NULL"
    # Truncate long outputs for readability
    if len(text) > 300:
        return text[:150] + "\n...[truncated]...\n" + text[-100:]
    return text

def analyze_file(model_name, file_path):
    print(f"\n## Model: {model_name}")
    
    if not os.path.exists(file_path):
        print(f"(File not found: {file_path})")
        return

    # Store one example per category
    examples = {k: None for k in CATEGORIES.keys()}
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                row = json.loads(line)
                
                base_ok = row.get('base_success', False)
                steer_ok = row.get('steer_success', False)
                base_act = row.get('base_action')
                steer_act = row.get('steer_action')
                
                for cat, condition in CATEGORIES.items():
                    if examples[cat] is None and condition(base_ok, base_act, steer_ok, steer_act):
                        examples[cat] = row
            except:
                continue
            
            # Stop if we found all examples
            if all(v is not None for v in examples.values()):
                break
    
    # Print Markdown
    for cat, row in examples.items():
        if row:
            print(f"\n### Category: {cat.replace('_', ' ')}")
            print(f"**Task:** `{row['task']}` (Seed: {row['seed']})")
            print("\n**Baseline Output:**")
            print("```")
            print(get_snippet(row['base_output']))
            print("```")
            print(f"*Parsed Action:* `{row['base_action']}`")
            print(f"*Success:* {row['base_success']}")
            
            print("\n**Steered Output:**")
            print("```")
            print(get_snippet(row['steer_output']))
            print("```")
            print(f"*Parsed Action:* `{row['steer_action']}`")
            print(f"*Success:* {row['steer_success']}")
            print("\n---")

if __name__ == "__main__":
    print("# Findings: Qualitative Analysis of Steering Effects (All Models)\n")
    for name, path in TARGETS.items():
        analyze_file(name, path)