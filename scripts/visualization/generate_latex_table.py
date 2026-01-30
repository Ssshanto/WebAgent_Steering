import json
import os
import glob
from pathlib import Path

TARGETS = {
    # Exp 13 (Final Sweep) - Alpha 3.0
    "smollm-360m": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/smollm-360m_L19_a3.0.jsonl",
    "0.5b": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/0.5b_L6_a3.0.jsonl",
    "qwen-coder-0.5b": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/qwen-coder-0.5b_L11_a3.0.jsonl",
    "llama-1b": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/llama-1b_L9_a3.0.jsonl",
    "gemma-1b": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/gemma-1b_L14_a3.0.jsonl",
    "tinyllama-1.1b": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/tinyllama-1.1b_L10_a3.0.jsonl",
    "opt-iml-1.3b": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/opt-iml-1.3b_L13_a3.0.jsonl",
    "qwen-1.5b": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/qwen-1.5b_L21_a3.0.jsonl",
    "stablelm-1.6b": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/stablelm-1.6b_L12_a3.0.jsonl",
    "smollm-1.7b": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/smollm-1.7b_L22_a3.0.jsonl",
    "qwen-vl-2b": "/mnt/code/Reaz/WebAgent_Steering/results/final_small_models_sweep/qwen-vl-2b_L27_a3.0.jsonl",
}

MODEL_META = {
    "smollm-360m": ("SmolLM", "0.36B"),
    "0.5b": ("Qwen 2.5", "0.5B"),
    "qwen-coder-0.5b": ("Qwen-Coder", "0.5B"),
    "llama-1b": ("Llama 3.2", "1.0B"),
    "gemma-1b": ("Gemma 3", "1.0B"),
    "tinyllama-1.1b": ("TinyLlama", "1.1B"),
    "opt-iml-1.3b": ("OPT-IML", "1.3B"),
    "stablelm-1.6b": ("StableLM 2", "1.6B"),
    "qwen-1.5b": ("Qwen 2.5", "1.5B"),
    "smollm-1.7b": ("SmolLM 2", "1.7B"),
    "qwen-vl-2b": ("Qwen2-VL", "2.0B"),
}

def analyze_file(model_key, file_path):
    if not os.path.exists(file_path):
        return None

    stats = {
        "base_ok": 0, "steer_ok": 0,
        "base_parse_ok": 0, "steer_parse_ok": 0,
        "fix": 0, "break": 0, "total": 0,
        "base_cond_ok": 0, "steer_cond_ok": 0
    }
    
    with open(file_path, 'r') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    
    stats["total"] = len(rows)
    if stats["total"] == 0: return None

    for r in rows:
        b_ok = r.get("base_success", False)
        s_ok = r.get("steer_success", False)
        b_act = r.get("base_action")
        s_act = r.get("steer_action")
        
        if b_ok: stats["base_ok"] += 1
        if s_ok: stats["steer_ok"] += 1
        if b_act is not None: stats["base_parse_ok"] += 1
        if s_act is not None: stats["steer_parse_ok"] += 1
        
        if not b_ok and s_ok: stats["fix"] += 1
        if b_ok and not s_ok: stats["break"] += 1
        
        if b_act is not None and b_ok: stats["base_cond_ok"] += 1
        if s_act is not None and s_ok: stats["steer_cond_ok"] += 1

    return stats

def main():
    print("| Model | Size | Base Acc | Steer Acc | Δ Acc | Base Parse | Steer Parse | Δ Parse | Cond Base | Cond Steer | Fix | Break |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    
    # Sort by size (manual order in TARGETS is roughly size, but let's stick to dictionary order or custom list)
    # I'll use the order in TARGETS which I sorted by size roughly.
    
    for key in TARGETS:
        path = TARGETS[key]
        meta = MODEL_META.get(key, (key, "?"))
        stats = analyze_file(key, path)
        
        if not stats:
            print(f"| {meta[0]} | {meta[1]} | N/A | ... |")
            continue
            
        t = stats["total"]
        base_acc = stats["base_ok"] / t
        steer_acc = stats["steer_ok"] / t
        delta_acc = steer_acc - base_acc
        
        base_parse = stats["base_parse_ok"] / t
        steer_parse = stats["steer_parse_ok"] / t
        delta_parse = steer_parse - base_parse
        
        # Conditioned Accuracy: Acc / Parse_Rate (avoid div/0)
        cond_base = stats["base_cond_ok"] / max(1, stats["base_parse_ok"])
        cond_steer = stats["steer_cond_ok"] / max(1, stats["steer_parse_ok"])
        
        fix_rate = stats["fix"] / t
        break_rate = stats["break"] / t
        
        # Formatting for LaTeX/MD
        row = f"| {meta[0]} | {meta[1]} | {base_acc:.1%} | {steer_acc:.1%} | **{delta_acc:+.1%}** | {base_parse:.1%} | {steer_parse:.1%} | {delta_parse:+.1%} | {cond_base:.1%} | {cond_steer:.1%} | {fix_rate:.1%} | {break_rate:.1%} |"
        print(row)

if __name__ == "__main__":
    main()
