import os
from transformers import AutoConfig, AutoProcessor

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct"
]

print("Verifying model artifacts in cache...\n")

for model_id in MODELS:
    try:
        print(f"Checking {model_id}...", end=" ", flush=True)
        # Try loading config first (fast)
        AutoConfig.from_pretrained(model_id, local_files_only=True)
        
        # For VLM, check processor too
        if "VL" in model_id:
            try:
                AutoProcessor.from_pretrained(model_id, local_files_only=True)
            except Exception:
                print("MISSING (Processor)")
                continue

        # Check for model weights existence (heuristic)
        # A full load takes too long, so we check if the snapshot dir exists and has safetensors
        cache_dir = os.environ.get("HF_HOME")
        repo_name = "models--" + model_id.replace("/", "--")
        snapshot_dir = os.path.join(cache_dir, "hub", repo_name, "snapshots")
        
        if not os.path.exists(snapshot_dir):
             print("MISSING (No snapshot dir)")
             continue
             
        # Find latest snapshot
        snapshots = os.listdir(snapshot_dir)
        if not snapshots:
            print("MISSING (Empty snapshot dir)")
            continue
            
        latest = snapshots[0] # Simplification
        model_path = os.path.join(snapshot_dir, latest)
        
        has_weights = any(f.endswith(".safetensors") or f.endswith(".bin") for f in os.listdir(model_path))
        
        if has_weights:
            print("OK")
        else:
            print("MISSING (No weights found)")
            
    except Exception as e:
        print(f"MISSING ({str(e)})")
