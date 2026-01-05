import os
import time
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2.5-3B-Instruct"
cache_dir = os.environ.get("HF_HOME")

print(f"Downloading {model_id} to {cache_dir}...")

max_retries = 50
for i in range(max_retries):
    try:
        snapshot_download(
            repo_id=model_id,
            resume_download=True,
            cache_dir=cache_dir,
            local_files_only=False
        )
        print("SUCCESS: Model downloaded completely.")
        break
    except Exception as e:
        print(f"Attempt {i+1}/{max_retries} failed: {e}")
        print("Retrying in 5 seconds...")
        time.sleep(5)
else:
    print("FAILURE: Max retries reached.")
    exit(1)
