import json
import re
import os
import gymnasium as gym
import miniwob
from pathlib import Path
from tqdm import tqdm

# Ensure miniwob is registered
gym.register_envs(miniwob)

INPUT_FILE = "results/exp8_robust_opt/L13_a4.0_s0.jsonl"
OUTPUT_JSON = "results/exp8_L13_a4.0_s0_enriched.json"
HTML_DIR = "results/html"

os.makedirs(HTML_DIR, exist_ok=True)

def simple_oracle(env, task_name, obs):
    """
    Heuristic oracle to guess the ground truth action based on task logic.
    This is not perfect but covers many MiniWoB tasks.
    """
    utterance = obs["utterance"]
    dom = obs["dom_elements"]
    
    # 1. Click-Test / Button
    if task_name in ["click-test", "click-button"]:
        # usually matches text in utterance
        target = utterance.lower().replace("click", "").replace("the", "").replace("button", "").strip()
        if not target and task_name == "click-test": target = "click me" 
        
        for el in dom:
            if el["tag"] == "button" or (el["tag"] == "input" and el["classes"] == "button"):
                if target in (el["text"] or "").lower() or target in (el["value"] or "").lower():
                    return f'click ref={el["ref"]}'
    
    # 2. Click-Link
    if task_name == "click-link":
        target = utterance.replace("Click on the link", "").replace('"', "").replace(".", "").strip()
        for el in dom:
            if el["tag"] == "span" and "alink" in el["classes"]:
                if target.lower() in (el["text"] or "").lower():
                    return f'click ref={el["ref"]}'

    # 3. Focus-Text
    if "focus-text" in task_name:
        for el in dom:
            if el["tag"] == "input_text":
                return f'click ref={el["ref"]}' # Focus usually implemented as click

    return "Unknown"

def process():
    data = []
    
    # Read existing results to get task/seed pairs
    records = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Processing {len(records)} episodes...")
    
    for i, rec in enumerate(tqdm(records)):
        task = rec["task"]
        seed = rec["seed"]
        
        # Replay environment
        env = gym.make(f"miniwob/{task}-v1")
        obs, _ = env.reset(seed=seed)
        
        # 1. Save HTML
        # Construct simplified HTML from DOM for readability (or full HTML if available)
        # MiniWoB environment usually doesn't give raw HTML string easily in 'obs',
        # but we can reconstruct a view or dump the DOM elements.
        # Ideally we'd use driver.page_source but we don't have driver access here easily 
        # without hacking the env.
        # We will dump the DOM elements structure as HTML-like text.
        
        html_filename = f"episode_{i}_{task}.html"
        html_path = os.path.join(HTML_DIR, html_filename)
        
        # Reconstruct HTML representation
        html_content = f"<!-- Task: {obs['utterance']} -->\n<html><body>\n"
        for el in obs["dom_elements"]:
            tag = el["tag"]
            ref = el["ref"]
            txt = el["text"] or ""
            val = el["value"] or ""
            classes = el["classes"] or ""
            
            # Simple visualization
            attr_str = f'data-ref="{ref}"'
            if classes: attr_str += f' class="{classes}"'
            if val: attr_str += f' value="{val}"'
            
            html_content += f"  <{tag} {attr_str}>{txt}</{tag}>\n"
        html_content += "</body></html>"
        
        with open(html_path, "w") as f:
            f.write(html_content)
            
        # 2. Extract Ground Truth (Heuristic)
        ground_truth = simple_oracle(env, task, obs)
        
        # If model succeeded, overwrite heuristic with model's correct action
        if rec.get("steer_success"):
            ground_truth = rec.get("steer_output")
        elif rec.get("base_success"):
            ground_truth = rec.get("base_output")

        # 3. Create Enriched Record
        entry = {
            "index": i,
            "task": task,
            "seed": seed,
            "instruction": obs["utterance"],
            "html_file": html_filename,
            "ground_truth": ground_truth,
            "base_response": rec.get("base_output"),
            "steer_response": rec.get("steer_output"),
            "base_success": rec.get("base_success"),
            "steer_success": rec.get("steer_success"),
            "prompts": {
                "positive": "Be accurate and precise...",
                "negative": "Be inaccurate and imprecise..."
            }
        }
        data.append(entry)
        env.close()

    # Save Enriched JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"Saved enriched data to {OUTPUT_JSON}")

if __name__ == "__main__":
    process()
