import json
import os
from datasets import load_dataset
from tqdm import tqdm

def create_dataset():
    print("Loading Mind2Web dataset (streaming)...")
    dataset = load_dataset("osunlp/Mind2Web", split="train", streaming=True)
    
    output_data = []
    target_count = 50
    collected = 0
    
    print(f"Generating {target_count} contrastive pairs...")
    
    for sample in tqdm(dataset, total=target_count):
        if collected >= target_count:
            break
            
        if not sample['actions']:
            continue
            
        # Use the first action step for simplicity
        action_step = sample['actions'][0]
        
        # Extract basic info
        html_context = action_step['cleaned_html']
        user_goal = sample['confirmed_task']
        
        op_info = action_step['operation']
        op_type = op_info['op']
        op_value = op_info.get('value', '')
        
        # Get target element ID
        # (Using backend_node_id as the unique identifier)
        candidates = action_step['pos_candidates']
        if not candidates:
            continue
        target_id = candidates[0]['backend_node_id']
        
        # --- Template Logic ---
        positive_completion = ""
        negative_completion = ""
        
        if op_type == "CLICK":
            # JSON (Strict)
            positive_completion = json.dumps({
                "action": "click",
                "target": target_id
            })
            # Chat (Natural)
            negative_completion = f"I will click on the element with id {target_id} to proceed."
            
        elif op_type == "TYPE":
            positive_completion = json.dumps({
                "action": "type",
                "target": target_id,
                "value": op_value
            })
            negative_completion = f"I will type '{op_value}' into the element {target_id}."
            
        elif op_type == "SELECT":
            positive_completion = json.dumps({
                "action": "select",
                "target": target_id,
                "value": op_value
            })
            negative_completion = f"I will select the option '{op_value}' from the element {target_id}."
            
        else:
            # Skip unknown operations
            continue
            
        # Add to dataset
        output_data.append({
            "html": html_context,
            "goal": user_goal,
            "positive": positive_completion,
            "negative": negative_completion,
            "meta": {
                "op": op_type,
                "target": target_id,
                "value": op_value
            }
        })
        collected += 1

    # Save to file
    os.makedirs("data", exist_ok=True)
    output_path = "data/contrastive_dataset.json"
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Saved {len(output_data)} pairs to {output_path}")

if __name__ == "__main__":
    create_dataset()
