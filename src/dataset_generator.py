import json
import os
from datasets import load_dataset
from tqdm import tqdm

def create_dataset():
    print("Loading Mind2Web dataset (streaming)...")
    dataset = load_dataset("osunlp/Mind2Web", split="train", streaming=True)
    
    output_data = []
    target_count = 200  # Reduced for POC speed
    collected = 0
    
    print(f"Generating {target_count} examples...")
    
    for sample in tqdm(dataset, total=target_count):
        if collected >= target_count:
            break
            
        if not sample['actions']:
            continue
            
        # Use the first action step
        action_step = sample['actions'][0]
        
        html_context = action_step['cleaned_html']
        user_goal = sample['confirmed_task']
        
        op_info = action_step['operation']
        op_type = op_info.get('original_op', op_info.get('op')) # Handle variations
        op_value = op_info.get('value', '')
        
        candidates = action_step['pos_candidates']
        if not candidates:
            continue
        target_id = candidates[0]['backend_node_id']
        
        # --- Template Logic ---
        # Normalized Op Codes
        # 1: CLICK, 2: SELECT, 3: TYPE
        # Or strings "CLICK", "SELECT", "TYPE"
        
        normalized_op = ""
        if op_type == "CLICK" or op_type == 1:
            normalized_op = "CLICK"
            positive_json = {"action": "click", "target": target_id}
            negative_text = f"I will click on the element with id {target_id}."
        elif op_type == "TYPE" or op_type == 3:
            normalized_op = "TYPE"
            positive_json = {"action": "type", "target": target_id, "value": op_value}
            negative_text = f"I will type '{op_value}' into the element {target_id}."
        elif op_type == "SELECT" or op_type == 2:
            normalized_op = "SELECT"
            positive_json = {"action": "select", "target": target_id, "value": op_value}
            negative_text = f"I will select the option '{op_value}' from the element {target_id}."
        else:
            continue
            
        positive_str = json.dumps(positive_json)
        
        # Construct SFT Training Prompt
        # Format: User instruction -> Assistant JSON
        system_prompt = "You are a web automation agent. You must output the next action as a strict JSON object. Do not explain."
        user_input = f"HTML: {html_context[:2000]}...\nGoal: {user_goal}\nAction:"
        
        # We need a 'text' field for SFTTrainer
        # Using ChatML format or similar is good, but simple concatenation works for base SFT
        # Let's use Qwen's Chat Template format if possible, but raw text is safer for simple SFT
        # Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
        
        sft_text = f"<|im_start|>user\n{system_prompt}\n\n{user_input}<|im_end|>\n<|im_start|>assistant\n{positive_str}<|im_end|>"

        output_data.append({
            "html": html_context,
            "goal": user_goal,
            "positive": positive_str,
            "negative": negative_text,
            "text": sft_text, # For Fine-Tuning
            "meta": {
                "op": normalized_op,
                "target": target_id
            }
        })
        collected += 1

    # Save to file
    os.makedirs("data", exist_ok=True)
    output_path = "data/training_dataset.json"
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Saved {len(output_data)} examples to {output_path}")

if __name__ == "__main__":
    create_dataset()