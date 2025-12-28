import json
from datasets import load_dataset
from tqdm import tqdm

def get_contrastive_pairs(num_samples=50):
    """
    Downloads Mind2Web (streaming) and constructs Positive (JSON) vs Negative (Chat) pairs.
    """
    print(f"Loading {num_samples} examples from Mind2Web...")
    # streaming=False to download full dataset (more robust)
    dataset = load_dataset("osunlp/Mind2Web", split="train", streaming=False, cache_dir="dataset_cache")
    
    pairs = []
    
    for sample in tqdm(dataset):
        if len(pairs) >= num_samples:
            break
            
        if not sample['actions']: continue
        
        # Extract context
        action = sample['actions'][0]
        html = action['cleaned_html']
        goal = sample['confirmed_task']
        op = action['operation']['op'] # 1=CLICK, 2=SELECT, 3=TYPE
        
        # Get target ID
        if not action['pos_candidates']: continue
        target_id = action['pos_candidates'][0]['backend_node_id']
        
        # Create Completions
        pos_json, neg_chat = "", ""
        
        if op == 1: # CLICK
            pos_json = json.dumps({"action": "click", "target": target_id})
            neg_chat = f"I will click on the element {target_id}."
        elif op == 2: # SELECT
            val = action['operation'].get('value', '')
            pos_json = json.dumps({"action": "select", "target": target_id, "value": val})
            neg_chat = f"I will select '{val}' from {target_id}."
        elif op == 3: # TYPE
            val = action['operation'].get('value', '')
            pos_json = json.dumps({"action": "type", "target": target_id, "value": val})
            neg_chat = f"I will type '{val}' into {target_id}."
        else:
            continue
            
        pairs.append({
            "html": html,
            "goal": goal,
            "positive": pos_json,
            "negative": neg_chat
        })
        
    return pairs
