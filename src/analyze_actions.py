from datasets import load_dataset
import collections
import json

def analyze_mind2web_actions():
    print("Loading Mind2Web dataset (train split)...")
    # Using a small slice just to get the schema and action types quickly
    dataset = load_dataset("osunlp/Mind2Web", split="train", streaming=True)
    
    action_counts = collections.Counter()
    example_actions = {}

    print("Analyzing first 100 examples...")
    for i, sample in enumerate(dataset):
        if i >= 100:
            break
            
        # Mind2Web structure is a bit nested.
        # usually sample['action_reprs'] contains the action string like:
        # "CLICK", "TYPE ...", "SELECT ..."
        
        # In the raw dataset, the operation is often inside 'operation' dict or parsed from 'action_reprs'
        # Let's inspect the 'operation' field which is most reliable.
        # structure: sample['operation']['op'] -> 1 (click), 2 (select), 3 (type)
        
        op_code = sample['operation']['op']
        
        # Mapping op codes to readable names (based on Mind2Web paper/docs)
        # 1: CLICK, 2: SELECT, 3: TYPE
        op_map = {1: "CLICK", 2: "SELECT", 3: "TYPE"}
        op_name = op_map.get(op_code, f"UNKNOWN_{op_code}")
        
        action_counts[op_name] += 1
        
        # Keep one example for each type to verify structure
        if op_name not in example_actions:
            example_actions[op_name] = sample

    print("\nAction Distribution (in first 500 samples):")
    for action, count in action_counts.items():
        print(f"  {action}: {count}")
        
    print("\nProposed Templates:")
    
    # CLICK
    if "CLICK" in example_actions:
        print("\n[CLICK]")
        print("  JSON: {\"action\": \"click\", \"target\": \"[ID]\"}")
        print("  Chat: \"I will click on the element [ID].\"")
        
    # TYPE
    if "TYPE" in example_actions:
        print("\n[TYPE]")
        # Type operations usually have a 'value'
        print("  JSON: {\"action\": \"type\", \"target\": \"[ID]\", \"value\": \"[TEXT]\"}")
        print("  Chat: \"I will type '[TEXT]' into the element [ID].\"")
        
    # SELECT
    if "SELECT" in example_actions:
        print("\n[SELECT]")
        # Select operations usually have a 'value' (the option)
        print("  JSON: {\"action\": \"select\", \"target\": \"[ID]\", \"value\": \"[OPTION]\"}")
        print("  Chat: \"I will select the option '[OPTION]' from the element [ID].\"")

if __name__ == "__main__":
    analyze_mind2web_actions()
