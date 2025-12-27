from datasets import load_dataset
import collections
import json
from tqdm import tqdm

def analyze_full_actions():
    print("Loading Mind2Web dataset (train split)...")
    dataset = load_dataset("osunlp/Mind2Web", split="train", streaming=True)
    
    action_counts = collections.Counter()
    example_actions = {}
    
    # Analyze 500 examples
    target_count = 500
    print(f"Analyzing first {target_count} examples...")
    
    for i, sample in tqdm(enumerate(dataset), total=target_count):
        if i >= target_count:
            break
            
        if not sample['actions']:
            continue
            
        for action in sample['actions']:
            op_code = action['operation']['op']
            
            # Mapping based on previous findings + potential new ones
            op_map = {1: "CLICK", 2: "SELECT", 3: "TYPE"}
            op_name = op_map.get(op_code, f"UNKNOWN_{op_code}")
            
            action_counts[op_name] += 1
            
            if op_name not in example_actions:
                example_actions[op_name] = action

    print("\nAction Distribution:")
    for action, count in action_counts.items():
        print(f"  {action}: {count}")
        
    print("\nExample Data for each type:")
    for op_name, action in example_actions.items():
        print(f"\n[{op_name}]")
        print(f"  Operation: {action['operation']}")
        # print(f"  HTML Snippet: {action['cleaned_html'][:100]}...")

if __name__ == "__main__":
    analyze_full_actions()
