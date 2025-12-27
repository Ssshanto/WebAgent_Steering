import json
import os

def prepare_data():
    input_path = "data/contrastive_dataset.json"
    output_path = "data/training_dataset.json"
    
    if not os.path.exists(input_path):
        print("Error: Input dataset not found.")
        return

    with open(input_path, "r") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} original examples.")
    
    training_data = []
    
    # We repeat the data to simulate a larger epoch
    repeats = 4 
    
    for _ in range(repeats):
        for item in data:
            # Construct SFT Training Prompt
            system_prompt = "You are a web automation agent. You must output the next action as a strict JSON object. Do not explain."
            user_input = f"HTML: {item['html'][:2000]}...\nGoal: {item['goal']}\nAction:"
            
            # The 'positive' field in the existing json is already the JSON string
            assistant_response = item['positive'] 
            
            # Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>")
            sft_text = f"<|im_start|>user\n{system_prompt}\n\n{user_input}<|im_end|>\n<|im_start|>assistant\n{assistant_response}<|im_end|>"
            
            new_item = item.copy()
            new_item['text'] = sft_text
            training_data.append(new_item)
            
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)
        
    print(f"Saved {len(training_data)} training examples to {output_path}")

if __name__ == "__main__":
    prepare_data()