import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from data import get_contrastive_pairs

# Config
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER_IDX = 12
COEFF = 2.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SteeredModel:
    def __init__(self):
        print(f"Loading {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
            device_map="auto"
        )
        self.vector = None

    def extract_vector(self, pairs):
        print("Extracting steering vector...")
        diffs = []
        
        for item in pairs:
            prompt = f"HTML: {item['html'][:1000]}...\nGoal: {item['goal']}\nAction:"
            
            # Get hidden states for Positive (JSON) and Negative (Chat)
            pos_state = self._get_last_token_state(prompt + item['positive'])
            neg_state = self._get_last_token_state(prompt + item['negative'])
            
            diffs.append(pos_state - neg_state)

        # PCA to find main direction
        pca = PCA(n_components=1)
        pca.fit(np.array(diffs))
        self.vector = torch.tensor(pca.components_[0], device=DEVICE, dtype=self.model.dtype)
        print("Vector extracted.")

    def _get_last_token_state(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # Get state from target layer, last token
        return out.hidden_states[LAYER_IDX][0, -1, :].cpu().numpy()

    def generate(self, prompt, steer=False):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Define Hook
        def hook(module, input, output):
            if steer and self.vector is not None:
                # Add vector to output of the layer
                # output[0] is (batch, seq, hidden)
                output[0][:, :, :] += COEFF * self.vector
            return output

        # Register Hook
        handle = self.model.model.layers[LAYER_IDX].register_forward_hook(hook)
        
        # Generate
        out = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        
        handle.remove()
        return text[len(prompt):].strip()

def is_valid_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json.loads(text[start:end+1])
            return True
    except:
        pass
    return False

def run():
    import pandas as pd
    from tqdm import tqdm
    
    # 1. Setup
    agent = SteeredModel()
    
    # Fetch enough data for Extraction (50) + Evaluation (500)
    total_needed = 550
    print(f"Fetching {total_needed} samples from Mind2Web...")
    all_data = get_contrastive_pairs(total_needed)
    
    if not all_data:
        print("ERROR: No data found! Check dataset download.")
        return

    # Split
    train_pairs = all_data[:50]
    eval_pairs = all_data[50:]
    
    print(f"Splitting data: {len(train_pairs)} for Extraction, {len(eval_pairs)} for Evaluation.")
    
    # 2. Extract Vector (using train set)
    agent.extract_vector(train_pairs)
    
    # 3. Evaluate (using eval set)
    print(f"\n--- STARTING EVALUATION ON {len(eval_pairs)} SAMPLES ---")
    prompt_template = "You are a web agent. Output strictly JSON.\n\nHTML: {html}...\nGoal: {goal}\nAction:"
    
    results = []
    
    for i, item in tqdm(enumerate(eval_pairs), total=len(eval_pairs)):
        prompt = prompt_template.format(html=item['html'][:1000], goal=item['goal'])
        
        # Base Run
        base_out = agent.generate(prompt, steer=False)
        base_valid = is_valid_json(base_out)
        
        # Steered Run
        steered_out = agent.generate(prompt, steer=True)
        steered_valid = is_valid_json(steered_out)
        
        results.append({
            "id": i,
            "goal": item['goal'],
            "base_output": base_out,
            "base_valid": base_valid,
            "steered_output": steered_out,
            "steered_valid": steered_valid
        })
        
        # Periodic Save (every 50)
        if (i+1) % 50 == 0:
            pd.DataFrame(results).to_csv("full_results.csv", index=False)
            
    # Final Save
    df = pd.DataFrame(results)
    df.to_csv("full_results.csv", index=False)
    
    print("\n--- RESULTS ---")
    print(f"Base Valid Rate:    {df['base_valid'].mean():.2%}")
    print(f"Steered Valid Rate: {df['steered_valid'].mean():.2%}")
    print("Saved to full_results.csv")

if __name__ == "__main__":
    run()
