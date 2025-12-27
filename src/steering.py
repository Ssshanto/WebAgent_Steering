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

def run():
    # 1. Setup
    agent = SteeredModel()
    pairs = get_contrastive_pairs(30)
    
    # 2. Extract
    agent.extract_vector(pairs)
    
    # 3. Evaluate
    print("\n--- EVALUATION (Sample) ---")
    prompt_template = "You are a web agent. Output strictly JSON.\n\nHTML: {html}...\nGoal: {goal}\nAction:"
    
    for i in range(5): # Test first 5
        item = pairs[i]
        prompt = prompt_template.format(html=item['html'][:500], goal=item['goal'])
        
        print(f"\n[Goal]: {item['goal']}")
        
        # Baseline
        base = agent.generate(prompt, steer=False)
        print(f"Base:    {base}")
        
        # Steered
        steered = agent.generate(prompt, steer=True)
        print(f"Steered: {steered}")

if __name__ == "__main__":
    run()
