import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import os

def get_hidden_states(model, tokenizer, prompt, completion, device):
    """
    Feeds prompt + completion to the model and extracts hidden states 
    for the completion tokens.
    """
    full_text = prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    # We want to mask out the prompt so we only look at the completion's impact
    # But for steering, looking at the *last token* of the completion is a 
    # robust and simple metric for "global style".
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    # outputs.hidden_states is a tuple of (layer_0, layer_1, ... layer_N)
    # Each shape: (batch, seq_len, hidden_dim)
    
    # We take the last token's state from every layer
    # shape: (num_layers, hidden_dim)
    states = []
    for layer_state in outputs.hidden_states:
        # layer_state[0, -1, :] -> Last token of first batch element
        states.append(layer_state[0, -1, :].cpu().numpy())
        
    return np.array(states)

def extract_steering_vector():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    model.to(device)
    
    # Load Data
    with open("data/contrastive_dataset.json", "r") as f:
        data = json.load(f)
        
    print(f"Processing {len(data)} pairs...")
    
    # Store differences for each layer
    # structure: layer_idx -> list of difference vectors
    layer_diffs = {}
    
    for item in tqdm(data):
        # Construct Prompt (Simplified template)
        prompt = f"HTML: {item['html'][:1000]}...\nGoal: {item['goal']}\nAction:"
        
        # Get States
        pos_states = get_hidden_states(model, tokenizer, prompt, item['positive'], device)
        neg_states = get_hidden_states(model, tokenizer, prompt, item['negative'], device)
        
        # Compute Difference (Pos - Neg)
        diff = pos_states - neg_states # shape: (num_layers, hidden_dim)
        
        for layer_idx, layer_diff in enumerate(diff):
            if layer_idx not in layer_diffs:
                layer_diffs[layer_idx] = []
            layer_diffs[layer_idx].append(layer_diff)
            
    print("Computing PCA for each layer...")
    steering_vectors = {}
    
    for layer_idx, diffs in layer_diffs.items():
        X = np.array(diffs) # (num_samples, hidden_dim)
        
        # PCA to find the "Average Direction"
        pca = PCA(n_components=1)
        pca.fit(X)
        
        # The first component is our steering vector
        steering_vec = pca.components_[0]
        steering_vectors[layer_idx] = steering_vec.tolist()
        
    # Save
    os.makedirs("vectors", exist_ok=True)
    output_path = "vectors/steering_vectors.json"
    with open(output_path, "w") as f:
        json.dump(steering_vectors, f)
        
    print(f"Saved steering vectors for {len(steering_vectors)} layers to {output_path}")

if __name__ == "__main__":
    extract_steering_vector()
