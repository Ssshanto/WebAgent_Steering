import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variable to store the steering vector
STEERING_VECTOR = None
TARGET_LAYER = 12
COEFF = 2.0

def steering_hook(module, input, output):
    """
    Adds the steering vector to the module's output.
    Output in transformers is usually a tuple (hidden_state, ...).
    We modify the first element.
    """
    global STEERING_VECTOR, COEFF
    
    if STEERING_VECTOR is not None:
        # output[0] shape: (batch_size, seq_len, hidden_dim)
        # STEERING_VECTOR shape: (hidden_dim,)
        
        # We add it to every token position (simplest approach)
        # Or just the last token? For generation, we usually add to all 
        # because the model attends to past tokens.
        
        # Ensure shapes match for broadcasting
        vector_tensor = STEERING_VECTOR.to(output[0].device).to(output[0].dtype)
        
        # Check shape to handle both prefill (3D) and decoding (2D or 3D)
        hidden_states = output[0]
        # print(f"DEBUG: Hidden state shape: {hidden_states.shape}") 
        
        if hidden_states.dim() == 3:
            # (batch, seq, hidden)
            hidden_states[:, :, :] += COEFF * vector_tensor
        elif hidden_states.dim() == 2:
            # (batch, hidden) - usually during token-by-token generation
            hidden_states[:, :] += COEFF * vector_tensor
            
    return output

def test_steering():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    model.to(device)
    
    # Load Vectors
    with open("vectors/steering_vectors.json", "r") as f:
        vectors = json.load(f)
        
    # Prepare the vector for Layer 12
    global STEERING_VECTOR
    STEERING_VECTOR = torch.tensor(vectors[str(TARGET_LAYER)])
    
    # Register Hook
    # We find layer 12 in the model structure
    # Qwen structure: model.model.layers[i]
    layer_to_hook = model.model.layers[TARGET_LAYER]
    handle = layer_to_hook.register_forward_hook(steering_hook)
    
    # Test Input (Using a synthetic one to see clear effect)
    # A prompt that usually triggers chat
    prompt = "HTML: <button id='buy'>Buy Now</button>\nGoal: Buy the item.\nAction:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("\n--- BASELINE (Coeff = 0.0) ---")
    global COEFF
    COEFF = 0.0
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    
    print(f"\n--- STEERED (Coeff = +1.5) Layer {TARGET_LAYER} ---")
    COEFF = 1.5
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    
    print(f"\n--- NEGATIVE STEERING (Coeff = -1.5) ---")
    # Should be EXTRA chatty
    COEFF = -1.5
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

    handle.remove()

if __name__ == "__main__":
    test_steering()
