import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STEERING_VECTOR = None
CURRENT_LAYER_HOOK = None
COEFF = 0.0

def steering_hook(module, input, output):
    global STEERING_VECTOR, COEFF
    if STEERING_VECTOR is not None:
        hidden_states = output[0]
        vector_tensor = STEERING_VECTOR.to(hidden_states.device).to(hidden_states.dtype)
        if hidden_states.dim() == 3:
            hidden_states[:, :, :] += COEFF * vector_tensor
        elif hidden_states.dim() == 2:
            hidden_states[:, :] += COEFF * vector_tensor
    return output

def run_sweep():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    model.to(device)
    
    with open("vectors/steering_vectors.json", "r") as f:
        vectors = json.load(f)

    # Test Prompt (Instruction-based)
    prompt = """You are a web automation agent. You must output the next action as a strict JSON object. Do not explain.

HTML: <div id='nav'><button id='login'>Log In</button></div>
Goal: Log in to the site.
Action:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    layers_to_test = [12, 16, 20]
    coeffs = [-2.0, 0.0, 2.0] # Can we break it? Can we fix it?

    for layer_idx in layers_to_test:
        print(f"\n\n=== TESTING LAYER {layer_idx} ===")
        
        # Load Vector
        global STEERING_VECTOR
        STEERING_VECTOR = torch.tensor(vectors[str(layer_idx)])
        
        # Register Hook
        layer = model.model.layers[layer_idx]
        handle = layer.register_forward_hook(steering_hook)
        
        for coeff in coeffs:
            global COEFF
            COEFF = coeff
            
            # Generate
            torch.manual_seed(42)
            # Reduce max tokens to see if it stops cleanly
            output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Clean up output
            gen_text = text[len(prompt):].strip().replace("\n", " ")
            print(f"Coeff {coeff}: {gen_text}")
            
        handle.remove()

if __name__ == "__main__":
    run_sweep()
