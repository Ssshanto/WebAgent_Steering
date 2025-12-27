import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import gc

STEERING_VECTOR = None
LAYER_IDX = 12
COEFF = 2.0

def steering_hook(module, input, output):
    if STEERING_VECTOR is not None:
        hidden_states = output[0]
        vector_tensor = STEERING_VECTOR.to(hidden_states.device).to(hidden_states.dtype)
        if hidden_states.dim() == 3:
            hidden_states[:, :, :] += COEFF * vector_tensor
        elif hidden_states.dim() == 2:
            hidden_states[:, :] += COEFF * vector_tensor
    return output

def is_valid_json(text):
    text = text.strip()
    # Attempt to find JSON structure if wrapped in text
    try:
        # Simple heuristic: find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            candidate = text[start:end+1]
            json.loads(candidate)
            return True
    except:
        pass
    return False

def evaluate():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "finetuned_model"
    vector_path = "vectors/steering_vectors.json"
    data_path = "data/contrastive_dataset.json"
    
    # Load Data
    with open(data_path, "r") as f:
        data = json.load(f)[:20] # Test on 20 examples
        
    results = []
    
    # --- PHASE 1: Base Model & Steering ---
    print("Loading Base Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    
    # Load Vectors
    with open(vector_path, "r") as f:
        vectors = json.load(f)
    global STEERING_VECTOR
    STEERING_VECTOR = torch.tensor(vectors[str(LAYER_IDX)])
    
    # Register Hook
    layer = model.model.layers[LAYER_IDX]
    hook_handle = layer.register_forward_hook(steering_hook)
    
    print("Evaluating Base & Steered...")
    
    for i, item in tqdm(enumerate(data), total=len(data)):
        prompt = f"You are a web automation agent. You must output the next action as a strict JSON object. Do not explain.\n\nHTML: {item['html'][:1000]}...\nGoal: {item['goal']}\nAction:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # 1. Steered (Hook Active)
        global COEFF
        COEFF = 2.0
        out_steered = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        text_steered = tokenizer.decode(out_steered[0], skip_special_tokens=True)[len(prompt):].strip()
        
        # 2. Base (Hook Inactive via Coeff 0)
        COEFF = 0.0
        out_base = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        text_base = tokenizer.decode(out_base[0], skip_special_tokens=True)[len(prompt):].strip()
        
        results.append({
            "id": i,
            "goal": item['goal'],
            "base_output": text_base,
            "steered_output": text_steered,
            "base_valid": is_valid_json(text_base),
            "steered_valid": is_valid_json(text_steered)
        })
        
    hook_handle.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # --- PHASE 2: Fine-Tuned Model ---
    print("Loading FT Model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("Evaluating FT...")
    for i, item in tqdm(enumerate(data), total=len(data)):
        prompt = f"<|im_start|>user\nYou are a web automation agent. You must output the next action as a strict JSON object. Do not explain.\n\nHTML: {item['html'][:1000]}...\nGoal: {item['goal']}\nAction:<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        out_ft = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        # Decode only the new tokens
        text_ft = tokenizer.decode(out_ft[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        results[i]["ft_output"] = text_ft
        results[i]["ft_valid"] = is_valid_json(text_ft)
        
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print("Results saved to results.csv")
    
    # Print Summary
    print("\nSummary:")
    print(f"Base Valid Rate: {df['base_valid'].mean():.2%}")
    print(f"Steered Valid Rate: {df['steered_valid'].mean():.2%}")
    print(f"FT Valid Rate: {df['ft_valid'].mean():.2%}")

if __name__ == "__main__":
    evaluate()
