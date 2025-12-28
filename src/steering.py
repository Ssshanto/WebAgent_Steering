import argparse
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
    def __init__(self, model_name=MODEL_NAME, layer_idx=LAYER_IDX, coeff=COEFF, apply_mode="all_tokens"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
            device_map="auto"
        )
        self.layer_idx = layer_idx
        self.coeff = coeff
        self.apply_mode = apply_mode
        self.vector = None
        self._vector_cache = {}

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
        # Keep on CPU; move to the active layer device during hooks (device_map may shard layers).
        self.vector = torch.tensor(pca.components_[0], device="cpu", dtype=torch.float32)
        self._vector_cache.clear()
        print("Vector extracted.")

    def _get_last_token_state(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # Get state from target layer, last token
        return out.hidden_states[self.layer_idx][0, -1, :].cpu().numpy()

    def generate(self, prompt, steer=False, max_new_tokens=40):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Define Hook
        def hook(module, input, output):
            if steer and self.vector is not None:
                def vec_for(tensor):
                    dev = tensor.device
                    vec = self._vector_cache.get(dev)
                    if vec is None:
                        vec = self.vector.to(device=dev, dtype=tensor.dtype)
                        self._vector_cache[dev] = vec
                    return vec

                if torch.is_tensor(output):
                    vec = vec_for(output)
                    if self.apply_mode == "last_token":
                        if output.dim() == 3:
                            output[:, -1, :] += self.coeff * vec
                        elif output.dim() == 2:
                            output[-1, :] += self.coeff * vec
                        else:
                            output += self.coeff * vec
                    else:
                        output += self.coeff * vec
                elif isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                    vec = vec_for(output[0])
                    if self.apply_mode == "last_token":
                        if output[0].dim() == 3:
                            output[0][:, -1, :] += self.coeff * vec
                        elif output[0].dim() == 2:
                            output[0][-1, :] += self.coeff * vec
                        else:
                            output[0] += self.coeff * vec
                    else:
                        output[0] += self.coeff * vec
            return output

        # Register Hook
        handle = self.model.model.layers[self.layer_idx].register_forward_hook(hook)
        
        # Generate
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--layer-idx", type=int, default=LAYER_IDX)
    parser.add_argument("--coeff", type=float, default=COEFF)
    parser.add_argument("--apply-mode", choices=["all_tokens", "last_token"], default="all_tokens")
    parser.add_argument("--extract-samples", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--eval-offset", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--out-csv", default="full_results.csv")
    args = parser.parse_args()
    
    # 1. Setup
    agent = SteeredModel(
        model_name=args.model_name,
        layer_idx=args.layer_idx,
        coeff=args.coeff,
        apply_mode=args.apply_mode,
    )
    
    # Fetch enough data for Extraction (50) + Evaluation (500)
    eval_offset = args.eval_offset if args.eval_offset is not None else args.extract_samples
    total_needed = max(args.extract_samples, eval_offset + args.eval_samples)
    print(f"Fetching {total_needed} samples from Mind2Web...")
    all_data = get_contrastive_pairs(total_needed)
    
    if not all_data:
        print("ERROR: No data found! Check dataset download.")
        return

    # Split
    train_pairs = all_data[:args.extract_samples]
    eval_pairs = all_data[eval_offset:eval_offset + args.eval_samples]
    
    print(
        f"Config: layer={args.layer_idx} coeff={args.coeff} mode={args.apply_mode} "
        f"extract={len(train_pairs)} eval={len(eval_pairs)} eval_offset={eval_offset} "
        f"max_new_tokens={args.max_new_tokens}"
    )
    
    # 2. Extract Vector (using train set)
    agent.extract_vector(train_pairs)
    
    # 3. Evaluate (using eval set)
    print(f"\n--- STARTING EVALUATION ON {len(eval_pairs)} SAMPLES ---")
    prompt_template = "You are a web agent. Output strictly JSON.\n\nHTML: {html}...\nGoal: {goal}\nAction:"
    
    results = []
    
    for i, item in tqdm(enumerate(eval_pairs), total=len(eval_pairs)):
        prompt = prompt_template.format(html=item['html'][:1000], goal=item['goal'])
        
        # Base Run
        base_out = agent.generate(prompt, steer=False, max_new_tokens=args.max_new_tokens)
        base_valid = is_valid_json(base_out)
        
        # Steered Run
        steered_out = agent.generate(prompt, steer=True, max_new_tokens=args.max_new_tokens)
        steered_valid = is_valid_json(steered_out)
        
        results.append({
            "id": eval_offset + i,
            "goal": item['goal'],
            "base_output": base_out,
            "base_valid": base_valid,
            "steered_output": steered_out,
            "steered_valid": steered_valid
        })
        
        # Periodic Save (every 50)
        if (i+1) % 50 == 0:
            pd.DataFrame(results).to_csv(args.out_csv, index=False)
            
    # Final Save
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    
    print("\n--- RESULTS ---")
    print(f"Base Valid Rate:    {df['base_valid'].mean():.2%}")
    print(f"Steered Valid Rate: {df['steered_valid'].mean():.2%}")
    print(f"Saved to {args.out_csv}")

if __name__ == "__main__":
    run()
