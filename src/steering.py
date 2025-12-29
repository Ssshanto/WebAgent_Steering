import argparse
import re
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import get_mind2web_tasks, iter_action_steps

# Config
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER_IDX = 12
COEFF = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are a helpful assistant that is great at website design, navigation, "
    "and executing tasks for the user"
)

OUTPUT_FORMAT = (
    "Please respond in the following format:\n"
    "Answer: <LETTER>.\n"
    "Action: <CLICK|SELECT|TYPE>\n"
    "Value: <text>"
)

class SteeredModel:
    def __init__(self, model_name=MODEL_NAME, layer_idx=LAYER_IDX, coeff=COEFF, apply_mode="last_token"):
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

    def _get_last_token_state(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # Get state from target layer, last token
        return out.hidden_states[self.layer_idx][0, -1, :].cpu().numpy()

    def set_vector_from_tta(self, prompt, raw_output, formatted_output):
        neg_state = self._get_last_token_state(prompt + raw_output)
        pos_state = self._get_last_token_state(prompt + formatted_output)
        vec = pos_state - neg_state
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        # Keep on CPU; move to the active layer device during hooks (device_map may shard layers).
        self.vector = torch.tensor(vec, device="cpu", dtype=torch.float32)
        self._vector_cache.clear()

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


ANSWER_RE = re.compile(r"Answer\s*:\s*([A-Za-z])", re.IGNORECASE)
ACTION_RE = re.compile(r"Action\s*:\s*([A-Za-z]+)", re.IGNORECASE)
VALUE_RE = re.compile(r"Value\s*:\s*(.*)", re.IGNORECASE)


def _normalize_action(action):
    if not action:
        return None
    action = action.strip().upper()
    if action.startswith("CLICK"):
        return "CLICK"
    if action.startswith("SELECT"):
        return "SELECT"
    if action.startswith("TYPE"):
        return "TYPE"
    return None


def _normalize_value(value):
    if value is None:
        return ""
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1].strip()
    return value


def parse_mind2web_output(text):
    answer = None
    action = None
    value = None
    match = ANSWER_RE.search(text or "")
    if match:
        answer = match.group(1).strip().upper()
    match = ACTION_RE.search(text or "")
    if match:
        action = _normalize_action(match.group(1))
    match = VALUE_RE.search(text or "")
    if match:
        value = match.group(1).splitlines()[0].strip()
    return {"answer": answer, "action": action, "value": value}


def format_mind2web_output(parsed):
    answer = parsed.get("answer") or "A"
    action = parsed.get("action") or "CLICK"
    value = parsed.get("value") or ""
    return f"Answer: {answer}.\nAction: {action}\nValue: {value}"


def build_prompt(step):
    html_block = f"```html\n{step['html']}\n```"
    prev_actions = step["previous_actions"]
    prev_text = "None" if not prev_actions else "\n".join(prev_actions)
    options_text = "\n".join(
        f"{label}. {text}" for label, _, text in step["options"]
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{html_block}\n"
        "Based on the HTML webpage above, try to complete the following task:\n"
        f"Task: {step['goal']}\n"
        "Previous actions:\n"
        f"{prev_text}\n"
        "What should be the next action? Please select from the following choices "
        "(If the correct action is not in the page above, please select A. 'None of the above'):\n"
        f"{options_text}\n\n"
        f"{OUTPUT_FORMAT}\n"
    )


def step_success(parsed, step, option_map):
    pred_action = parsed.get("action")
    pred_answer = parsed.get("answer")
    pred_value = _normalize_value(parsed.get("value"))

    gold_action = step["gold_op"]
    gold_value = _normalize_value(step["gold_value"])
    gold_targets = step["gold_targets"]

    target_id = option_map.get(pred_answer)
    target_ok = target_id is not None and target_id in gold_targets
    action_ok = pred_action == gold_action

    value_ok = True
    if gold_action in ("SELECT", "TYPE") and gold_value:
        value_ok = pred_value.lower() == gold_value.lower()

    return target_ok and action_ok and value_ok

def run():
    import pandas as pd
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--layer-idx", type=int, default=LAYER_IDX)
    parser.add_argument("--coeff", type=float, default=COEFF)
    parser.add_argument("--apply-mode", choices=["all_tokens", "last_token"], default="last_token")
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--max-neg-candidates", type=int, default=7)
    parser.add_argument("--max-html-chars", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--out-csv", default="mind2web_results.csv")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--max-actions", type=int, default=None)
    args = parser.parse_args()
    
    # 1. Setup
    agent = SteeredModel(
        model_name=args.model_name,
        layer_idx=args.layer_idx,
        coeff=args.coeff,
        apply_mode=args.apply_mode,
    )
    
    tasks = get_mind2web_tasks(
        num_tasks=args.num_tasks,
        task_offset=args.task_offset,
    )
    if not tasks:
        print("ERROR: No tasks found! Check dataset download.")
        return

    print(
        f"Config: layer={args.layer_idx} coeff={args.coeff} mode={args.apply_mode} "
        f"tasks={len(tasks)} max_new_tokens={args.max_new_tokens} "
        f"max_neg_candidates={args.max_neg_candidates} tta={'off' if args.no_tta else 'on'}"
    )

    results = []
    action_count = 0
    total_actions = sum(len(t.get("actions") or []) for t in tasks)
    print(f"\n--- STARTING EVALUATION ON {total_actions} ACTIONS ---")

    for task in tqdm(tasks, total=len(tasks)):
        for step in iter_action_steps(
            task,
            max_neg_candidates=args.max_neg_candidates,
            max_html_chars=args.max_html_chars,
        ):
            prompt = build_prompt(step)
            option_map = {label: backend_id for label, backend_id, _ in step["options"]}

            base_out = agent.generate(prompt, steer=False, max_new_tokens=args.max_new_tokens)
            base_parsed = parse_mind2web_output(base_out)
            base_success = step_success(base_parsed, step, option_map)

            if args.no_tta:
                steered_out = agent.generate(prompt, steer=False, max_new_tokens=args.max_new_tokens)
            else:
                formatted = format_mind2web_output(base_parsed)
                agent.set_vector_from_tta(prompt, base_out, formatted)
                steered_out = agent.generate(prompt, steer=True, max_new_tokens=args.max_new_tokens)

            steered_parsed = parse_mind2web_output(steered_out)
            steered_success = step_success(steered_parsed, step, option_map)

            results.append(
                {
                    "annotation_id": step["annotation_id"],
                    "action_uid": step["action_uid"],
                    "goal": step["goal"],
                    "gold_action": step["gold_op"],
                    "gold_value": step["gold_value"],
                    "base_output": base_out,
                    "base_answer": base_parsed.get("answer"),
                    "base_action": base_parsed.get("action"),
                    "base_value": base_parsed.get("value"),
                    "base_step_success": base_success,
                    "steered_output": steered_out,
                    "steered_answer": steered_parsed.get("answer"),
                    "steered_action": steered_parsed.get("action"),
                    "steered_value": steered_parsed.get("value"),
                    "steered_step_success": steered_success,
                }
            )

            action_count += 1
            if args.max_actions and action_count >= args.max_actions:
                break
        if args.max_actions and action_count >= args.max_actions:
            break

        if action_count and action_count % 50 == 0:
            pd.DataFrame(results).to_csv(args.out_csv, index=False)

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)

    print("\n--- RESULTS ---")
    print(f"Base Step Success:    {df['base_step_success'].mean():.2%}")
    print(f"Steered Step Success: {df['steered_step_success'].mean():.2%}")
    print(f"Saved to {args.out_csv}")

if __name__ == "__main__":
    run()
