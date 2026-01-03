import argparse
import json
import random

import gymnasium as gym
import miniwob
import numpy as np
import torch
from miniwob.action import ActionTypes
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}

SINGLE_STEP_TASKS = [
    "click-test",
    "click-test-2",
    "click-test-transfer",
    "click-button",
    "click-link",
    "click-color",
    "click-dialog",
    "click-dialog-2",
    "click-pie",
    "click-pie-nodelay",
    "click-shape",
    "click-tab",
    "click-widget",
    "focus-text",
    "focus-text-2",
    "grid-coordinate",
    "identify-shape",
    "unicode-test",
]

SYSTEM_PROMPT = "You are a web automation engine. Output a JSON action only."
FORMAT_EXAMPLE = {"action": "CLICK|TYPE", "ref": 123, "text": "optional"}
POS_INSTR = "Respond with strict JSON action only."
NEG_INSTR = "Explain the action in natural language."


class SteeredModel:
    def __init__(self, model_name, layer_idx, coeff):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.layer_idx = layer_idx
        self.coeff = coeff
        self.vector = None
        self._vector_cache = {}

    def _last_token_state(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states[self.layer_idx][0, -1].float().cpu().numpy()

    def set_vector(self, vec):
        self.vector = torch.tensor(vec, dtype=torch.float32, device="cpu")
        self._vector_cache.clear()

    def generate(self, prompt, steer=False, max_new_tokens=80):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        def hook(_module, _input, output):
            if not steer or self.vector is None:
                return output
            if torch.is_tensor(output):
                target = output
            elif isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                target = output[0]
            else:
                return output

            vec = self._vector_cache.get(target.device)
            if vec is None:
                vec = self.vector.to(device=target.device, dtype=target.dtype)
                self._vector_cache[target.device] = vec

            if target.dim() == 3:
                target[:, -1, :] += self.coeff * vec
            elif target.dim() == 2:
                target[-1, :] += self.coeff * vec
            else:
                target += self.coeff * vec
            return output

        handles = []
        for i in range(self.layer_idx, len(self.model.model.layers)):
            handles.append(self.model.model.layers[i].register_forward_hook(hook))

        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        for h in handles:
            h.remove()

        if text.startswith(prompt):
            return text[len(prompt) :].strip()
        return text.strip()


def dom_to_text(dom_elements, max_elems):
    lines = []
    for el in dom_elements:
        text = (el.get("text") or "").strip().replace("\n", " ")
        value = (el.get("value") or "").strip().replace("\n", " ")
        elem_id = (el.get("id") or "").strip()
        classes = (el.get("classes") or "").strip()
        if not (text or value or elem_id or classes):
            continue
        parts = [f"ref={el['ref']}", f"tag={el.get('tag', '')}"]
        if text:
            parts.append(f'text="{text}"')
        if value:
            parts.append(f'value="{value}"')
        if elem_id:
            parts.append(f'id="{elem_id}"')
        if classes:
            parts.append(f'class="{classes}"')
        lines.append(" ".join(parts))
        if len(lines) >= max_elems:
            break
    return "\n".join(lines)


def build_prompt(obs, max_elems):
    dom_text = dom_to_text(obs["dom_elements"], max_elems)
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Task: {obs['utterance']}\n"
        f"DOM:\n{dom_text}\n"
        f"Return JSON only in this format: {json.dumps(FORMAT_EXAMPLE)}"
    )


def parse_action(text):
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    action = (obj.get("action") or "").upper()
    ref = obj.get("ref")
    if action not in ("CLICK", "TYPE"):
        return None
    if not isinstance(ref, int):
        return None
    return {"action": action, "ref": ref, "text": obj.get("text", "")}


def step_env(env, action):
    if not action:
        act = env.unwrapped.create_action(ActionTypes.NONE)
    elif action["action"] == "CLICK":
        act = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=action["ref"])
    else:
        act = env.unwrapped.create_action(
            ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
            ref=action["ref"],
            text=str(action.get("text", "")),
        )
    _obs, reward, terminated, truncated, _info = env.step(act)
    return reward, terminated or truncated


def split_steps(total_steps, num_tasks):
    base = total_steps // num_tasks
    extra = total_steps % num_tasks
    return [base + (1 if i < extra else 0) for i in range(num_tasks)]


def compute_vector(model, tasks, steps, max_elems):
    totals = None
    pbar = tqdm(total=steps, desc="vector")
    per_task = split_steps(steps, len(tasks))
    for task, count in zip(tasks, per_task):
        if count == 0:
            continue
        env = gym.make(f"miniwob/{task}-v1")
        for _ in range(count):
            obs, _ = env.reset()
            prompt = build_prompt(obs, max_elems)
            pos = f"{prompt}\n{POS_INSTR}"
            neg = f"{prompt}\n{NEG_INSTR}"
            diff = model._last_token_state(pos) - model._last_token_state(neg)
            totals = diff if totals is None else totals + diff
            pbar.update(1)
        env.close()
    pbar.close()

    vec = totals / max(1, steps)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    model.set_vector(vec)


def evaluate(model, tasks, steps, max_elems, max_new_tokens, out_path):
    base_hits = 0
    steer_hits = 0
    pbar = tqdm(total=steps, desc="eval")
    per_task = split_steps(steps, len(tasks))

    with open(out_path, "w", encoding="utf-8") as f:
        for task, count in zip(tasks, per_task):
            if count == 0:
                continue
            env = gym.make(f"miniwob/{task}-v1")
            for _ in range(count):
                seed = random.randint(0, 2**31 - 1)
                obs, _ = env.reset(seed=seed)
                prompt = build_prompt(obs, max_elems)

                base_out = model.generate(prompt, steer=False, max_new_tokens=max_new_tokens)
                base_action = parse_action(base_out)
                base_reward, _ = step_env(env, base_action)
                base_success = base_reward > 0

                obs, _ = env.reset(seed=seed)
                steer_out = model.generate(prompt, steer=True, max_new_tokens=max_new_tokens)
                steer_action = parse_action(steer_out)
                steer_reward, _ = step_env(env, steer_action)
                steer_success = steer_reward > 0

                base_hits += int(base_success)
                steer_hits += int(steer_success)

                record = {
                    "task": task,
                    "seed": seed,
                    "prompt": prompt,
                    "base_output": base_out,
                    "base_action": base_action,
                    "base_reward": base_reward,
                    "base_success": base_success,
                    "steer_output": steer_out,
                    "steer_action": steer_action,
                    "steer_reward": steer_reward,
                    "steer_success": steer_success,
                }
                f.write(json.dumps(record) + "\n")

                pbar.update(1)
                denom = max(1, pbar.n)
                pbar.set_postfix(
                    base=f"{base_hits / denom:.2%}",
                    steer=f"{steer_hits / denom:.2%}",
                )
            env.close()
    pbar.close()

    return base_hits / max(1, steps), steer_hits / max(1, steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_MAP.keys(), default="0.5b")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--coeff", type=float, default=2.0)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--max-elems", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--out", default="miniwob_results.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gym.register_envs(miniwob)

    if args.tasks == "all":
        tasks = SINGLE_STEP_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    model_name = MODEL_MAP[args.model_size]
    model = SteeredModel(model_name, layer_idx=args.layer, coeff=args.coeff)

    compute_vector(model, tasks, args.train_steps, args.max_elems)
    base_acc, steer_acc = evaluate(
        model,
        tasks,
        args.eval_steps,
        args.max_elems,
        args.max_new_tokens,
        args.out,
    )

    print(f"Base accuracy: {base_acc:.2%}")
    print(f"Steer accuracy: {steer_acc:.2%}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
