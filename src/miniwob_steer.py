"""
Representation Engineering for Web Agents
=========================================
Steering LLM web agents via Contrastive Activation Addition (CAA).

This script implements zero-shot steering for MiniWob++ benchmark tasks.
The hypothesis: steering can improve action-space understanding without
task-specific fine-tuning.
"""

import argparse
import json
import random
import re

import gymnasium as gym
import miniwob
import numpy as np
import torch
from miniwob.action import ActionTypes
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}

# =============================================================================
# TASK CONFIGURATION
# =============================================================================

# All supported MiniWob++ tasks
TASKS = [
    # Click tasks
    "click-test", "click-test-2", "click-test-transfer",
    "click-button", "click-link", "click-color",
    "click-dialog", "click-dialog-2",
    "click-pie", "click-pie-nodelay",
    "click-shape", "click-tab", "click-widget",
    "click-checkboxes", "click-option",
    # Type tasks
    "focus-text", "focus-text-2", "unicode-test",
    "enter-date", "enter-time",
    # Selection tasks
    "choose-list", "choose-date",
    # Other tasks
    "grid-coordinate", "identify-shape", "guess-number",
]

# =============================================================================
# SYSTEM PROMPT (Unified, Format-Neutral)
# =============================================================================

SYSTEM_PROMPT = (
    "You are a web automation agent. Execute the task by outputting action commands.\n"
    "Rules:\n"
    "- Output only action commands, one per line\n"
    "- No explanations, no reasoning, no markdown\n"
    "- Format: click ref=N | type ref=N text=\"...\" | select ref=N option=\"...\""
)

ACTION_FORMAT = (
    "Available actions:\n"
    "- click ref=<int>\n"
    "- type ref=<int> text=\"<text>\"\n"
    "- select ref=<int> option=\"<text>\""
)

# =============================================================================
# STEERING PROMPT CONFIGURATIONS
# =============================================================================

PROMPT_CONFIGS = {
    # --- ORIGINAL PROMPTS ---
    "accuracy": {
        "pos": "Be accurate and precise. Ensure your answer is exactly correct.",
        "neg": "Be inaccurate and imprecise. Answer without checking.",
    },
    "verification": {
        "pos": "Before responding, carefully verify that your selected element matches ALL required attributes. Double-check your answer.",
        "neg": "Respond immediately with your first instinct. Skip verification.",
    },
    "format": {
        "pos": "Output only the action command. No explanations.",
        "neg": "Explain your reasoning in detail before the action.",
    },

    # --- TIER 1: HIGH-CONFIDENCE ---
    "refined_accuracy": {
        "pos": "Be accurate and precise. Read each element carefully. Match the exact requirements before responding.",
        "neg": "Be inaccurate and imprecise. Skim quickly. Respond without matching requirements.",
    },
    "attention": {
        "pos": "Pay close attention to every detail. Consider each option carefully before deciding.",
        "neg": "Pay no attention to details. Make a quick decision without considering options.",
    },
    "confidence": {
        "pos": "Be confident and decisive. Output your answer directly with no hesitation.",
        "neg": "Be uncertain and hesitant. Express doubt and explain your uncertainty.",
    },
    "format_accuracy": {
        "pos": "Output one precise action. Be accurate. No explanations.",
        "neg": "Explain your reasoning. Be careless. Verbose output.",
    },

    # --- TIER 2: MEDIUM-CONFIDENCE ---
    "element_selection": {
        "pos": "Select the element that exactly matches the task. Verify the ref number is correct.",
        "neg": "Select any element without checking. Don't verify the ref number.",
    },
    "attribute_matching": {
        "pos": "Match all attributes exactly. The text, id, and class must align with requirements.",
        "neg": "Ignore attribute matching. Select based on first impression only.",
    },
    "task_compliance": {
        "pos": "Follow the task instruction exactly. Do precisely what is asked.",
        "neg": "Ignore the task instruction. Do something approximate or unrelated.",
    },
    "deliberation": {
        "pos": "Think carefully before acting. Consider the consequences of your choice.",
        "neg": "Act impulsively. Don't think about consequences.",
    },

    # --- TIER 3: EXPLORATORY ---
    "minimalism": {
        "pos": "Respond with the absolute minimum. One line. No extra words.",
        "neg": "Respond with maximum verbosity. Explain everything in detail.",
    },
    "goal_directed": {
        "pos": "Achieve the goal successfully. Ensure your action leads to task completion.",
        "neg": "Don't care about the goal. Your action doesn't need to work.",
    },
    "self_correction": {
        "pos": "Check your answer before responding. Correct any mistakes silently.",
        "neg": "Output your first thought. Don't check or correct anything.",
    },
    "dom_reading": {
        "pos": "Read the HTML structure carefully. Parse each element's attributes.",
        "neg": "Skim the HTML quickly. Don't parse element attributes.",
    },

    # --- TIER 4: COMPOSITIONAL ---
    "composite_1": {
        "pos": "One line output. Pay attention. Be accurate.",
        "neg": "Verbose explanation. Inattentive. Inaccurate.",
    },
    "composite_2": {
        "pos": "Minimal output. Precise action. Follow task exactly.",
        "neg": "Maximum verbosity. Imprecise. Ignore task.",
    },
    "composite_3": {
        "pos": "Confident. One line. Achieve the goal.",
        "neg": "Uncertain. Explain at length. Don't care about goal.",
    },
}

# =============================================================================
# STEERED MODEL
# =============================================================================

class SteeredModel:
    """LLM with activation steering capability."""

    def __init__(self, model_name, layer_idx, coeff, steer_all_layers=False, vector_method="response"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.layer_idx = layer_idx
        self.coeff = coeff
        self.steer_all_layers = steer_all_layers
        self.vector_method = vector_method
        self.vector = None
        self._vector_cache = {}

    def _last_token_state(self, text):
        """Extract activation from last token of text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states[self.layer_idx][0, -1].float().cpu().numpy()

    def _prompt_activation(self, prompt):
        """Extract activation from prompt before generation (standard CAA)."""
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states[self.layer_idx][0, -1].float().cpu().numpy()

    def set_vector(self, vec):
        """Set the steering vector."""
        self.vector = torch.tensor(vec, dtype=torch.float32, device="cpu")
        self._vector_cache.clear()

    def generate(self, prompt, steer=False, max_new_tokens=80):
        """Generate text, optionally with steering."""
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

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
            return output

        handle = self.model.model.layers[self.layer_idx].register_forward_hook(hook)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        handle.remove()

        generated_tokens = out[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# =============================================================================
# DOM PROCESSING
# =============================================================================

def dom_to_html(dom_elements, max_elems=80):
    """Convert DOM elements to HTML string."""
    lines = []
    for el in dom_elements:
        text = (el.get("text") or "").strip().replace("\n", " ")
        value = (el.get("value") or "").strip().replace("\n", " ")
        elem_id = (el.get("id") or "").strip()
        classes = (el.get("classes") or "").strip()
        if not (text or value or elem_id or classes):
            continue
        tag = el.get("tag") or "div"
        attrs = [f'ref="{el["ref"]}"']
        if elem_id:
            attrs.append(f'id="{elem_id}"')
        if classes:
            attrs.append(f'class="{classes}"')
        if value:
            attrs.append(f'value="{value}"')
        attr_text = " " + " ".join(attrs)
        lines.append(f"<{tag}{attr_text}>{text}</{tag}>")
        if len(lines) >= max_elems:
            break
    return "\n".join(lines)


def build_prompt(obs, max_elems=80):
    """Build prompt from observation."""
    dom_text = dom_to_html(obs["dom_elements"], max_elems)
    return f"{SYSTEM_PROMPT}\n\nTask: {obs['utterance']}\n\nHTML:\n{dom_text}\n\n{ACTION_FORMAT}"

# =============================================================================
# ACTION PARSING
# =============================================================================

def parse_action(text):
    """Parse action(s) from model output. Returns list of actions or None."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None

    actions = []
    for line in lines:
        # Click
        match = re.fullmatch(r"click\s+ref=(\d+)", line, flags=re.IGNORECASE)
        if match:
            actions.append({"action": "CLICK", "ref": int(match.group(1))})
            continue

        # Type
        match = re.fullmatch(r'type\s+ref=(\d+)\s+text="(.*)"', line, flags=re.IGNORECASE)
        if match:
            actions.append({"action": "TYPE", "ref": int(match.group(1)), "text": match.group(2)})
            continue

        # Select
        match = re.fullmatch(r'select\s+ref=(\d+)\s+option="(.*)"', line, flags=re.IGNORECASE)
        if match:
            actions.append({"action": "SELECT", "ref": int(match.group(1)), "option": match.group(2)})
            continue

    return actions if actions else None

# =============================================================================
# ENVIRONMENT INTERACTION
# =============================================================================

def step_env(env, actions):
    """Execute action(s) in environment."""
    if not actions:
        act = env.unwrapped.create_action(ActionTypes.NONE)
        _obs, reward, terminated, truncated, _info = env.step(act)
        return reward, terminated or truncated

    final_reward = 0
    final_terminated = False

    for action in actions:
        if action["action"] == "CLICK":
            act = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=action["ref"])
        elif action["action"] == "SELECT":
            act = env.unwrapped.create_action(
                ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
                ref=action["ref"],
                text=action.get("option", ""),
            )
        else:  # TYPE
            act = env.unwrapped.create_action(
                ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
                ref=action["ref"],
                text=action.get("text", ""),
            )

        _obs, reward, terminated, truncated, _info = env.step(act)
        final_reward = reward
        final_terminated = terminated or truncated

        if final_terminated:
            break

    return final_reward, final_terminated

# =============================================================================
# STEERING VECTOR COMPUTATION
# =============================================================================

def compute_vector(model, tasks, steps, max_elems, max_new_tokens, prompt_type):
    """Compute steering vector from contrastive prompts."""
    
    if prompt_type == "combined":
        print("Computing Combined Vector (format_accuracy + composite_1)...")
        vec_a_sum = None
        vec_b_sum = None
        
        pbar = tqdm(total=steps, desc="Computing combined vector")
        steps_per_task = max(1, steps // len(tasks))

        for task in tasks:
            env = gym.make(f"miniwob/{task}-v1")
            for _ in range(steps_per_task):
                seed = random.randint(0, 2**31 - 1)
                obs, _ = env.reset(seed=seed)
                prompt = build_prompt(obs, max_elems)
                
                # Vector A: format_accuracy
                pos_a = f"{prompt}\n{PROMPT_CONFIGS['format_accuracy']['pos']}"
                neg_a = f"{prompt}\n{PROMPT_CONFIGS['format_accuracy']['neg']}"
                
                if model.vector_method == "prompt":
                    diff_a = model._prompt_activation(pos_a) - model._prompt_activation(neg_a)
                else:
                    pos_text = model.generate(pos_a, steer=False, max_new_tokens=max_new_tokens)
                    neg_text = model.generate(neg_a, steer=False, max_new_tokens=max_new_tokens)
                    diff_a = model._last_token_state(pos_text) - model._last_token_state(neg_text)
                
                vec_a_sum = diff_a if vec_a_sum is None else vec_a_sum + diff_a
                
                # Vector B: composite_1
                pos_b = f"{prompt}\n{PROMPT_CONFIGS['composite_1']['pos']}"
                neg_b = f"{prompt}\n{PROMPT_CONFIGS['composite_1']['neg']}"
                
                if model.vector_method == "prompt":
                    diff_b = model._prompt_activation(pos_b) - model._prompt_activation(neg_b)
                else:
                    pos_text = model.generate(pos_b, steer=False, max_new_tokens=max_new_tokens)
                    neg_text = model.generate(neg_b, steer=False, max_new_tokens=max_new_tokens)
                    diff_b = model._last_token_state(pos_text) - model._last_token_state(neg_b)
                
                vec_b_sum = diff_b if vec_b_sum is None else vec_b_sum + diff_b
                
                pbar.update(1)
                if pbar.n >= steps: break
            env.close()
            if pbar.n >= steps: break
        pbar.close()
        
        vec_a = vec_a_sum / max(1, pbar.n)
        vec_b = vec_b_sum / max(1, pbar.n)
        
        vec_a = vec_a / np.linalg.norm(vec_a)
        vec_b = vec_b / np.linalg.norm(vec_b)
        
        combined = vec_a + vec_b
        combined = combined / np.linalg.norm(combined)
        model.set_vector(combined)
        return

    # Standard single-prompt logic
    totals = None
    pos_instr = PROMPT_CONFIGS[prompt_type]["pos"]
    neg_instr = PROMPT_CONFIGS[prompt_type]["neg"]
    
    pbar = tqdm(total=steps, desc="Computing steering vector")
    steps_per_task = max(1, steps // len(tasks))

    for task in tasks:
        env = gym.make(f"miniwob/{task}-v1")
        for _ in range(steps_per_task):
            seed = random.randint(0, 2**31 - 1)
            obs, _ = env.reset(seed=seed)
            prompt = build_prompt(obs, max_elems)
            pos = f"{prompt}\n{pos_instr}"
            neg = f"{prompt}\n{neg_instr}"

            if model.vector_method == "prompt":
                diff = model._prompt_activation(pos) - model._prompt_activation(neg)
            else:
                pos_text = model.generate(pos, steer=False, max_new_tokens=max_new_tokens)
                neg_text = model.generate(neg, steer=False, max_new_tokens=max_new_tokens)
                diff = model._last_token_state(pos_text) - model._last_token_state(neg_text)

            totals = diff if totals is None else totals + diff
            pbar.update(1)

            if pbar.n >= steps:
                break
        env.close()
        if pbar.n >= steps:
            break
    pbar.close()

    vec = totals / max(1, pbar.n)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    model.set_vector(vec)

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, tasks, steps, max_elems, max_new_tokens, out_path, base_only=False):
    """Evaluate model on tasks, comparing baseline vs steered."""
    base_hits = 0
    steer_hits = 0
    parse_fails_base = 0
    parse_fails_steer = 0
    total = 0

    pbar = tqdm(total=steps, desc="Evaluating")
    steps_per_task = max(1, steps // len(tasks))

    with open(out_path, "w", encoding="utf-8") as f:
        for task in tasks:
            env = gym.make(f"miniwob/{task}-v1")
            for _ in range(steps_per_task):
                seed = random.randint(0, 2**31 - 1)
                obs, _ = env.reset(seed=seed)
                prompt = build_prompt(obs, max_elems)

                # Baseline
                base_out = model.generate(prompt, steer=False, max_new_tokens=max_new_tokens)
                base_action = parse_action(base_out)
                base_reward, _ = step_env(env, base_action)
                base_success = base_reward > 0
                base_hits += int(base_success)
                if base_action is None:
                    parse_fails_base += 1

                record = {
                    "task": task,
                    "seed": seed,
                    "base_output": base_out,
                    "base_action": base_action,
                    "base_success": base_success,
                }

                if not base_only:
                    # Steered
                    obs, _ = env.reset(seed=seed)
                    steer_out = model.generate(prompt, steer=True, max_new_tokens=max_new_tokens)
                    steer_action = parse_action(steer_out)
                    steer_reward, _ = step_env(env, steer_action)
                    steer_success = steer_reward > 0
                    steer_hits += int(steer_success)
                    if steer_action is None:
                        parse_fails_steer += 1

                    record.update({
                        "steer_output": steer_out,
                        "steer_action": steer_action,
                        "steer_success": steer_success,
                    })

                f.write(json.dumps(record) + "\n")
                total += 1
                pbar.update(1)

                if base_only:
                    pbar.set_postfix(acc=f"{base_hits/total:.1%}")
                else:
                    pbar.set_postfix(
                        base=f"{base_hits/total:.1%}",
                        steer=f"{steer_hits/total:.1%}",
                        delta=f"{(steer_hits-base_hits)/total:+.1%}"
                    )

                if pbar.n >= steps:
                    break
            env.close()
            if pbar.n >= steps:
                break
    pbar.close()

    base_acc = base_hits / max(1, total)
    steer_acc = steer_hits / max(1, total) if not base_only else 0

    return {
        "base_accuracy": base_acc,
        "steer_accuracy": steer_acc,
        "improvement": steer_acc - base_acc,
        "base_parse_fail": parse_fails_base / max(1, total),
        "steer_parse_fail": parse_fails_steer / max(1, total) if not base_only else 0,
        "total_episodes": total,
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Web Agent Steering Experiment")
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="0.5b")
    parser.add_argument("--layer", type=int, default=14, help="Intervention layer")
    parser.add_argument("--coeff", type=float, default=4.0, help="Steering coefficient")
    parser.add_argument("--prompt-type", choices=list(PROMPT_CONFIGS.keys()) + ["combined"], default="accuracy")
    parser.add_argument("--vector-method", choices=["response", "prompt"], default="response")
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--tasks", default="all", help="Task list or 'all'")
    parser.add_argument("--out", default="results.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-only", action="store_true", help="Evaluate baseline only")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Register environments
    gym.register_envs(miniwob)

    # Select tasks
    if args.tasks == "all":
        tasks = TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    # Initialize model
    print(f"Model: {args.model}, Layer: {args.layer}, Coeff: {args.coeff}")
    print(f"Prompt type: {args.prompt_type}, Vector method: {args.vector_method}")

    model = SteeredModel(
        MODEL_MAP[args.model],
        layer_idx=args.layer,
        coeff=args.coeff,
        vector_method=args.vector_method,
    )

    # Compute steering vector
    if not args.base_only:
        compute_vector(model, tasks, args.train_steps, 80, 80, args.prompt_type)

    # Evaluate
    results = evaluate(model, tasks, args.eval_steps, 80, 80, args.out, args.base_only)

    # Print results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Baseline Accuracy:  {results['base_accuracy']:.1%}")
    if not args.base_only:
        print(f"Steered Accuracy:   {results['steer_accuracy']:.1%}")
        print(f"Improvement:        {results['improvement']:+.1%}")
        print(f"Parse Fail (base):  {results['base_parse_fail']:.1%}")
        print(f"Parse Fail (steer): {results['steer_parse_fail']:.1%}")
    print(f"Total Episodes:     {results['total_episodes']}")
    print(f"Output:             {args.out}")


if __name__ == "__main__":
    main()