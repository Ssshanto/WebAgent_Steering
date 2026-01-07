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

MODEL_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}

SINGLE_STEP_TASKS = [
    # Simple click tasks
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
    # Text input tasks
    "focus-text",
    "focus-text-2",
    "unicode-test",
    # Other simple tasks
    "grid-coordinate",
    "identify-shape",
    # Complex interaction tasks (NEW)
    "click-checkboxes",     # Multi-selection state
    "click-option",         # RadioButton state
    "choose-list",          # Dropdown interaction
    "choose-date",          # Date picker interaction
    "enter-date",           # Semantic constraint typing
    "enter-time",           # Semantic constraint typing
    "guess-number",         # Feedback loop/logic
]

# Task categories for analysis
TASK_CATEGORIES = {
    "simple_click": [
        "click-test", "click-test-2", "click-test-transfer", "click-button",
        "click-link", "click-color", "click-dialog", "click-dialog-2",
        "click-pie", "click-pie-nodelay", "click-shape", "click-tab", "click-widget"
    ],
    "simple_type": [
        "focus-text", "focus-text-2", "unicode-test"
    ],
    "multi_select": [
        "click-checkboxes", "click-option"
    ],
    "dropdown": [
        "choose-list", "choose-date"
    ],
    "semantic_type": [
        "enter-date", "enter-time"
    ],
    "logic": [
        "guess-number"
    ],
    "other": [
        "grid-coordinate", "identify-shape"
    ]
}

SYSTEM_PROMPT = (
    "You are a web automation engine. Output action commands.\n"
    "Strict format rules:\n"
    "- Output one action per line.\n"
    "- For single actions: output exactly one line.\n"
    "- For multiple actions: output multiple lines (e.g., for checkboxes).\n"
    "- No explanations, no preamble, no lists, no code fences.\n"
    "- Each line must match one of the allowed action formats."
)
ACTION_FORMAT = (
    "Actions (match exactly):\n"
    "- click ref=<int>\n"
    "- type ref=<int> text=\"<text>\"\n"
    "- select ref=<int> option=\"<text>\""
)
# Steering prompt configurations
# Select via --prompt-type argument
PROMPT_CONFIGS = {
    # === ORIGINAL PROMPTS (Exp 1-5) ===
    # Verification-focused (Exp 3-4): Target cognitive process
    "verification": {
        "pos": "Before responding, carefully verify that your selected element matches ALL required attributes. Double-check your answer against the task requirements.",
        "neg": "Respond immediately with your first instinct. Skip verification and double-checking.",
    },
    # Format-focused (Exp 5): Target output compliance - good for smaller models
    "format": {
        "pos": "Output exactly one line with the action command. No explanations, no extra text, just the action.",
        "neg": "Explain your reasoning step by step before giving the action. Be verbose and detailed.",
    },
    # Accuracy-focused (Exp 1-2): Abstract outcome - CURRENT BEST
    "accuracy": {
        "pos": "Be accurate and precise. Read the given information carefully. Ensure your answer is exactly correct before responding.",
        "neg": "Be inaccurate and imprecise. Skim the given information quickly. Answer without ensuring correctness.",
    },

    # === TIER 1: HIGH-CONFIDENCE (Exp 7) ===
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

    # === TIER 2: MEDIUM-CONFIDENCE ===
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

    # === TIER 3: EXPLORATORY ===
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

    # === TIER 4: COMPOSITIONAL ===
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

# Default prompt type
PROMPT_TYPE = "verification"
POS_INSTR = PROMPT_CONFIGS[PROMPT_TYPE]["pos"]
NEG_INSTR = PROMPT_CONFIGS[PROMPT_TYPE]["neg"]

# High-potential task subset (65-86% base accuracy) - used in Exp 3
# Result: Ceiling effect at 89.5% base, no steering improvement
HIGH_POTENTIAL_TASKS = [
    "click-dialog",      # 100% base
    "click-dialog-2",    # 64% base
    "click-button",      # 82% base
    "click-link",        # 64% base
    "focus-text",        # 100% base
    "focus-text-2",      # 100% base
]

# Medium-difficulty task subset (54-64% base accuracy) - Exp 4 target
# These tasks have room for improvement and are solvable from DOM (not visual)
MEDIUM_DIFFICULTY_TASKS = [
    "click-widget",      # 54.5% base - widget interaction
    "click-dialog-2",    # 63.6% base - dialog with options
    "click-link",        # 63.6% base - link selection
    "click-button",      # 81.8% base - included for statistical power
]

# Expanded action space tasks (NEW - Exp 10)
# Complex interactions beyond simple click/type
EXPANDED_TASKS = [
    "click-checkboxes",  # Multi-selection state
    "click-option",      # RadioButton state
    "choose-list",       # Dropdown interaction
    "choose-date",       # Date picker interaction
    "enter-date",        # Semantic constraint typing
    "enter-time",        # Semantic constraint typing
    "guess-number",      # Feedback loop/logic
]


class SteeredModel:
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
        """Non-standard: Extract activation from last token of generated response."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states[self.layer_idx][0, -1].float().cpu().numpy()

    def _prompt_activation(self, prompt):
        """Standard CAA: Extract activation from prompt BEFORE generation."""
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states[self.layer_idx][0, -1].float().cpu().numpy()

    def set_vector(self, vec):
        self.vector = torch.tensor(vec, dtype=torch.float32, device="cpu")
        self._vector_cache.clear()

    def generate(self, prompt, steer=False, max_new_tokens=80, strip_prompt=True):
        # Format prompt using Qwen's chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
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
            else:
                target += self.coeff * vec
            return output

        # Register hooks on target layers
        if self.steer_all_layers:
            # Steer all layers from layer_idx onwards
            num_layers = len(self.model.model.layers)
            handles = [
                self.model.model.layers[i].register_forward_hook(hook)
                for i in range(self.layer_idx, num_layers)
            ]
        else:
            # Steer only the specified layer
            handles = [self.model.model.layers[self.layer_idx].register_forward_hook(hook)]

        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        for h in handles:
            h.remove()

        if strip_prompt:
            # Decode only the newly generated tokens
            generated_tokens = out[0][input_length:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return text.strip()
        else:
            # Decode everything for _last_token_state usage
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            return text


def dom_to_html(dom_elements, max_elems):
    lines = []
    for el in dom_elements:
        text = (el.get("text") or "").strip().replace("\n", " ")
        value = (el.get("value") or "").strip().replace("\n", " ")
        elem_id = (el.get("id") or "").strip()
        classes = (el.get("classes") or "").strip()
        if not (text or value or elem_id or classes):
            continue
        tag = el.get("tag") or "div"
        attrs = [f'data-ref="{el["ref"]}"']
        if elem_id:
            attrs.append(f'id="{elem_id}"')
        if classes:
            attrs.append(f'class="{classes}"')
        if value:
            attrs.append(f'value="{value}"')
        attr_text = " " + " ".join(attrs) if attrs else ""
        lines.append(f"<{tag}{attr_text}>{text}</{tag}>")
        if len(lines) >= max_elems:
            break
    return "\n".join(lines)


def build_prompt(obs, max_elems):
    dom_text = dom_to_html(obs["dom_elements"], max_elems)
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Task: {obs['utterance']}\n"
        f"HTML:\n{dom_text}\n"
        f"{ACTION_FORMAT}"
    )


def parse_action(text):
    """Parse action(s) from model output. Returns list of actions or None."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    
    actions = []
    for line in lines:
        # Try click pattern
        match = re.fullmatch(r"click\s+ref=(\d+)", line, flags=re.IGNORECASE)
        if match:
            actions.append({"action": "CLICK", "ref": int(match.group(1)), "text": "", "option": ""})
            continue
        
        # Try type pattern
        match = re.fullmatch(r'type\s+ref=(\d+)\s+text=\"(.*)\"', line, flags=re.IGNORECASE)
        if match:
            actions.append({"action": "TYPE", "ref": int(match.group(1)), "text": match.group(2), "option": ""})
            continue
        
        # Try select pattern
        match = re.fullmatch(r'select\s+ref=(\d+)\s+option=\"(.*)\"', line, flags=re.IGNORECASE)
        if match:
            actions.append({"action": "SELECT", "ref": int(match.group(1)), "text": "", "option": match.group(2)})
            continue
        
        # If line doesn't match any pattern, ignore it (allows for robustness)
    
    return actions if actions else None


def step_env(env, actions):
    """Execute action(s) in environment. Supports single or multiple actions."""
    if not actions:
        act = env.unwrapped.create_action(ActionTypes.NONE)
        _obs, reward, terminated, truncated, _info = env.step(act)
        return reward, terminated or truncated
    
    # Execute all actions sequentially
    final_reward = 0
    final_terminated = False
    
    for action in actions:
        if action["action"] == "CLICK":
            act = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=action["ref"])
        elif action["action"] == "SELECT":
            # SELECT is implemented as TYPE for dropdowns (MiniWob limitation)
            act = env.unwrapped.create_action(
                ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
                ref=action["ref"],
                text=action["option"],
            )
        else:  # TYPE
            act = env.unwrapped.create_action(
                ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
                ref=action["ref"],
                text=str(action.get("text", "")),
            )
        
        _obs, reward, terminated, truncated, _info = env.step(act)
        final_reward = reward  # Keep last reward
        final_terminated = terminated or truncated
        
        if final_terminated:
            break
    
    return final_reward, final_terminated


def split_steps(total_steps, num_tasks):
    base = total_steps // num_tasks
    extra = total_steps % num_tasks
    return [base + (1 if i < extra else 0) for i in range(num_tasks)]


def compute_vector(model, tasks, steps, max_elems, max_new_tokens):
    totals = None
    pbar = tqdm(total=steps, desc="vector")
    per_task = split_steps(steps, len(tasks))
    
    for task, count in zip(tasks, per_task):
        if count == 0:
            continue
        env = gym.make(f"miniwob/{task}-v1")
        for _ in range(count):
            seed = random.randint(0, 2**31 - 1)
            obs, _ = env.reset(seed=seed)
            prompt = build_prompt(obs, max_elems)
            pos = f"{prompt}\n{POS_INSTR}"
            neg = f"{prompt}\n{NEG_INSTR}"
            
            if model.vector_method == "prompt":
                # Standard CAA: Extract from prompt before generation
                diff = model._prompt_activation(pos) - model._prompt_activation(neg)
            else:
                # Non-standard (original): Extract from generated response
                pos_text = model.generate(
                    pos,
                    steer=False,
                    max_new_tokens=max_new_tokens,
                    strip_prompt=False,
                )
                neg_text = model.generate(
                    neg,
                    steer=False,
                    max_new_tokens=max_new_tokens,
                    strip_prompt=False,
                )
                diff = model._last_token_state(pos_text) - model._last_token_state(neg_text)
            
            totals = diff if totals is None else totals + diff
            pbar.update(1)
        env.close()
    pbar.close()

    vec = totals / max(1, steps)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    model.set_vector(vec)


def evaluate(model, tasks, steps, max_elems, max_new_tokens, out_path, base_only=False):
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
                base_hits += int(base_success)

                if base_only:
                    # Skip steered evaluation
                    record = {
                        "task": task,
                        "seed": seed,
                        "prompt": prompt,
                        "base_output": base_out,
                        "base_action": base_action,
                        "base_reward": base_reward,
                        "base_success": base_success,
                    }
                    pbar.set_postfix(base=f"{base_hits / max(1, pbar.n + 1):.2%}")
                else:
                    # Run steered evaluation
                    obs, _ = env.reset(seed=seed)
                    steer_out = model.generate(prompt, steer=True, max_new_tokens=max_new_tokens)
                    steer_action = parse_action(steer_out)
                    steer_reward, _ = step_env(env, steer_action)
                    steer_success = steer_reward > 0
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
                    pbar.set_postfix(
                        base=f"{base_hits / max(1, pbar.n + 1):.2%}",
                        steer=f"{steer_hits / max(1, pbar.n + 1):.2%}",
                    )

                f.write(json.dumps(record) + "\n")
                pbar.update(1)
            env.close()
    pbar.close()

    return base_hits / max(1, steps), steer_hits / max(1, steps)


def main():
    # Monkeypatch MiniWoB to use the correct Chrome binary
    from miniwob.selenium_instance import SeleniumInstance
    from selenium import webdriver
    
    def patched_create_driver(self):
        assert not hasattr(self, "driver"), f"Instance {self.index} already has a driver"
        options = webdriver.ChromeOptions()
        # Fix for absolute path issue on some systems
        options.binary_location = "/usr/bin/chromium-browser"
        options.add_argument(f"window-size={self.window_width},{self.window_height}")
        if self.headless:
            options.add_argument("headless")
            options.add_argument("disable-gpu")
            options.add_argument("no-sandbox")
        else:
            options.add_argument("app=" + self.url)
        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(5)
        if self.headless:
            self.driver.get(self.url)
        
        from selenium.webdriver.support.wait import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
        try:
            WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.ID, self.SYNC_SCREEN_ID))
            )
        except Exception as e:
            import logging
            logging.error("Page did not load properly. Wrong URL?")
            raise e
        self.inner_width, self.inner_height = self.driver.execute_script(
            "return [window.innerWidth, window.innerHeight];"
        )
    SeleniumInstance.create_driver = patched_create_driver

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_MAP.keys(), default="0.5b")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--coeff", type=float, default=1.0)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--max-elems", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--out", default="miniwob_results.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-only", action="store_true", help="Skip steering, evaluate base model only")
    parser.add_argument("--steer-all-layers", action="store_true", help="Apply steering to all layers from --layer onwards (multi-layer steering)")
    parser.add_argument("--prompt-type", choices=PROMPT_CONFIGS.keys(), default="verification",
                        help="Steering prompt type: verification, format, or accuracy")
    parser.add_argument("--vector-method", choices=["response", "prompt"], default="response",
                        help="Vector computation method: 'response' (non-standard, from generated text) or 'prompt' (standard CAA, from prompt only)")
    args = parser.parse_args()

    # Set steering prompts based on --prompt-type
    global POS_INSTR, NEG_INSTR
    POS_INSTR = PROMPT_CONFIGS[args.prompt_type]["pos"]
    NEG_INSTR = PROMPT_CONFIGS[args.prompt_type]["neg"]
    print(f"Using prompt type: {args.prompt_type}")
    print(f"Using vector method: {args.vector_method}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gym.register_envs(miniwob)

    if args.tasks == "all":
        tasks = SINGLE_STEP_TASKS
    elif args.tasks == "high-potential":
        tasks = HIGH_POTENTIAL_TASKS
    elif args.tasks == "medium":
        tasks = MEDIUM_DIFFICULTY_TASKS
    elif args.tasks == "expanded":
        tasks = EXPANDED_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    model_name = MODEL_MAP[args.model_size]
    model = SteeredModel(
        model_name,
        layer_idx=args.layer,
        coeff=args.coeff,
        steer_all_layers=args.steer_all_layers,
        vector_method=args.vector_method
    )

    if not args.base_only:
        compute_vector(model, tasks, args.train_steps, args.max_elems, args.max_new_tokens)

    base_acc, steer_acc = evaluate(
        model,
        tasks,
        args.eval_steps,
        args.max_elems,
        args.max_new_tokens,
        args.out,
        base_only=args.base_only,
    )

    print(f"Base accuracy: {base_acc:.2%}")
    if not args.base_only:
        print(f"Steer accuracy: {steer_acc:.2%}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
