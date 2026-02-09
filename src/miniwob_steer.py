"""
Representation Engineering for Web Agents
=========================================
Steering LLM web agents via Contrastive Activation Addition (CAA).

This script implements zero-shot steering for MiniWob++ benchmark tasks.
The hypothesis: steering can improve action-space understanding without
task-specific fine-tuning.

Supports:
- Text-only LLMs (Qwen, Llama, Gemma, Phi, SmolLM)
"""

import argparse
import json
import os
import random

import gymnasium as gym
import browsergym.miniwob
from browsergym.core.action.highlevel import HighLevelActionSet
import numpy as np
import torch
from browsergym.utils.obs import flatten_dom_to_str, flatten_axtree_to_str, prune_html
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_MAP = {
    # Qwen family (original)
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    # Qwen family (extended)
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-coder-0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    # Qwen3 family
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    # Llama family
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    # Other families
    "gemma-2b": "google/gemma-2-2b-it",
    "gemma-1b": "google/gemma-3-1b-it",
    "phi-3.8b": "microsoft/Phi-3.5-mini-instruct",
    "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "stablelm-1.6b": "stabilityai/stablelm-2-1_6b-chat",
    "opt-iml-1.3b": "facebook/opt-iml-1.3b",
}

# Layer depths for 50% intervention point (mid-layer steering)
LAYER_MAP = {
    "0.5b": 11,  # 24 layers → L11 (46%)
    "3b": 18,  # 36 layers → L18 (50%)
    "qwen-7b": 16,  # 32 layers → L16 (50%)
    "qwen-1.5b": 14,  # 28 layers → L14 (50%)
    "qwen-coder-0.5b": 11,  # 24 layers → L11 (46%)
    "llama-1b": 8,  # 16 layers → L8 (50%)
    "llama-3b": 14,  # 28 layers → L14 (50%)
    "gemma-2b": 13,  # 26 layers → L13 (50%)
    "gemma-1b": 13,  # 26 layers → L13 (50%)
    "phi-3.8b": 16,  # 32 layers → L16 (50%)
    "smollm-1.7b": 12,  # 24 layers → L12 (50%)
    "smollm-360m": 16,  # 32 layers → L16 (50%)
    "tinyllama-1.1b": 11,  # 22 layers → L11 (50%)
    "stablelm-1.6b": 12,  # 24 layers → L12 (50%)
    "opt-iml-1.3b": 12,  # 24 layers → L12 (50%)
}

# Model architecture types for layer access patterns
# Different model families have different internal structures
MODEL_ARCH = {
    # Qwen/Llama-style: model.model.layers
    "0.5b": "qwen",
    "3b": "qwen",
    "qwen-7b": "qwen",
    "qwen-1.5b": "qwen",
    "qwen-coder-0.5b": "qwen",
    "qwen3-0.6b": "qwen",
    "qwen3-1.7b": "qwen",
    "qwen3-4b": "qwen",
    "qwen3-8b": "qwen",
    "llama-1b": "llama",
    "llama-3b": "llama",
    "gemma-2b": "gemma",
    "gemma-1b": "gemma",
    "phi-3.8b": "phi",
    # SmolLM2 is Llama-based: model.model.layers
    "smollm-1.7b": "qwen",
    "smollm-360m": "qwen",
    # TinyLlama uses Llama-2 style: model.model.layers
    "tinyllama-1.1b": "llama",
    # StableLM 2 is Llama-like: model.model.layers
    "stablelm-1.6b": "qwen",
    # OPT uses: model.model.decoder.layers
    "opt-iml-1.3b": "opt",
}

# Models that don't support chat templates (need raw prompting)
NO_CHAT_TEMPLATE = {"opt-iml-1.3b"}


def _apply_template(tokenizer, messages, model_key):
    """Apply chat template with model-specific options."""
    # Qwen3 defaults to reasoning mode; disable it for action-only outputs.
    if model_key and model_key.startswith("qwen3-"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return f"{text}\n/no_think"

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_layer(model_key, layer_arg):
    """Get layer index, supporting 'auto' for automatic selection."""
    if layer_arg == "auto":
        return LAYER_MAP.get(model_key, 14)
    return int(layer_arg)


def get_model_layers(model, model_key):
    """Get the layers module for a given model based on its architecture.

    Different model families have different internal structures:
    - Qwen/Llama/Gemma/Phi: model.model.layers
    - SmolLM (GPT-2 style): model.transformer.h
    - StableLM (GPTNeoX style): model.gpt_neox.layers
    - OPT: model.model.decoder.layers
    """
    arch = MODEL_ARCH.get(model_key, "qwen")

    if arch in ("qwen", "llama", "gemma", "phi"):
        return model.model.layers
    elif arch == "smollm":
        return model.transformer.h
    elif arch == "stablelm":
        return model.gpt_neox.layers
    elif arch == "opt":
        return model.model.decoder.layers
    else:
        # Default fallback
        return model.model.layers


def get_additional_stop_tokens(tokenizer, model_key):
    """Get additional stop tokens for specific models.

    Different models use different end-of-turn tokens:
    - Llama: <|eot_id|>
    - Gemma: <end_of_turn>
    - TinyLlama: </s>
    """
    stop_tokens = []
    arch = MODEL_ARCH.get(model_key, "qwen")

    # Llama-style models
    if arch == "llama":
        try:
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != tokenizer.unk_token_id:
                stop_tokens.append(eot_id)
        except Exception:
            pass

    # Gemma models
    elif arch == "gemma":
        try:
            end_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_turn is not None and end_turn != tokenizer.unk_token_id:
                stop_tokens.append(end_turn)
        except Exception:
            pass

    # StableLM models
    elif arch == "stablelm":
        try:
            # StableLM uses <|endoftext|> and sometimes <|im_end|>
            for token in ["<|endoftext|>", "<|im_end|>"]:
                tok_id = tokenizer.convert_tokens_to_ids(token)
                if tok_id is not None and tok_id != tokenizer.unk_token_id:
                    stop_tokens.append(tok_id)
        except Exception:
            pass

    return stop_tokens


# =============================================================================
# TASK CONFIGURATION
# =============================================================================


def list_miniwob_tasks():
    """Return the full MiniWob++ task list from the Gym registry."""
    env_ids = [
        env_id
        for env_id in gym.envs.registry.keys()
        if env_id.startswith("browsergym/miniwob.")
    ]
    tasks = [env_id.split("browsergym/miniwob.", 1)[1] for env_id in env_ids]
    return sorted(tasks)


def make_miniwob_env(task):
    """Create MiniWob env with BrowserGym-native action mapping."""
    return gym.make(
        f"browsergym/miniwob.{task}",
        action_mapping=DEMO_PROMPT_ACTION_SET.to_python_code,
    )


# =============================================================================
# SYSTEM PROMPT (Unified, Format-Neutral)
# =============================================================================

SYSTEM_PROMPT = (
    "# Instructions\n\n"
    "Review the current state of the page and all other information to find the best\n"
    "possible next action to accomplish your goal. Your answer will be interpreted\n"
    "and executed by a program, make sure to follow the formatting instructions."
)

DEMO_PROMPT_ACTION_SET = HighLevelActionSet(
    subsets=["miniwob_all"],
    strict=False,
    multiaction=False,
    demo_mode="off",
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
        "pos": "Select the element that exactly matches the task. Verify the bid is correct.",
        "neg": "Select any element without checking. Don't verify the bid.",
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
    "failure_conditioned": {
        "pos": "Output exactly one valid BrowserGym action. Use an existing bid from the current DOM. Match action type to element capability. Avoid non-interactable targets and malformed action syntax.",
        "neg": "Output an invalid or careless action. Use a missing or wrong bid, mismatched action type for the element, and malformed or non-interactable action syntax.",
    },
}

# =============================================================================
# STEERED MODEL
# =============================================================================


class SteeredModel:
    """LLM with activation steering capability."""

    def __init__(
        self,
        model_name,
        layer_idx,
        coeff,
        steer_all_layers=False,
        vector_method="response",
        model_key=None,
        steer_action_window=False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_key = model_key  # Store for architecture-specific handling
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Detect stop tokens based on model architecture
        self.stop_token_ids = [self.tokenizer.eos_token_id]
        self.stop_token_ids.extend(
            get_additional_stop_tokens(self.tokenizer, model_key)
        )
        # Remove duplicates and None values
        self.stop_token_ids = list(set(t for t in self.stop_token_ids if t is not None))

        # Check if model needs raw prompting (no chat template)
        self.use_chat_template = model_key not in NO_CHAT_TEMPLATE

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.layer_idx = layer_idx
        self.coeff = coeff
        self.steer_all_layers = steer_all_layers
        self.vector_method = vector_method
        self.vector = None  # Active steering vector for current layer
        self.vectors = {}  # Dictionary mapping layer_idx -> vector tensor
        self._vector_cache = {}
        self.steer_action_window = steer_action_window

    def _last_token_state(self, text):
        """Extract activation from last token of text for all layers."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # Return all hidden states (tuple of tensors, one per layer)
        return out.hidden_states

    def _prompt_activation(self, prompt):
        """Extract activation from prompt before generation for all layers."""
        if self.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = _apply_template(self.tokenizer, messages, self.model_key)
        else:
            # Models without chat template (e.g., OPT)
            formatted = f"User: {prompt}\nAssistant:"
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # Return all hidden states (tuple of tensors, one per layer)
        return out.hidden_states

    def set_vector(self, vec, layer_idx=None):
        """Set the steering vector(s).

        Args:
            vec: Either a single vector (tensor/array) or a dict mapping layer_idx -> vector
            layer_idx: If vec is a single vector, which layer it's for (defaults to self.layer_idx)
        """
        if isinstance(vec, dict):
            # Setting multiple vectors at once
            self.vectors = {
                k: torch.tensor(v, dtype=torch.float32, device="cpu")
                for k, v in vec.items()
            }
            # Set active vector to the target layer if available
            if self.layer_idx in self.vectors:
                self.vector = self.vectors[self.layer_idx]
        else:
            # Setting a single vector
            target_layer = layer_idx if layer_idx is not None else self.layer_idx
            tensor_vec = torch.tensor(vec, dtype=torch.float32, device="cpu")
            self.vectors[target_layer] = tensor_vec
            if target_layer == self.layer_idx:
                self.vector = tensor_vec
        self._vector_cache.clear()

    def generate(self, prompt, steer=False, max_new_tokens=80, deterministic=False):
        """Generate text, optionally with steering."""
        if self.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = _apply_template(self.tokenizer, messages, self.model_key)
        else:
            # Models without chat template (e.g., OPT)
            formatted_prompt = f"User: {prompt}\nAssistant:"

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        hook_calls = {"count": 0}

        def hook(_module, _input, output):
            if not steer or self.vector is None:
                return output
            hook_calls["count"] += 1
            if self.steer_action_window and hook_calls["count"] == 1:
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

        # Get layers based on model architecture
        layers = get_model_layers(self.model, self.model_key)
        handle = layers[self.layer_idx].register_forward_hook(hook)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.stop_token_ids,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.model_key and self.model_key.startswith("qwen3-"):
            gen_kwargs.update(
                {
                    "do_sample": not deterministic,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        out = self.model.generate(
            **inputs,
            **gen_kwargs,
        )
        handle.remove()

        generated_tokens = out[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# =============================================================================
# DOM PROCESSING
# =============================================================================


def build_prompt(obs, max_elems=80):
    """Build BrowserGym demo_agent-style prompt from observation."""
    _ = max_elems
    dom_text = prune_html(flatten_dom_to_str(obs["dom_object"]))
    axtree_text = flatten_axtree_to_str(obs["axtree_object"])

    goal = obs.get("goal", "")
    goal_obj = obs.get("goal_object")
    if isinstance(goal_obj, list):
        goal_parts = []
        for x in goal_obj:
            if isinstance(x, dict) and x.get("type") == "text":
                goal_parts.append(str(x.get("text", "")))
        if goal_parts:
            goal = "\n".join(goal_parts).strip()

    open_tabs = []
    urls = obs.get("open_pages_urls", [])
    titles = obs.get("open_pages_titles", [])
    active_idx = obs.get("active_page_index", 0)
    for i, (url, title) in enumerate(zip(urls, titles)):
        active = " (active tab)" if i == active_idx else ""
        open_tabs.append(f"Tab {i}{active}\n  Title: {title}\n  URL: {url}")
    tabs_text = (
        "\n".join(open_tabs)
        if open_tabs
        else "Tab 0 (active tab)\n  Title: unknown\n  URL: unknown"
    )

    action_space = DEMO_PROMPT_ACTION_SET.describe(
        with_long_description=False,
        with_examples=True,
    )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"# Goal\n{goal}\n\n"
        f"# Currently open tabs\n{tabs_text}\n\n"
        f"# Current page Accessibility Tree\n{axtree_text}\n\n"
        f"# Current page DOM\n{dom_text}\n\n"
        f"# Action Space\n\n{action_space}\n\n"
        f"# Error message from last action\n\n{obs.get('last_action_error', '')}\n\n"
        "# Next action\n"
        "You will now think step by step and produce your next best action. Reflect on your past actions, "
        "any resulting error message, and the current state of the page before deciding on your next action."
    )


def _run_episode(env, model, seed, max_steps, max_elems, max_new_tokens, steer):
    obs, _ = env.reset(seed=seed)
    outputs = []
    actions = []
    errors = []
    total_reward = 0.0
    success = False

    for _ in range(max_steps):
        prompt = build_prompt(obs, max_elems)
        output = model.generate(prompt, steer=steer, max_new_tokens=max_new_tokens)
        action = str(output or "").strip()

        outputs.append(output)
        actions.append(action)

        try:
            obs, reward, terminated, truncated, _info = env.step(action)
            reward = float(reward)
            done = bool(terminated or truncated)
            error_text = ""
            if isinstance(obs, dict):
                error_text = str(obs.get("last_action_error", "") or "")
        except Exception as exc:
            reward = 0.0
            done = True
            error_text = f"step_exception:{type(exc).__name__}"

        total_reward += reward
        success = success or (reward > 0)
        errors.append(error_text)

        if done:
            break

    last_error = ""
    for err in reversed(errors):
        if err:
            last_error = err
            break

    return {
        "outputs": outputs,
        "actions": actions,
        "steps": len(actions),
        "total_reward": total_reward,
        "success": bool(success),
        "error": last_error,
    }


# =============================================================================
# STEERING VECTOR COMPUTATION
# =============================================================================


def _get_prompt(model, obs, env, max_elems):
    """Build text prompt for steering vector construction."""
    _ = env
    _ = model
    prompt = build_prompt(obs, max_elems)
    return prompt


def _get_hidden_state_offset(model, states):
    """Return offset so hidden_states[offset] maps to block 0."""
    num_layers = len(get_model_layers(model.model, model.model_key))
    num_states = len(states)
    if num_states < num_layers:
        raise ValueError("hidden_states shorter than model layers")
    return num_states - num_layers


def _compute_activation_diff(model, pos, neg, max_new_tokens):
    """Compute activation difference for a contrastive pair across all layers.

    Returns:
        dict: Mapping block_idx -> activation difference (numpy array)
    """
    if model.vector_method == "prompt":
        # Standard CAA: Extract from prompt
        pos_states = model._prompt_activation(pos)
        neg_states = model._prompt_activation(neg)
    else:
        # Non-standard: Extract from generated response
        pos_text = model.generate(
            pos, steer=False, max_new_tokens=max_new_tokens, deterministic=True
        )
        neg_text = model.generate(
            neg, steer=False, max_new_tokens=max_new_tokens, deterministic=True
        )
        pos_states = model._last_token_state(pos_text)
        neg_states = model._last_token_state(neg_text)

    # Align hidden_states indices to transformer block indices
    offset = _get_hidden_state_offset(model, pos_states)
    num_layers = len(pos_states) - offset

    diffs = {}
    for block_idx in range(num_layers):
        state_idx = block_idx + offset
        pos_layer = pos_states[state_idx][0, -1].float().cpu().numpy()
        neg_layer = neg_states[state_idx][0, -1].float().cpu().numpy()
        diffs[block_idx] = pos_layer - neg_layer

    return diffs


def compute_vector(
    model,
    tasks,
    steps,
    max_elems,
    max_new_tokens,
    prompt_type,
    cache_dir="vectors",
    model_alias=None,
    seed=0,
):
    """Compute steering vectors for all layers from contrastive prompts.

    Computes and caches vectors for all layers simultaneously, then loads the target layer.

    Args:
        model: SteeredModel instance
        tasks: List of task names
        steps: Number of training steps
        max_elems: Max DOM elements
        max_new_tokens: Max tokens for generation
        prompt_type: Type of prompt configuration
        cache_dir: Directory to cache vectors
        model_alias: Short model name for cache path
        seed: Random seed for cache path
    """

    rng = random.Random(seed)

    if prompt_type == "combined":
        print(
            "Computing Combined Vector for all layers (format_accuracy + composite_1)..."
        )
        vec_a_sums = {}  # layer_idx -> accumulated vector A
        vec_b_sums = {}  # layer_idx -> accumulated vector B

        pbar = tqdm(total=steps, desc="Computing combined vectors")
        steps_per_task = max(1, steps // len(tasks))

        for task in tasks:
            env = make_miniwob_env(task)
            for _ in range(steps_per_task):
                seed_val = rng.randint(0, 2**31 - 1)
                obs, _ = env.reset(seed=seed_val)

                base_prompt = _get_prompt(model, obs, env, max_elems)

                # Vector A: format_accuracy
                pos_a = f"{base_prompt}\n{PROMPT_CONFIGS['format_accuracy']['pos']}"
                neg_a = f"{base_prompt}\n{PROMPT_CONFIGS['format_accuracy']['neg']}"
                diffs_a = _compute_activation_diff(model, pos_a, neg_a, max_new_tokens)

                # Vector B: composite_1
                pos_b = f"{base_prompt}\n{PROMPT_CONFIGS['composite_1']['pos']}"
                neg_b = f"{base_prompt}\n{PROMPT_CONFIGS['composite_1']['neg']}"
                diffs_b = _compute_activation_diff(model, pos_b, neg_b, max_new_tokens)

                # Accumulate for each layer
                for layer_idx in diffs_a.keys():
                    if layer_idx not in vec_a_sums:
                        vec_a_sums[layer_idx] = diffs_a[layer_idx]
                        vec_b_sums[layer_idx] = diffs_b[layer_idx]
                    else:
                        vec_a_sums[layer_idx] += diffs_a[layer_idx]
                        vec_b_sums[layer_idx] += diffs_b[layer_idx]

                pbar.update(1)
                if pbar.n >= steps:
                    break
            env.close()
            if pbar.n >= steps:
                break
        pbar.close()

        # Normalize and combine for each layer
        all_vectors = {}
        for layer_idx in vec_a_sums.keys():
            vec_a = vec_a_sums[layer_idx] / max(1, pbar.n)
            vec_b = vec_b_sums[layer_idx] / max(1, pbar.n)

            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a > 0:
                vec_a = vec_a / norm_a
            if norm_b > 0:
                vec_b = vec_b / norm_b

            combined = vec_a + vec_b
            norm_c = np.linalg.norm(combined)
            if norm_c > 0:
                combined = combined / norm_c
            all_vectors[layer_idx] = combined

        # Save all vectors to cache
        if cache_dir and model_alias is not None:
            cache_subdir = os.path.join(cache_dir, model_alias, f"seed_{seed}")
            os.makedirs(cache_subdir, exist_ok=True)
            for layer_idx, vec in all_vectors.items():
                cache_path = os.path.join(
                    cache_subdir, f"{prompt_type}_L{layer_idx}.pt"
                )
                torch.save(
                    torch.tensor(vec, dtype=torch.float32, device="cpu"), cache_path
                )
            print(f">>> Saved {len(all_vectors)} vectors to {cache_subdir}")

        # Set all vectors in model
        model.set_vector(all_vectors)
        return

    # Standard single-prompt logic
    totals = {}  # layer_idx -> accumulated difference
    pos_instr = PROMPT_CONFIGS[prompt_type]["pos"]
    neg_instr = PROMPT_CONFIGS[prompt_type]["neg"]

    pbar = tqdm(total=steps, desc="Computing steering vectors for all layers")
    steps_per_task = max(1, steps // len(tasks))

    for task in tasks:
        env = gym.make(f"browsergym/miniwob.{task}")
        for _ in range(steps_per_task):
            seed_val = rng.randint(0, 2**31 - 1)
            obs, _ = env.reset(seed=seed_val)

            base_prompt = _get_prompt(model, obs, env, max_elems)
            pos = f"{base_prompt}\n{pos_instr}"
            neg = f"{base_prompt}\n{neg_instr}"

            # Compute activation difference for all layers
            diffs = _compute_activation_diff(model, pos, neg, max_new_tokens)

            # Accumulate for each layer
            for layer_idx, diff in diffs.items():
                if layer_idx not in totals:
                    totals[layer_idx] = diff
                else:
                    totals[layer_idx] += diff

            pbar.update(1)

            if pbar.n >= steps:
                break
        env.close()
        if pbar.n >= steps:
            break
    pbar.close()

    # Normalize each layer's vector
    all_vectors = {}
    for layer_idx, total in totals.items():
        vec = total / max(1, pbar.n)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        all_vectors[layer_idx] = vec

    # Save all vectors to cache
    if cache_dir and model_alias is not None:
        cache_subdir = os.path.join(cache_dir, model_alias, f"seed_{seed}")
        os.makedirs(cache_subdir, exist_ok=True)
        for layer_idx, vec in all_vectors.items():
            cache_path = os.path.join(cache_subdir, f"{prompt_type}_L{layer_idx}.pt")
            torch.save(torch.tensor(vec, dtype=torch.float32, device="cpu"), cache_path)
        print(f">>> Saved {len(all_vectors)} vectors to {cache_subdir}")

    # Set all vectors in model
    model.set_vector(all_vectors)


# =============================================================================
# EVALUATION
# =============================================================================


def load_base_jsonl(path):
    """Load base results from JSONL and index by (task, seed)."""
    base_records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task = record.get("task")
            seed = record.get("seed")
            if task is None or seed is None:
                continue
            base_records[(task, int(seed))] = record
    return base_records


def evaluate(
    model,
    tasks,
    steps,
    max_elems,
    max_new_tokens,
    out_path,
    base_only=False,
    steer_only=False,
    eval_seed=0,
    base_records=None,
    episode_steps=10,
):
    """Evaluate model on tasks, comparing baseline vs steered."""
    if base_only and steer_only:
        raise ValueError("Cannot set both base_only and steer_only")
    if steer_only and base_records is None:
        raise ValueError("steer_only requires base_records")

    base_hits = 0
    steer_hits = 0
    error_episodes_base = 0
    error_episodes_steer = 0
    total = 0
    base_total = 0

    # Three-episodes-per-task evaluation for fairness and consistency
    steps_per_task = 3
    target_episodes = len(tasks) * steps_per_task
    pbar = tqdm(total=target_episodes, desc="Evaluating")

    seed_rng = random.Random(eval_seed)

    with open(out_path, "w", encoding="utf-8") as f:
        for task in tasks:
            env = make_miniwob_env(task)
            for _ in range(steps_per_task):
                seed = seed_rng.randint(0, 2**31 - 1)
                obs, _ = env.reset(seed=seed)

                record = {"task": task, "seed": seed}
                base_success = None

                if steer_only:
                    assert base_records is not None
                    base_record = base_records.get((task, seed))
                    if base_record is None:
                        raise ValueError(
                            f"Missing baseline record for task={task}, seed={seed} in steer-only mode"
                        )

                    base_action = base_record.get("base_action")
                    base_success = base_record.get("base_success")
                    base_error = str(base_record.get("base_error", "") or "")
                    base_failed = bool(base_error)

                    record.update(
                        {
                            "base_output": base_record.get("base_output"),
                            "base_outputs": base_record.get("base_outputs", []),
                            "base_action": base_action,
                            "base_actions": base_record.get("base_actions", []),
                            "base_steps": base_record.get("base_steps"),
                            "base_total_reward": base_record.get("base_total_reward"),
                            "base_success": base_success,
                            "base_error": base_error,
                            "base_error_episode": bool(base_failed),
                        }
                    )
                    if base_success is not None:
                        base_hits += int(base_success)
                        base_total += 1
                    if base_failed:
                        error_episodes_base += 1
                else:
                    base_episode = _run_episode(
                        env,
                        model,
                        seed,
                        episode_steps,
                        max_elems,
                        max_new_tokens,
                        steer=False,
                    )
                    base_success = base_episode["success"]
                    base_error = base_episode["error"]
                    base_hits += int(base_success)
                    base_total += 1
                    if base_error:
                        error_episodes_base += 1

                    record.update(
                        {
                            "base_output": base_episode["outputs"][-1]
                            if base_episode["outputs"]
                            else "",
                            "base_outputs": base_episode["outputs"],
                            "base_action": base_episode["actions"][-1]
                            if base_episode["actions"]
                            else "",
                            "base_actions": base_episode["actions"],
                            "base_steps": base_episode["steps"],
                            "base_total_reward": base_episode["total_reward"],
                            "base_success": base_success,
                            "base_error": base_error,
                            "base_error_episode": bool(base_error),
                        }
                    )

                if not base_only:
                    steer_episode = _run_episode(
                        env,
                        model,
                        seed,
                        episode_steps,
                        max_elems,
                        max_new_tokens,
                        steer=True,
                    )
                    steer_success = steer_episode["success"]
                    steer_error = steer_episode["error"]
                    steer_hits += int(steer_success)
                    if steer_error:
                        error_episodes_steer += 1

                    record.update(
                        {
                            "steer_output": steer_episode["outputs"][-1]
                            if steer_episode["outputs"]
                            else "",
                            "steer_outputs": steer_episode["outputs"],
                            "steer_action": steer_episode["actions"][-1]
                            if steer_episode["actions"]
                            else "",
                            "steer_actions": steer_episode["actions"],
                            "steer_steps": steer_episode["steps"],
                            "steer_total_reward": steer_episode["total_reward"],
                            "steer_success": steer_success,
                            "steer_error": steer_error,
                            "steer_error_episode": bool(steer_error),
                        }
                    )

                f.write(json.dumps(record) + "\n")
                total += 1
                pbar.update(1)

                if base_only:
                    pbar.set_postfix(acc=f"{base_hits / max(1, base_total):.1%}")
                else:
                    base_acc = base_hits / max(1, base_total)
                    steer_acc = steer_hits / max(1, total)
                    pbar.set_postfix(
                        base=f"{base_acc:.1%}",
                        steer=f"{steer_acc:.1%}",
                        delta=f"{(steer_acc - base_acc):+.1%}",
                    )

                if pbar.n >= target_episodes:
                    break
            env.close()
            if pbar.n >= target_episodes:
                break
    pbar.close()

    base_acc = base_hits / max(1, base_total)
    steer_acc = steer_hits / max(1, total) if not base_only else 0

    return {
        "base_accuracy": base_acc,
        "steer_accuracy": steer_acc,
        "improvement": steer_acc - base_acc,
        "base_parse_fail": error_episodes_base / max(1, base_total),
        "steer_parse_fail": error_episodes_steer / max(1, total)
        if not base_only
        else 0,
        "total_episodes": total,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Web Agent Steering Experiment")
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="0.5b")
    parser.add_argument(
        "--layer", default="auto", help="Intervention layer (int or 'auto')"
    )
    parser.add_argument("--coeff", type=float, default=3.0, help="Steering coefficient")
    parser.add_argument(
        "--prompt-type",
        choices=list(PROMPT_CONFIGS.keys()) + ["combined"],
        default="accuracy",
    )
    parser.add_argument(
        "--vector-method", choices=["response", "prompt"], default="response"
    )
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--episode-steps", type=int, default=10)
    parser.add_argument("--tasks", default="all", help="Task list or 'all'")
    parser.add_argument("--out", default="results.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--base-only", action="store_true", help="Evaluate baseline only"
    )
    parser.add_argument(
        "--steer-only",
        action="store_true",
        help="Evaluate steered only (requires --base-jsonl)",
    )
    parser.add_argument(
        "--base-jsonl",
        default=None,
        help="Path to baseline JSONL for steer-only mode",
    )
    parser.add_argument(
        "--cache-dir", default="vectors", help="Directory to cache steering vectors"
    )
    parser.add_argument(
        "--action-window",
        action="store_true",
        help="Apply steering after prefill (generation window only)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of steering vector",
    )
    args = parser.parse_args()

    # Resolve layer
    layer_idx = get_layer(args.model, args.layer)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # BrowserGym environments are registered automatically when importing browsergym.miniwob
    # No need for gym.register_envs() like with miniwob package

    # Select tasks
    if args.tasks == "all":
        tasks = list_miniwob_tasks()
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    if args.base_only and args.steer_only:
        raise ValueError("Cannot set both --base-only and --steer-only")

    base_records = None
    if args.steer_only:
        if not args.base_jsonl:
            raise ValueError("--steer-only requires --base-jsonl")
        base_records = load_base_jsonl(args.base_jsonl)

    # Initialize model
    print(f"Model: {args.model} ({MODEL_MAP[args.model]})")
    print(f"Layer: {layer_idx} ({'auto' if args.layer == 'auto' else 'manual'})")
    print(f"Coeff: {args.coeff}")
    print(f"Prompt type: {args.prompt_type}, Vector method: {args.vector_method}")
    model = SteeredModel(
        MODEL_MAP[args.model],
        layer_idx=layer_idx,
        coeff=args.coeff,
        vector_method=args.vector_method,
        model_key=args.model,
        steer_action_window=args.action_window,
    )

    # Compute or load steering vector
    if not args.base_only:
        # Construct cache path for the target layer
        cache_subdir = os.path.join(args.cache_dir, args.model, f"seed_{args.seed}")
        cache_path = os.path.join(cache_subdir, f"{args.prompt_type}_L{layer_idx}.pt")

        # Check if target layer vector is cached
        if os.path.exists(cache_path) and not args.force_recompute:
            print(f">>> Loading cached vector from {cache_path}")
            cached_vector = torch.load(cache_path, map_location="cpu")
            model.set_vector(cached_vector, layer_idx=layer_idx)
        else:
            # Cache miss: compute all layers
            print(
                f">>> Computing vectors for all layers (cache {'disabled' if args.force_recompute else 'miss'})"
            )
            compute_vector(
                model,
                tasks,
                args.train_steps,
                80,
                80,
                args.prompt_type,
                cache_dir=args.cache_dir,
                model_alias=args.model,
                seed=args.seed,
            )
            # compute_vector saves all layers and sets model.vectors
            # Verify the target layer was loaded
            if model.vector is None:
                raise RuntimeError(f"Failed to load vector for layer {layer_idx}")

    # Evaluate
    results = evaluate(
        model,
        tasks,
        args.eval_steps,
        80,
        80,
        args.out,
        args.base_only,
        args.steer_only,
        eval_seed=args.seed,
        base_records=base_records,
        episode_steps=args.episode_steps,
    )

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
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
