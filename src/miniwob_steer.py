"""
Representation Engineering for Web Agents
=========================================
Steering LLM web agents via Contrastive Activation Addition (CAA).

This script implements zero-shot steering for MiniWob++ benchmark tasks.
The hypothesis: steering can improve action-space understanding without
task-specific fine-tuning.

Supports:
- Text-only LLMs (Qwen, Llama, Gemma, Phi, SmolLM)
- Vision-Language Models (Qwen-VL)
"""

import argparse
import json
import os
import random
import re

import gymnasium as gym
import browsergym.miniwob
from browsergym.core.action.highlevel import HighLevelActionSet
import numpy as np
import torch
from browsergym.utils.obs import flatten_dom_to_str, flatten_axtree_to_str
from PIL import Image
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
    # VLM - Use Qwen2-VL-2B (stable, proven)
    "qwen-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
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
    "qwen-vl-2b": 14,  # 28 LLM layers → L14 (50% of LLM backbone)
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
    # VLM
    "qwen-vl-2b": "qwen-vl",
}

# Models that don't support chat templates (need raw prompting)
NO_CHAT_TEMPLATE = {"opt-iml-1.3b"}

# Models that require VLM mode
VLM_MODELS = {"qwen-vl-2b"}


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


# =============================================================================
# VLM HELPERS
# =============================================================================


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


def get_screenshot_from_obs(obs):
    """Get screenshot from BrowserGym observation.

    BrowserGym provides screenshots directly in obs["screenshot"] as numpy array.
    """
    screenshot_array = obs.get("screenshot")
    if screenshot_array is None:
        return None

    # Convert numpy array to PIL Image
    img = Image.fromarray(screenshot_array)
    return img


def load_vlm(model_id):
    """Load VLM model and processor."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading VLM: {model_id}")
    print(f"  Device: {device}, dtype: {dtype}")

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,  # Required for Qwen models
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        print(f"✓ VLM loaded successfully")
        return model, processor

    except Exception as e:
        print(f"✗ Failed to load VLM: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check model exists: https://huggingface.co/{model_id}")
        print(f"  2. Update transformers: pip install --upgrade transformers")
        print(f"  3. Try different model: Qwen/Qwen2-VL-7B-Instruct")
        raise


# =============================================================================
# TASK CONFIGURATION
# =============================================================================


# Tasks that are visually/spatially grounded and unfair for text-only agents.
# Keep this list explicit for peer-review defensibility.
EXCLUDED_TASKS = {
    # Geometry / spatial perception
    "bisect-angle",
    "circle-center",
    "draw-circle",
    "draw-line",
    "right-angle",
    # Visual attributes (color/shades/shape)
    "click-color",
    "click-shades",
    "click-shape",
    "count-shape",
    "count-sides",
    "identify-shape",
    "visual-addition",
    # Pie/segment selection
    "click-pie",
    "click-pie-nodelay",
    # Drag-and-drop (spatial layout dependent)
    "drag-box",
    "drag-circle",
    "drag-cube",
    "drag-items",
    "drag-items-grid",
    "drag-shapes",
    "drag-shapes-2",
    "drag-single-shape",
    "drag-sort-numbers",
}


def list_miniwob_tasks():
    """Return the full MiniWob++ task list from the Gym registry."""
    env_ids = [
        env_id
        for env_id in gym.envs.registry.keys()
        if env_id.startswith("browsergym/miniwob.")
    ]
    tasks = [env_id.split("browsergym/miniwob.", 1)[1] for env_id in env_ids]
    tasks = [t for t in tasks if t not in EXCLUDED_TASKS]
    return sorted(tasks)


def make_miniwob_env(task):
    """Create MiniWob env with BrowserGym-native action mapping."""
    action_set = HighLevelActionSet(
        subsets=["miniwob_all"],
        strict=False,
        multiaction=False,
        demo_mode="off",
    )
    return gym.make(
        f"browsergym/miniwob.{task}",
        action_mapping=action_set.to_python_code,
    )


# =============================================================================
# SYSTEM PROMPT (Unified, Format-Neutral)
# =============================================================================

SYSTEM_PROMPT = (
    "You are a web automation agent. Execute the task by outputting action commands.\n"
    "Rules:\n"
    "- Output only action commands, one per line\n"
    "- No explanations, no reasoning, no markdown\n"
    "- Format: one action per line as described below"
)

ACTION_FORMAT = (
    'Available actions:\n- click("<bid>")\n- fill("<bid>", "<text>")\n- noop()'
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
        self.is_vlm = False

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

    def generate(self, prompt, steer=False, max_new_tokens=80):
        """Generate text, optionally with steering."""
        if self.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = _apply_template(self.tokenizer, messages, self.model_key)
        else:
            # Models without chat template (e.g., OPT)
            formatted_prompt = f"User: {prompt}\nAssistant:"

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
                    "do_sample": True,
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


class SteeredVLM:
    """Vision-Language Model with activation steering on LLM backbone."""

    def __init__(self, model_name, layer_idx, coeff, vector_method="response"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = load_vlm(model_name)

        # Detect stop tokens (Llama models need <|eot_id|> in addition to <|end_of_text|>)
        self.stop_token_ids = [self.processor.tokenizer.eos_token_id]
        try:
            eot_id = self.processor.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != self.processor.tokenizer.unk_token_id:
                self.stop_token_ids.append(eot_id)
        except Exception:
            pass

        self.layer_idx = layer_idx
        self.coeff = coeff
        self.vector_method = vector_method
        self.vector = None  # Active steering vector for current layer
        self.vectors = {}  # Dictionary mapping layer_idx -> vector tensor
        self._vector_cache = {}
        self.is_vlm = True

    def _get_llm_layers(self):
        """Get LLM backbone layers for steering (skip ViT encoder)."""
        # Qwen2-VL architecture: model.model.language_model.layers contains LLM layers
        return self.model.model.language_model.layers

    def _last_token_state(self, text):
        """Extract activation from last token of text for all layers."""
        inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # Return all hidden states (tuple of tensors, one per layer)
        return out.hidden_states

    def _prompt_activation(self, prompt, image=None):
        """Extract activation from multimodal prompt before generation for all layers."""
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if image is not None:
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        else:
            inputs = self.processor(text=[text], return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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

    def generate(self, prompt, steer=False, max_new_tokens=80, image=None):
        """Generate text from multimodal input, optionally with steering."""
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if image is not None:
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        else:
            inputs = self.processor(text=[text], return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

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

        # Hook into LLM backbone layers only (not ViT)
        llm_layers = self._get_llm_layers()
        handle = llm_layers[self.layer_idx].register_forward_hook(hook)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        )
        handle.remove()

        generated_tokens = out[0][input_length:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True).strip()


# =============================================================================
# DOM PROCESSING
# =============================================================================


def build_prompt(obs, max_elems=80):
    """Build prompt from BrowserGym observation."""
    _ = max_elems
    dom_text = flatten_dom_to_str(obs["dom_object"])
    axtree_text = flatten_axtree_to_str(obs["axtree_object"])
    return (
        f"{SYSTEM_PROMPT}\n\nTask: {obs['goal']}\n\n"
        f"DOM:\n{dom_text}\n\n"
        f"AXTree:\n{axtree_text}\n\n"
        f"{ACTION_FORMAT}"
    )


def _normalize_action_text(text):
    if not text:
        return "noop()"
    action = text.strip()
    if not action:
        return "noop()"

    action = action.splitlines()[0].strip()
    action = re.sub(r"(?m)^\s*-\s+", "", action)
    action = re.sub(r"click\(\s*(\d+)\s*\)", r'click("\1")', action)
    action = re.sub(r"fill\(\s*(\d+)\s*,", r'fill("\1",', action)
    action = re.sub(r"select\(\s*(\d+)\s*,", r'select("\1",', action)
    return action


def _single_step(env, action_text):
    try:
        obs, reward, terminated, truncated, _info = env.step(action_text)
        error_text = ""
        if isinstance(obs, dict):
            error_text = str(obs.get("last_action_error", "") or "")
        return (
            float(reward),
            bool(terminated or truncated),
            bool(error_text),
            error_text,
        )
    except Exception:
        return 0.0, True, True, "step_exception"


def build_vlm_prompt(obs):
    """Build prompt for VLM (image-based) mode."""
    return (
        f"{SYSTEM_PROMPT}\n\nTask: {obs['goal']}\n\n"
        "Use the screenshot to identify the correct target element by its bid and output one valid action.\n\n"
        f"{ACTION_FORMAT}"
    )


# =============================================================================
# STEERING VECTOR COMPUTATION
# =============================================================================


def _get_prompt_and_image(model, obs, env, max_elems):
    """Get prompt and optionally image based on model type."""
    _ = env
    if hasattr(model, "is_vlm") and model.is_vlm:
        # VLM mode: use screenshot from BrowserGym observation
        screenshot = get_screenshot_from_obs(obs)
        prompt = build_vlm_prompt(obs)
        return prompt, screenshot
    else:
        # Text mode: use BrowserGym observation flatteners
        prompt = build_prompt(obs, max_elems)
        return prompt, None


def _get_hidden_state_offset(model, states):
    """Return offset so hidden_states[offset] maps to block 0."""
    if model.is_vlm:
        num_layers = len(model._get_llm_layers())
    else:
        num_layers = len(get_model_layers(model.model, model.model_key))
    num_states = len(states)
    if num_states < num_layers:
        raise ValueError("hidden_states shorter than model layers")
    return num_states - num_layers


def _compute_activation_diff(model, pos, neg, max_new_tokens, image=None):
    """Compute activation difference for a contrastive pair across all layers.

    Returns:
        dict: Mapping block_idx -> activation difference (numpy array)
    """
    if model.vector_method == "prompt":
        # Standard CAA: Extract from prompt
        if image is not None:
            pos_states = model._prompt_activation(pos, image=image)
            neg_states = model._prompt_activation(neg, image=image)
        else:
            pos_states = model._prompt_activation(pos)
            neg_states = model._prompt_activation(neg)
    else:
        # Non-standard: Extract from generated response
        if image is not None:
            pos_text = model.generate(
                pos, steer=False, max_new_tokens=max_new_tokens, image=image
            )
            neg_text = model.generate(
                neg, steer=False, max_new_tokens=max_new_tokens, image=image
            )
        else:
            pos_text = model.generate(pos, steer=False, max_new_tokens=max_new_tokens)
            neg_text = model.generate(neg, steer=False, max_new_tokens=max_new_tokens)

        # For VLM, _last_token_state doesn't need image
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
        model: SteeredModel or SteeredVLM instance
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

                # Get prompt and image based on model type
                base_prompt, image = _get_prompt_and_image(model, obs, env, max_elems)

                # Vector A: format_accuracy
                pos_a = f"{base_prompt}\n{PROMPT_CONFIGS['format_accuracy']['pos']}"
                neg_a = f"{base_prompt}\n{PROMPT_CONFIGS['format_accuracy']['neg']}"
                diffs_a = _compute_activation_diff(
                    model, pos_a, neg_a, max_new_tokens, image
                )

                # Vector B: composite_1
                pos_b = f"{base_prompt}\n{PROMPT_CONFIGS['composite_1']['pos']}"
                neg_b = f"{base_prompt}\n{PROMPT_CONFIGS['composite_1']['neg']}"
                diffs_b = _compute_activation_diff(
                    model, pos_b, neg_b, max_new_tokens, image
                )

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

            # Get prompt and image based on model type
            base_prompt, image = _get_prompt_and_image(model, obs, env, max_elems)
            pos = f"{base_prompt}\n{pos_instr}"
            neg = f"{base_prompt}\n{neg_instr}"

            # Compute activation difference for all layers
            diffs = _compute_activation_diff(model, pos, neg, max_new_tokens, image)

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
):
    """Evaluate model on tasks, comparing baseline vs steered."""
    if base_only and steer_only:
        raise ValueError("Cannot set both base_only and steer_only")
    if steer_only and base_records is None:
        raise ValueError("steer_only requires base_records")

    base_hits = 0
    steer_hits = 0
    parse_fails_base = 0
    parse_fails_steer = 0
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

                # Get prompt and image based on model type
                prompt, image = _get_prompt_and_image(model, obs, env, max_elems)

                record = {"task": task, "seed": seed}
                base_success = None

                if steer_only:
                    assert base_records is not None
                    base_record = base_records.get((task, seed))
                    if base_record is not None:
                        assert base_record is not None
                        base_action = base_record.get("base_action")
                        base_success = base_record.get("base_success")
                        record.update(
                            {
                                "base_output": base_record.get("base_output"),
                                "base_action": base_action,
                                "base_success": base_success,
                            }
                        )
                        if base_success is not None:
                            base_hits += int(base_success)
                            base_total += 1
                        if base_action is None:
                            parse_fails_base += 1
                else:
                    # Baseline
                    if image is not None:
                        base_out = model.generate(
                            prompt,
                            steer=False,
                            max_new_tokens=max_new_tokens,
                            image=image,
                        )
                    else:
                        base_out = model.generate(
                            prompt, steer=False, max_new_tokens=max_new_tokens
                        )

                    base_action = _normalize_action_text(base_out)
                    base_reward, _, base_failed, base_error = _single_step(
                        env, base_action
                    )
                    base_success = base_reward > 0
                    base_hits += int(base_success)
                    base_total += 1
                    if base_failed:
                        parse_fails_base += 1

                    record.update(
                        {
                            "base_output": base_out,
                            "base_action": base_action,
                            "base_success": base_success,
                            "base_error": base_error,
                        }
                    )

                if not base_only:
                    # Steered - reset and get prompt/image again
                    obs, _ = env.reset(seed=seed)
                    prompt, image = _get_prompt_and_image(model, obs, env, max_elems)

                    if image is not None:
                        steer_out = model.generate(
                            prompt,
                            steer=True,
                            max_new_tokens=max_new_tokens,
                            image=image,
                        )
                    else:
                        steer_out = model.generate(
                            prompt, steer=True, max_new_tokens=max_new_tokens
                        )

                    steer_action = _normalize_action_text(steer_out)
                    steer_reward, _, steer_failed, steer_error = _single_step(
                        env, steer_action
                    )
                    steer_success = steer_reward > 0
                    steer_hits += int(steer_success)
                    if steer_failed:
                        parse_fails_steer += 1

                    record.update(
                        {
                            "steer_output": steer_out,
                            "steer_action": steer_action,
                            "steer_success": steer_success,
                            "steer_error": steer_error,
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
        "base_parse_fail": parse_fails_base / max(1, base_total),
        "steer_parse_fail": parse_fails_steer / max(1, total) if not base_only else 0,
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
        "--vlm", action="store_true", help="Enable VLM mode (screenshot + SoM)"
    )
    parser.add_argument(
        "--cache-dir", default="vectors", help="Directory to cache steering vectors"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of steering vector",
    )
    args = parser.parse_args()

    # Resolve layer
    layer_idx = get_layer(args.model, args.layer)

    # Check VLM mode
    is_vlm = args.vlm or args.model in VLM_MODELS

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
    print(f"VLM mode: {is_vlm}")

    if is_vlm:
        model = SteeredVLM(
            MODEL_MAP[args.model],
            layer_idx=layer_idx,
            coeff=args.coeff,
            vector_method=args.vector_method,
        )
    else:
        model = SteeredModel(
            MODEL_MAP[args.model],
            layer_idx=layer_idx,
            coeff=args.coeff,
            vector_method=args.vector_method,
            model_key=args.model,
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
