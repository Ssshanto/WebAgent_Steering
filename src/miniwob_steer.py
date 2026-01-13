"""
Representation Engineering for Web Agents
=========================================
Steering LLM web agents via Contrastive Activation Addition (CAA).

This script implements zero-shot steering for MiniWob++ benchmark tasks.
The hypothesis: steering can improve action-space understanding without
task-specific fine-tuning.

Supports:
- Text-only LLMs (Qwen, Llama, Gemma, Phi, SmolLM)
- Vision-Language Models (Qwen-VL) with Set-of-Marks annotation
"""

import argparse
import io
import json
import os
import random
import re

import gymnasium as gym
import miniwob
import numpy as np
import torch
from miniwob.action import ActionTypes
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_MAP = {
    # Qwen family (original)
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    # Qwen family (extended)
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    # Llama family
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    # Other families
    "gemma-2b": "google/gemma-2-2b-it",
    "phi-3.8b": "microsoft/Phi-3.5-mini-instruct",
    "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # VLM - Use Qwen2-VL-2B (stable, proven)
    "qwen-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
}

# Layer depths for 50% intervention point (mid-layer steering)
LAYER_MAP = {
    "0.5b": 11,        # 24 layers → L11 (46%)
    "3b": 18,          # 36 layers → L18 (50%)
    "qwen-1.5b": 14,   # 28 layers → L14 (50%)
    "llama-1b": 8,     # 16 layers → L8 (50%)
    "llama-3b": 14,    # 28 layers → L14 (50%)
    "gemma-2b": 13,    # 26 layers → L13 (50%)
    "phi-3.8b": 16,    # 32 layers → L16 (50%)
    "smollm-1.7b": 12, # 24 layers → L12 (50%)
    "qwen-vl-2b": 14,  # 28 LLM layers → L14 (50% of LLM backbone)
}

# Models that require VLM mode
VLM_MODELS = {"qwen-vl-2b"}

# =============================================================================
# VLM HELPERS
# =============================================================================

def get_layer(model_key, layer_arg):
    """Get layer index, supporting 'auto' for automatic selection."""
    if layer_arg == "auto":
        return LAYER_MAP.get(model_key, 14)
    return int(layer_arg)


def capture_screenshot(env):
    """Capture screenshot from MiniWob environment."""
    driver = env.unwrapped.instance.driver
    png_bytes = driver.get_screenshot_as_png()
    img = Image.open(io.BytesIO(png_bytes))
    return img


def extract_element_positions(dom_elements):
    """Extract bounding boxes from DOM elements for SoM overlay."""
    positions = []
    for el in dom_elements:
        ref = el.get("ref")
        if ref is None:
            continue
        # MiniWob provides bounding info via JavaScript execution
        # For now, use approximate positions from DOM structure
        # Real implementation would use driver.execute_script to get boundingClientRect
        left = el.get("left", 0)
        top = el.get("top", 0) 
        width = el.get("width", 50)
        height = el.get("height", 20)
        positions.append({
            "ref": ref,
            "x": left,
            "y": top,
            "w": width,
            "h": height,
            "text": (el.get("text") or "")[:20],
        })
    return positions


def annotate_screenshot_with_marks(img, elements):
    """Overlay element IDs on screenshot (Set-of-Marks annotation)."""
    draw = ImageDraw.Draw(img)
    
    # Try to load font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()
    
    for elem in elements:
        x, y, w, h = elem["x"], elem["y"], elem["w"], elem["h"]
        ref = elem["ref"]
        
        # Draw bounding box
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        
        # Draw reference label
        label = f"[{ref}]"
        draw.rectangle([x, y - 15, x + 25, y], fill="red")
        draw.text((x + 2, y - 14), label, fill="white", font=font)
    
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
        
        # Detect stop tokens (Llama models need <|eot_id|> in addition to <|end_of_text|>)
        self.stop_token_ids = [self.tokenizer.eos_token_id]
        try:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
                self.stop_token_ids.append(eot_id)
        except Exception:
            pass
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.layer_idx = layer_idx
        self.coeff = coeff
        self.steer_all_layers = steer_all_layers
        self.vector_method = vector_method
        self.vector = None  # Active steering vector for current layer
        self.vectors = {}   # Dictionary mapping layer_idx -> vector tensor
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
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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
            self.vectors = {k: torch.tensor(v, dtype=torch.float32, device="cpu") for k, v in vec.items()}
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
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
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
        self.vectors = {}   # Dictionary mapping layer_idx -> vector tensor
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
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
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
            self.vectors = {k: torch.tensor(v, dtype=torch.float32, device="cpu") for k, v in vec.items()}
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
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
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
            pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
        )
        handle.remove()

        generated_tokens = out[0][input_length:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True).strip()

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


def build_vlm_prompt(obs):
    """Build prompt for VLM (image-based) mode."""
    return f"{SYSTEM_PROMPT}\n\nTask: {obs['utterance']}\n\nThe screenshot shows the webpage with elements marked by [ref] numbers.\n\n{ACTION_FORMAT}"

# =============================================================================
# ACTION PARSING
# =============================================================================

def parse_action(text):
    """Parse action(s) from model output. Returns list of actions or None.

    Uses lenient parsing (re.match) to tolerate trailing characters like pipes,
    consistent with standard web agent benchmarks (WebArena, Mind2Web, SeeAct).
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None

    actions = []
    for line in lines:
        # Click - matches from start, tolerates trailing characters
        match = re.match(r"click\s+ref=(\d+)", line, flags=re.IGNORECASE)
        if match:
            actions.append({"action": "CLICK", "ref": int(match.group(1))})
            continue

        # Type - matches from start, tolerates trailing characters
        match = re.match(r'type\s+ref=(\d+)\s+text="(.*?)"', line, flags=re.IGNORECASE)
        if match:
            actions.append({"action": "TYPE", "ref": int(match.group(1)), "text": match.group(2)})
            continue

        # Select - matches from start, tolerates trailing characters
        match = re.match(r'select\s+ref=(\d+)\s+option="(.*?)"', line, flags=re.IGNORECASE)
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

def _get_prompt_and_image(model, obs, env, max_elems):
    """Get prompt and optionally image based on model type."""
    if hasattr(model, 'is_vlm') and model.is_vlm:
        # VLM mode: use screenshot with SoM annotation
        screenshot = capture_screenshot(env)
        elements = extract_element_positions(obs["dom_elements"])
        annotated_image = annotate_screenshot_with_marks(screenshot, elements)
        prompt = build_vlm_prompt(obs)
        return prompt, annotated_image
    else:
        # Text mode: use DOM HTML
        prompt = build_prompt(obs, max_elems)
        return prompt, None


def _compute_activation_diff(model, pos, neg, max_new_tokens, image=None):
    """Compute activation difference for a contrastive pair across all layers.
    
    Returns:
        dict: Mapping layer_idx -> activation difference (numpy array)
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
            pos_text = model.generate(pos, steer=False, max_new_tokens=max_new_tokens, image=image)
            neg_text = model.generate(neg, steer=False, max_new_tokens=max_new_tokens, image=image)
        else:
            pos_text = model.generate(pos, steer=False, max_new_tokens=max_new_tokens)
            neg_text = model.generate(neg, steer=False, max_new_tokens=max_new_tokens)
        
        # For VLM, _last_token_state doesn't need image
        pos_states = model._last_token_state(pos_text)
        neg_states = model._last_token_state(neg_text)
    
    # pos_states and neg_states are tuples of tensors (one per layer)
    # Compute difference for each layer
    diffs = {}
    for layer_idx in range(len(pos_states)):
        pos_layer = pos_states[layer_idx][0, -1].float().cpu().numpy()
        neg_layer = neg_states[layer_idx][0, -1].float().cpu().numpy()
        diffs[layer_idx] = pos_layer - neg_layer
    
    return diffs


def compute_vector(model, tasks, steps, max_elems, max_new_tokens, prompt_type, cache_dir="vectors", model_alias=None, seed=0):
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
    
    if prompt_type == "combined":
        print("Computing Combined Vector for all layers (format_accuracy + composite_1)...")
        vec_a_sums = {}  # layer_idx -> accumulated vector A
        vec_b_sums = {}  # layer_idx -> accumulated vector B
        
        pbar = tqdm(total=steps, desc="Computing combined vectors")
        steps_per_task = max(1, steps // len(tasks))

        for task in tasks:
            env = gym.make(f"miniwob/{task}-v1")
            for _ in range(steps_per_task):
                seed_val = random.randint(0, 2**31 - 1)
                obs, _ = env.reset(seed=seed_val)
                
                # Get prompt and image based on model type
                base_prompt, image = _get_prompt_and_image(model, obs, env, max_elems)
                
                # Vector A: format_accuracy
                pos_a = f"{base_prompt}\n{PROMPT_CONFIGS['format_accuracy']['pos']}"
                neg_a = f"{base_prompt}\n{PROMPT_CONFIGS['format_accuracy']['neg']}"
                diffs_a = _compute_activation_diff(model, pos_a, neg_a, max_new_tokens, image)
                
                # Vector B: composite_1
                pos_b = f"{base_prompt}\n{PROMPT_CONFIGS['composite_1']['pos']}"
                neg_b = f"{base_prompt}\n{PROMPT_CONFIGS['composite_1']['neg']}"
                diffs_b = _compute_activation_diff(model, pos_b, neg_b, max_new_tokens, image)
                
                # Accumulate for each layer
                for layer_idx in diffs_a.keys():
                    if layer_idx not in vec_a_sums:
                        vec_a_sums[layer_idx] = diffs_a[layer_idx]
                        vec_b_sums[layer_idx] = diffs_b[layer_idx]
                    else:
                        vec_a_sums[layer_idx] += diffs_a[layer_idx]
                        vec_b_sums[layer_idx] += diffs_b[layer_idx]
                
                pbar.update(1)
                if pbar.n >= steps: break
            env.close()
            if pbar.n >= steps: break
        pbar.close()
        
        # Normalize and combine for each layer
        all_vectors = {}
        for layer_idx in vec_a_sums.keys():
            vec_a = vec_a_sums[layer_idx] / max(1, pbar.n)
            vec_b = vec_b_sums[layer_idx] / max(1, pbar.n)
            
            vec_a = vec_a / np.linalg.norm(vec_a)
            vec_b = vec_b / np.linalg.norm(vec_b)
            
            combined = vec_a + vec_b
            combined = combined / np.linalg.norm(combined)
            all_vectors[layer_idx] = combined
        
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
        return

    # Standard single-prompt logic
    totals = {}  # layer_idx -> accumulated difference
    pos_instr = PROMPT_CONFIGS[prompt_type]["pos"]
    neg_instr = PROMPT_CONFIGS[prompt_type]["neg"]
    
    pbar = tqdm(total=steps, desc="Computing steering vectors for all layers")
    steps_per_task = max(1, steps // len(tasks))

    for task in tasks:
        env = gym.make(f"miniwob/{task}-v1")
        for _ in range(steps_per_task):
            seed_val = random.randint(0, 2**31 - 1)
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
                
                # Get prompt and image based on model type
                prompt, image = _get_prompt_and_image(model, obs, env, max_elems)

                # Baseline
                if image is not None:
                    base_out = model.generate(prompt, steer=False, max_new_tokens=max_new_tokens, image=image)
                else:
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
                    # Steered - reset and get prompt/image again
                    obs, _ = env.reset(seed=seed)
                    prompt, image = _get_prompt_and_image(model, obs, env, max_elems)
                    
                    if image is not None:
                        steer_out = model.generate(prompt, steer=True, max_new_tokens=max_new_tokens, image=image)
                    else:
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
    # Monkeypatch MiniWoB to use the correct Chrome binary
    from miniwob.selenium_instance import SeleniumInstance
    from selenium import webdriver
    
    def patched_create_driver(self):
        assert not hasattr(self, "driver"), f"Instance {self.index} already has a driver"
        options = webdriver.ChromeOptions()
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

    parser = argparse.ArgumentParser(description="Web Agent Steering Experiment")
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="0.5b")
    parser.add_argument("--layer", default="auto", help="Intervention layer (int or 'auto')")
    parser.add_argument("--coeff", type=float, default=3.0, help="Steering coefficient")
    parser.add_argument("--prompt-type", choices=list(PROMPT_CONFIGS.keys()) + ["combined"], default="accuracy")
    parser.add_argument("--vector-method", choices=["response", "prompt"], default="response")
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--tasks", default="all", help="Task list or 'all'")
    parser.add_argument("--out", default="results.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-only", action="store_true", help="Evaluate baseline only")
    parser.add_argument("--vlm", action="store_true", help="Enable VLM mode (screenshot + SoM)")
    parser.add_argument("--cache-dir", default="vectors", help="Directory to cache steering vectors")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation of steering vector")
    args = parser.parse_args()

    # Resolve layer
    layer_idx = get_layer(args.model, args.layer)
    
    # Check VLM mode
    is_vlm = args.vlm or args.model in VLM_MODELS

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
            print(f">>> Computing vectors for all layers (cache {'disabled' if args.force_recompute else 'miss'})")
            compute_vector(
                model, tasks, args.train_steps, 80, 80, args.prompt_type,
                cache_dir=args.cache_dir,
                model_alias=args.model,
                seed=args.seed
            )
            # compute_vector saves all layers and sets model.vectors
            # Verify the target layer was loaded
            if model.vector is None:
                raise RuntimeError(f"Failed to load vector for layer {layer_idx}")

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