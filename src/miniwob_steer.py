"""CAA steering for zero-shot BrowserGym MiniWob++ agents."""

import argparse
import importlib
import json
import os
import random
import re
import time
from pathlib import Path

import gymnasium as gym
import browsergym.miniwob
import browsergym.core.chat as browsergym_chat
from browsergym.core.action.highlevel import HighLevelActionSet
import numpy as np
import torch
from browsergym.utils.obs import flatten_dom_to_str, flatten_axtree_to_str, prune_html
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
)
from interface_variants import (
    INTERFACE_MODES,
    action_metrics as interface_action_metrics,
    apply_interface_variant,
    executable_action_from_shown,
    interface_cache_tag,
    parse_interface_modes,
)

MODEL_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-coder-0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen3-0.5b": "Qwen/Qwen3-0.6B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma-2b": "google/gemma-2-2b-it",
    "gemma-3-270m": "google/gemma-3-270m-it",
    "gemma-1b": "google/gemma-3-1b-it",
    "gemma-3-1b": "google/gemma-3-1b-it",
    "gemma-3-4b": "google/gemma-3-4b-it",
    "phi-3.8b": "microsoft/Phi-3.5-mini-instruct",
    "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "stablelm-1.6b": "stabilityai/stablelm-2-1_6b-chat",
    "opt-iml-1.3b": "facebook/opt-iml-1.3b",
}

LAYER_MAP = {
    "0.5b": 11,  # 24 layers → L11 (46%)
    "3b": 18,  # 36 layers → L18 (50%)
    "qwen-7b": 16,  # 32 layers → L16 (50%)
    "qwen-1.5b": 14,  # 28 layers → L14 (50%)
    "qwen-coder-0.5b": 11,  # 24 layers → L11 (46%)
    "qwen3-0.5b": 13,  # 28 layers → L13 (46%)
    "qwen3-0.6b": 13,  # 28 layers → L13 (46%)
    "qwen3-1.7b": 13,  # 28 layers → L13 (46%)
    "qwen3-4b": 18,  # 36 layers → L18 (50%)
    "qwen3-8b": 18,  # 36 layers → L18 (50%)
    "llama-1b": 8,  # 16 layers → L8 (50%)
    "llama-3b": 14,  # 28 layers → L14 (50%)
    "gemma-2b": 13,  # 26 layers → L13 (50%)
    "gemma-3-270m": 9,  # small text model; use mid-depth layer
    "gemma-1b": 13,  # 26 layers → L13 (50%)
    "gemma-3-1b": 13,  # 26 layers → L13 (50%)
    "gemma-3-4b": 17,  # 34 layers → L17 (50%)
    "phi-3.8b": 16,  # 32 layers → L16 (50%)
    "smollm-1.7b": 12,  # 24 layers → L12 (50%)
    "smollm-360m": 16,  # 32 layers → L16 (50%)
    "tinyllama-1.1b": 11,  # 22 layers → L11 (50%)
    "stablelm-1.6b": 12,  # 24 layers → L12 (50%)
    "opt-iml-1.3b": 12,  # 24 layers → L12 (50%)
}

MODEL_ARCH = {
    "0.5b": "qwen",
    "3b": "qwen",
    "qwen-7b": "qwen",
    "qwen-1.5b": "qwen",
    "qwen-coder-0.5b": "qwen",
    "qwen3-0.5b": "qwen",
    "qwen3-0.6b": "qwen",
    "qwen3-1.7b": "qwen",
    "qwen3-4b": "qwen",
    "qwen3-8b": "qwen",
    "llama-1b": "llama",
    "llama-3b": "llama",
    "gemma-2b": "gemma",
    "gemma-3-270m": "gemma",
    "gemma-1b": "gemma",
    "gemma-3-1b": "gemma",
    "gemma-3-4b": "gemma3_conditional",
    "phi-3.8b": "phi",
    "smollm-1.7b": "qwen",
    "smollm-360m": "qwen",
    "tinyllama-1.1b": "llama",
    "stablelm-1.6b": "qwen",
    "opt-iml-1.3b": "opt",
}

NO_CHAT_TEMPLATE = {"opt-iml-1.3b"}
SEED_MODES = {"project", "browsergym"}
PROJECT_DEFAULT_SEED = 0
BROWSERGYM_BENCHMARK_SEED = 42
BROWSERGYM_BENCHMARK_REPEATS = 5
# Mirror the current BrowserGym benchmark helper exactly.
BROWSERGYM_BENCHMARK_SEED_MAX = 2**32
_PATCHED_CHAT_EXECUTABLE = None
DATASET_NAMES = ("miniwob", "webarena", "workarena")
DATASET_IMPORTS = {
    "miniwob": "browsergym.miniwob",
    "webarena": "browsergym.webarena",
    "workarena": "browsergym.workarena",
}
DATASET_ENV_PREFIXES = {
    "miniwob": "browsergym/miniwob.",
    "webarena": "browsergym/webarena.",
    "workarena": "browsergym/workarena.",
}
DATASET_ACTION_SUBSETS = {
    "miniwob": ["miniwob_all"],
    "webarena": ["webarena"],
    "workarena": ["workarena"],
}
DEFAULT_VECTOR_TASKS = {
    "miniwob": ["click-button", "click-link", "click-option", "choose-list", "focus-text"],
    "webarena": ["10"],
    "workarena": ["servicenow.filter-asset-list"],
}
ACTION_RE = re.compile(r"^\s*([A-Za-z_]\w*)\((.*)\)\s*$", re.S)
QUOTED_RE = re.compile(r"""['"]([^'"]+)['"]""")
ID_ARGUMENT_ACTIONS = {
    "click",
    "dblclick",
    "hover",
    "fill",
    "select_option",
    "check",
    "uncheck",
    "focus",
    "clear",
    "drag_and_drop",
    "upload_file",
    "press",
}


def _apply_template(tokenizer, messages, model_key):
    if model_key and model_key.startswith("qwen3-"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _gemma3_messages(prompt):
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def get_layer(model_key, layer_arg):
    if layer_arg == "auto":
        return LAYER_MAP[model_key]
    return int(layer_arg)


def get_model_layers(model, model_key):
    arch = MODEL_ARCH[model_key]

    if arch in ("qwen", "llama", "gemma", "phi"):
        return model.model.layers
    if arch == "gemma3_conditional":
        return model.model.language_model.layers
    if arch == "smollm":
        return model.transformer.h
    if arch == "stablelm":
        return model.gpt_neox.layers
    if arch == "opt":
        return model.model.decoder.layers
    raise ValueError(f"Unknown model architecture: {arch}")


def get_additional_stop_tokens(tokenizer, model_key):
    stop_tokens = []
    arch = MODEL_ARCH[model_key]

    if arch == "llama":
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != tokenizer.unk_token_id:
            stop_tokens.append(eot_id)
    elif arch in ("gemma", "gemma3_conditional"):
        end_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if end_turn is not None and end_turn != tokenizer.unk_token_id:
            stop_tokens.append(end_turn)
    elif arch == "stablelm":
        for token in ["<|endoftext|>", "<|im_end|>"]:
            tok_id = tokenizer.convert_tokens_to_ids(token)
            if tok_id is not None and tok_id != tokenizer.unk_token_id:
                stop_tokens.append(tok_id)

    return stop_tokens


def _patch_browsergym_chat_browser():
    global _PATCHED_CHAT_EXECUTABLE
    executable_path = os.environ.get("BROWSERGYM_CHROMIUM_EXECUTABLE")
    if not executable_path or _PATCHED_CHAT_EXECUTABLE == executable_path:
        return

    extra_args = ["--no-sandbox", "--disable-dev-shm-usage"]

    def _patched_chat_init(self, headless, chat_size=(500, 800), record_video_dir=None, modern=True):
        self.messages = []
        pw = browsergym_chat._get_global_playwright()
        self.browser = pw.chromium.launch(
            headless=headless,
            args=[f"--window-size={chat_size[0]},{chat_size[1]}", *extra_args],
            executable_path=executable_path,
        )
        self.context = self.browser.new_context(
            no_viewport=True,
            record_video_dir=Path(record_video_dir) / "chat_video" if record_video_dir else None,
            record_video_size=dict(width=chat_size[0], height=chat_size[1]),
        )
        self.page = self.context.new_page()
        self.recording_start_time = time.time() if record_video_dir else None
        self.page.expose_function(
            "send_user_message", lambda msg: self._js_user_message_received_callback(msg=msg)
        )
        if modern:
            self.page.set_content(browsergym_chat.get_chatbox_modern(browsergym_chat.CHATBOX_DIR))
        else:
            self.page.set_content(browsergym_chat.get_chatbox_classic(browsergym_chat.CHATBOX_DIR))

    browsergym_chat.Chat.__init__ = _patched_chat_init
    _PATCHED_CHAT_EXECUTABLE = executable_path


def _normalize_dataset(dataset):
    dataset = (dataset or "miniwob").strip().lower()
    if dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset: {dataset}")
    return dataset


def import_browsergym_dataset(dataset):
    dataset = _normalize_dataset(dataset)
    return importlib.import_module(DATASET_IMPORTS[dataset])


def get_action_set(dataset="miniwob"):
    dataset = _normalize_dataset(dataset)
    return DATASET_ACTION_SETS[dataset]


def _env_id_for_task(dataset, task):
    dataset = _normalize_dataset(dataset)
    task = str(task).strip()
    if task.startswith("browsergym/"):
        return task
    return f"{DATASET_ENV_PREFIXES[dataset]}{task}"


def list_browsergym_tasks(dataset="miniwob"):
    """Return registered task suffixes for a BrowserGym dataset."""
    dataset = _normalize_dataset(dataset)
    import_browsergym_dataset(dataset)
    prefix = DATASET_ENV_PREFIXES[dataset]
    env_ids = [
        env_id
        for env_id in gym.envs.registry.keys()
        if env_id.startswith(prefix)
    ]
    tasks = [env_id.split(prefix, 1)[1] for env_id in env_ids]
    return sorted(tasks)


def list_miniwob_tasks():
    """Return the full MiniWob++ task list from the Gym registry."""
    return list_browsergym_tasks("miniwob")


def make_browsergym_env(dataset, task):
    """Create a BrowserGym env with the dataset's high-level action mapping."""
    dataset = _normalize_dataset(dataset)
    import_browsergym_dataset(dataset)
    _patch_browsergym_chat_browser()
    pw_chromium_kwargs = {}
    executable_path = os.environ.get("BROWSERGYM_CHROMIUM_EXECUTABLE")
    if executable_path:
        pw_chromium_kwargs["executable_path"] = executable_path
    return gym.make(
        _env_id_for_task(dataset, task),
        action_mapping=get_action_set(dataset).to_python_code,
        pw_chromium_kwargs=pw_chromium_kwargs,
    )


def make_miniwob_env(task):
    """Create MiniWob env with BrowserGym-native action mapping."""
    return make_browsergym_env("miniwob", task)


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
    retry_with_force=True,
)
DATASET_ACTION_SETS = {
    name: HighLevelActionSet(
        subsets=subsets,
        strict=False,
        multiaction=False,
        demo_mode="off",
        retry_with_force=True,
    )
    for name, subsets in DATASET_ACTION_SUBSETS.items()
}
DATASET_ACTION_SETS["miniwob"] = DEMO_PROMPT_ACTION_SET


def parse_tasks_arg(tasks_arg, dataset="miniwob"):
    if tasks_arg == "all":
        return list_browsergym_tasks(dataset)
    return [task.strip() for task in tasks_arg.split(",") if task.strip()]


def _make_seed_sequence(count, seed, seed_mode):
    if seed_mode not in SEED_MODES:
        raise ValueError(f"Unknown seed mode: {seed_mode}")
    if count < 0:
        raise ValueError("count must be non-negative")

    if seed_mode == "browsergym":
        rng = np.random.RandomState(seed)
        return [
            int(x)
            for x in rng.randint(low=0, high=BROWSERGYM_BENCHMARK_SEED_MAX, size=count)
        ]

    rng = random.Random(seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(count)]


def build_episode_plan(tasks, episodes_per_task, seed, seed_mode):
    total = len(tasks) * int(episodes_per_task)
    seeds = _make_seed_sequence(total, seed, seed_mode)
    plan = []
    idx = 0
    for task in tasks:
        for _ in range(int(episodes_per_task)):
            plan.append((task, seeds[idx]))
            idx += 1
    return plan


def parse_plan_arg(plan_arg):
    plan = []
    for item in plan_arg.split(","):
        item = item.strip()
        if not item:
            continue
        task, seed_text = item.rsplit(":", 1)
        plan.append((task, int(seed_text)))
    return plan


def load_plan_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return parse_plan_arg(f.read().strip())


def build_training_plan(tasks, steps, seed, seed_mode):
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not tasks:
        raise ValueError("tasks must be non-empty")

    seeds = _make_seed_sequence(steps, seed, seed_mode)
    return [(tasks[idx % len(tasks)], seed_val) for idx, seed_val in enumerate(seeds)]

PROMPT_CONFIGS = {
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
    "action_syntax": {
        "pos": "Return only one executable BrowserGym action call, such as click('12') or fill('7', 'text'). Use exact syntax. No preface, no explanation, no markdown.",
        "neg": "Do not output an executable action call. Write a natural language explanation, include a preface, or use malformed action syntax.",
    },
    "valid_bid": {
        "pos": "Use an existing bid from the current Accessibility Tree or DOM. Choose the interactable element whose label and role exactly match the goal.",
        "neg": "Use a missing, guessed, or unrelated bid. Ignore whether the element is interactable or matches the goal.",
    },
    "webagent_discipline": {
        "pos": "Act like a careful MiniWob web agent: read the goal, inspect available elements, choose the minimal valid action that advances the task, and output only that action.",
        "neg": "Act like a careless web agent: skim the page, choose a plausible but unchecked action, and include extra text instead of only the action.",
    },
    "dom_bid_not_example": {
        "pos": "Do not copy Action Space examples; their ids are fake. Use only a bid that appears in the current Accessibility Tree or DOM, choose the action that satisfies the goal, and output exactly one action call.",
        "neg": "Copy an Action Space example or use a guessed bid even if it is not present in the current page. Ignore the current Accessibility Tree and DOM.",
    },
    "bid_semantics": {
        "pos": "The bracketed number in the Accessibility Tree is the bid to use. If the tree says [17] button 'no', output click('17'); if it says [15] textbox, use fill('15', value). Never use the visible label, color, or Action Space example id as the bid.",
        "neg": "Misread bids. Use the visible label, color, or an Action Space example id as the bid instead of the bracketed number from the current Accessibility Tree.",
    },
    "numeric_bid_strict": {
        "pos": "BrowserGym actions require the numeric bid from the current Accessibility Tree or DOM. Never use HTML id/name strings such as subbtn, to-copy, math-answer, username, or password as the bid. Output one valid action using the current numeric bid.",
        "neg": "Use HTML id/name/text strings as bids instead of numeric BrowserGym bids. Prefer ids such as subbtn, to-copy, math-answer, username, or password even when numeric bids are shown.",
    },
    "bid_plan": {
        "pos": "First identify the correct current element, then use its numeric bid in the action. For text-entry tasks, fill the numeric textbox bid and then click the numeric submit/button bid if needed. Do not use HTML ids or Action Space example ids.",
        "neg": "Skip identifying current numeric bids. Use stale example ids or HTML ids, and choose the action sequence from memory even if it does not match the current page.",
    },
    "submit_after_fill": {
        "pos": "For form tasks, inspect the current page state. If the required text/value is already present in the textbox, do not fill or type again; click the numeric bid of the submit, OK, done, or go button. Use exactly one action for the current step.",
        "neg": "For form tasks, repeat the same fill or type action even when the textbox already contains the required value. Avoid clicking submit, OK, done, or go buttons.",
    },
    "submit_and_bid": {
        "pos": "Use current numeric BrowserGym bids only. For form tasks, inspect the current page state: fill the required numeric textbox bid if empty, but if the required value is already present, click the numeric bid of the submit, OK, done, or go button. Output exactly one action for this step.",
        "neg": "Use HTML id/name/text strings as bids instead of numeric BrowserGym bids. Repeat fill or type actions even when the textbox already contains the required value, and avoid clicking submit, OK, done, or go buttons.",
    },
    "submit_pos_bid_neg": {
        "pos": "For form tasks, inspect the current page state. If the required text/value is already present in the textbox, do not fill or type again; click the numeric bid of the submit, OK, done, or go button. Use exactly one action for the current step.",
        "neg": "Use HTML id/name/text strings as bids instead of numeric BrowserGym bids. Prefer ids such as subbtn, to-copy, math-answer, username, or password even when numeric bids are shown.",
    },
    "gemma_bid_repair": {
        "pos": "The Action Space examples are syntax examples only; do not copy their bids. Pick a bid that appears in the current Accessibility Tree, and output exactly one executable action call such as click('12') or fill('7', 'text'). No words before or after the action.",
        "neg": "Copy an Action Space example bid such as 48 or a51, or answer with words like Click instead of an executable action.",
    },
    "gemma_tree_action": {
        "pos": "Use the current Accessibility Tree as the source of truth. The bid is the bracketed id before each element. Choose the element matching the goal and output only one action using that exact bid.",
        "neg": "Ignore the current Accessibility Tree. Guess an id from the examples or describe the action in natural language.",
    },
    "gemma_tree_vs_label": {
        "pos": "Use the current Accessibility Tree as the source of truth. The bid is the bracketed id before each element. Choose the element matching the goal and output only one action using that exact bid.",
        "neg": "Misread bids. Use the visible label, color, or an Action Space example id as the bid instead of the bracketed number from the current Accessibility Tree.",
    },
    "gemma_tree_vs_guess": {
        "pos": "Use the current Accessibility Tree as the source of truth. The bid is the bracketed id before each element. Choose the element matching the goal and output only one action using that exact bid.",
        "neg": "Output exactly one executable action, but do not verify the current Accessibility Tree bid. Guess a plausible bid from nearby elements, examples, or earlier actions even if it may target the wrong element.",
    },
    "gemma_tree_pos_minus_base": {
        "pos": "Use the current Accessibility Tree as the source of truth. The bid is the bracketed id before each element. Choose the element matching the goal and output only one action using that exact bid.",
        "neg": "",
    },
    "action_space_pos_minus_base": {
        "pos": "Read the current state and the listed action space carefully. Choose exactly one allowed action. Fill every action argument using only identifiers, handles, or values present in the current state. Output only that executable action, with no explanation.",
        "neg": "",
    },
    "action_space_pos_minus_neg": {
        "pos": "Read the current state and the listed action space carefully. Choose exactly one allowed action. Fill every action argument using only identifiers, handles, or values present in the current state. Output only that executable action, with no explanation.",
        "neg": "Ignore the listed action space and current state. Guess an action name or arguments from memory, examples, visible labels, or prior tasks, even if they are not valid in the current state.",
    },
    "current_state_action_guard": {
        "pos": "Action Space examples are syntax examples only; their bids and values are fake. Use only ids, bids, handles, URLs, options, and values that appear in the current state. For information-seeking tasks, interact with the current page until the answer is known, then call send_msg_to_user(answer). Output exactly one executable action and no explanation.",
        "neg": "",
    },
    "interface_current_id_binding": {
        "pos": "Bind actions to the current interface only. Use an identifier that appears on the current page now, not a label, example id, stale id, previous id, or guessed id. Output exactly one executable action.",
        "neg": "Do not bind actions to the current interface. Copy a label, example id, stale id, previous id, or guessed id even if it is not present on the current page now.",
    },
    "interface_action_type_binding": {
        "pos": "Choose an action type allowed for the current target element. Click buttons and links, fill textboxes, select listed options, and check or uncheck checkboxes only when that action matches the element. Output exactly one executable action.",
        "neg": "Use a mismatched action type for the target element, such as filling a button, clicking a textbox when text is required, selecting from a non-select element, or checking a non-checkbox.",
    },
    "interface_argument_binding": {
        "pos": "Fill action arguments using only current page values, identifiers, handles, options, and requested text. Do not invent ids, handles, options, or values. Output exactly one executable action.",
        "neg": "Invent action arguments from memory or examples. Use ids, handles, options, or values that are not present in the current page or requested by the goal.",
    },
}


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
        steer_position="last",
    ):
        if steer_position not in {"last", "all"}:
            raise ValueError("steer_position must be 'last' or 'all'")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_key = model_key
        self.processor = None
        arch = MODEL_ARCH.get(model_key)

        if arch == "gemma3_conditional":
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

        self.stop_token_ids = [self.tokenizer.eos_token_id]
        self.stop_token_ids.extend(
            get_additional_stop_tokens(self.tokenizer, model_key)
        )
        self.stop_token_ids = list(set(t for t in self.stop_token_ids if t is not None))

        self.use_chat_template = model_key not in NO_CHAT_TEMPLATE

        if self.device == "cuda" and arch == "gemma3_conditional":
            dtype = torch.bfloat16
        elif self.device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
        model_cls = (
            Gemma3ForConditionalGeneration
            if arch == "gemma3_conditional"
            else AutoModelForCausalLM
        )
        load_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        self.model = model_cls.from_pretrained(model_name, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.layer_idx = layer_idx
        self.coeff = coeff
        self.steer_all_layers = steer_all_layers
        self.vector_method = vector_method
        self.vector = None
        self.vectors = {}
        self._vector_cache = {}
        self.steer_action_window = steer_action_window
        self.steer_position = steer_position

    def _format_prompt(self, prompt):
        if self.processor is not None:
            return self.processor.apply_chat_template(
                _gemma3_messages(prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
        if self.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            return _apply_template(self.tokenizer, messages, self.model_key)
        return f"User: {prompt}\nAssistant:"

    def _tokenize_prompt(self, prompt):
        if self.processor is not None:
            inputs = self.processor.apply_chat_template(
                _gemma3_messages(prompt),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        formatted_prompt = self._format_prompt(prompt)
        return self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

    def _last_token_state(self, text):
        """Extract activation from last token of text for all layers."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states

    def _prompt_activation(self, prompt):
        """Extract activation from prompt before generation for all layers."""
        inputs = self._tokenize_prompt(prompt)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states

    def _response_activation(self, prompt, response):
        """Extract activation from the generated response in full prompt context."""
        formatted_prompt = self._format_prompt(prompt)
        full_text = formatted_prompt + (response or "")
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states

    def set_vector(self, vec, layer_idx=None):
        if isinstance(vec, dict):
            self.vectors = {
                k: torch.tensor(v, dtype=torch.float32, device="cpu")
                for k, v in vec.items()
            }
            if self.layer_idx in self.vectors:
                self.vector = self.vectors[self.layer_idx]
        else:
            target_layer = layer_idx if layer_idx is not None else self.layer_idx
            tensor_vec = torch.as_tensor(vec, dtype=torch.float32, device="cpu").clone()
            self.vectors[target_layer] = tensor_vec
            if target_layer == self.layer_idx:
                self.vector = tensor_vec
        self._vector_cache.clear()

    def generate(self, prompt, steer=False, max_new_tokens=80, deterministic=False):
        """Generate text, optionally with steering."""
        inputs = self._tokenize_prompt(prompt)
        input_length = inputs["input_ids"].shape[1]
        hook_calls = {}

        def hook_for_layer(layer_idx):
            def hook(_module, _input, output):
                vec_cpu = self.vectors.get(layer_idx) if self.steer_all_layers else self.vector
                if not steer or vec_cpu is None:
                    return output
                hook_calls[layer_idx] = hook_calls.get(layer_idx, 0) + 1
                if self.steer_action_window and hook_calls[layer_idx] == 1:
                    return output
                if torch.is_tensor(output):
                    target = output
                elif isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                    target = output[0]
                else:
                    return output

                cache_key = (target.device, layer_idx)
                vec = self._vector_cache.get(cache_key)
                if vec is None:
                    vec = vec_cpu.to(device=target.device, dtype=target.dtype)
                    self._vector_cache[cache_key] = vec

                if target.dim() == 3:
                    if self.steer_position == "all":
                        target[:, :, :] += self.coeff * vec.view(1, 1, -1)
                    else:
                        target[:, -1, :] += self.coeff * vec
                elif target.dim() == 2:
                    target[-1, :] += self.coeff * vec
                return output
            return hook

        layers = get_model_layers(self.model, self.model_key)
        if self.steer_all_layers:
            handles = [
                layers[layer_idx].register_forward_hook(hook_for_layer(layer_idx))
                for layer_idx in sorted(self.vectors)
                if 0 <= layer_idx < len(layers)
            ]
        else:
            handles = [layers[self.layer_idx].register_forward_hook(hook_for_layer(self.layer_idx))]
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.stop_token_ids,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.model_key and self.model_key.startswith("qwen3-") and not deterministic:
            gen_kwargs.update(
                {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
            )
        else:
            gen_kwargs["do_sample"] = False

        try:
            out = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        finally:
            for handle in handles:
                handle.remove()

        generated_tokens = out[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def _safe_obs_text(builder, obj):
    if obj is None:
        return ""
    try:
        return builder(obj)
    except Exception as exc:
        return f"[unavailable:{type(exc).__name__}]"


def _axtree_text_from_obs(obs):
    return _safe_obs_text(flatten_axtree_to_str, obs.get("axtree_object"))


def _dom_text_from_obs(obs):
    dom_text = _safe_obs_text(flatten_dom_to_str, obs.get("dom_object"))
    if not dom_text:
        return ""
    try:
        return prune_html(dom_text)
    except Exception:
        return dom_text


def _ids_from_axtree_text(axtree_text):
    return re.findall(r"\[([^\]\s]+)\]", axtree_text or "")


def _labels_by_id(axtree_text):
    labels = {}
    for line in (axtree_text or "").splitlines():
        match = re.search(r"\[([^\]\s]+)\](.*)", line)
        if match:
            labels[match.group(1)] = match.group(2).strip()
    return labels


def _valid_ids_from_obs(obs):
    return _ids_from_axtree_text(_axtree_text_from_obs(obs))


def parse_action(action):
    match = ACTION_RE.match(str(action or "").strip())
    if not match:
        return None
    quoted = QUOTED_RE.findall(match.group(2))
    return {"type": match.group(1), "first_arg": quoted[0] if quoted else ""}


def action_grounding_metrics(action, valid_ids, dataset="miniwob", labels=None):
    parsed = parse_action(action)
    action_type = parsed["type"] if parsed else ""
    first_arg = parsed["first_arg"] if parsed else ""
    valid = set(str(x) for x in valid_ids)
    label_text = " ".join((labels or {}).values()).lower()
    known_actions = set(get_action_set(dataset).action_set)
    id_argument_expected = action_type in ID_ARGUMENT_ACTIONS
    missing_current_id_arg = bool(parsed and id_argument_expected and not first_arg)
    invalid_current_id = bool(
        parsed and id_argument_expected and first_arg and first_arg not in valid
    )
    return {
        "parse_valid": bool(parsed),
        "action_type": action_type,
        "action_type_valid": bool(parsed and action_type in known_actions),
        "action_first_arg": first_arg,
        "id_argument_expected": bool(parsed and id_argument_expected),
        "valid_current_id": bool(parsed and id_argument_expected and first_arg in valid),
        "invalid_current_id": invalid_current_id,
        "invalid_or_bogus_argument": bool(
            (not parsed)
            or missing_current_id_arg
            or invalid_current_id
            and first_arg.lower() not in label_text
        ),
        "label_as_argument": bool(
            invalid_current_id and first_arg.lower() in label_text
        ),
    }


def build_prompt(
    obs,
    max_elems=80,
    dataset="miniwob",
    eval_instruction="",
    action_examples=True,
):
    """Build BrowserGym demo_agent-style prompt from observation."""
    _ = max_elems
    dom_text = _dom_text_from_obs(obs)
    axtree_text = _axtree_text_from_obs(obs)

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

    action_space = get_action_set(dataset).describe(
        with_long_description=False,
        with_examples=action_examples,
    )

    extra = f"\n\n{eval_instruction.strip()}" if eval_instruction else ""

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"# Goal\n{goal}\n\n"
        f"# Currently open tabs\n{tabs_text}\n\n"
        f"# Current page Accessibility Tree\n{axtree_text}\n\n"
        f"# Current page DOM\n{dom_text}\n\n"
        f"# Action Space\n\n{action_space}\n\n"
        f"# Error message from last action\n\n{obs.get('last_action_error', '')}\n\n"
        "# Next action\n"
        "Respond with exactly one valid action from the Action Space. Do not include any explanation or extra text."
        f"{extra}"
    )


def _run_episode(
    env,
    model,
    seed,
    max_steps,
    max_elems,
    max_new_tokens,
    steer,
    dataset="miniwob",
    eval_instruction="",
    action_examples=True,
    interface_mode="original",
):
    obs, _ = env.reset(seed=seed)
    outputs = []
    actions = []
    errors = []
    step_metrics = []
    total_reward = 0.0
    success = False

    for _ in range(max_steps):
        prompt = build_prompt(
            obs,
            max_elems,
            dataset=dataset,
            eval_instruction=eval_instruction,
            action_examples=action_examples,
        )
        shown_prompt, transform = apply_interface_variant(prompt, interface_mode, seed + len(actions))
        output = model.generate(shown_prompt, steer=steer, max_new_tokens=max_new_tokens)
        action = str(output or "").strip()
        real_action = executable_action_from_shown(action, transform)
        metrics = interface_action_metrics(action, transform=transform)
        real_metrics = action_grounding_metrics(
            real_action,
            _valid_ids_from_obs(obs),
            dataset=dataset,
            labels=_labels_by_id(_axtree_text_from_obs(obs)),
        )

        outputs.append(output)
        actions.append(action)
        step_metrics.append({**metrics, "real_action": real_action, "real_metrics": real_metrics})

        try:
            obs, reward, terminated, truncated, _info = env.step(real_action)
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
        "action_metrics": step_metrics,
        "steps": len(actions),
        "total_reward": total_reward,
        "success": bool(success),
        "error": last_error,
    }


def _get_prompt(model, obs, env, max_elems, dataset="miniwob"):
    """Build text prompt for steering vector construction."""
    _ = env
    _ = model
    prompt = build_prompt(obs, max_elems, dataset=dataset)
    return prompt


def _get_hidden_state_offset(model, states):
    """Return offset so hidden_states[offset] maps to block 0."""
    num_layers = len(get_model_layers(model.model, model.model_key))
    num_states = len(states)
    if num_states < num_layers:
        raise ValueError("hidden_states shorter than model layers")
    return num_states - num_layers


def _compute_activation_diff(model, pos, neg, max_new_tokens):
    if model.vector_method == "prompt":
        pos_states = model._prompt_activation(pos)
        neg_states = model._prompt_activation(neg)
    else:
        pos_text = model.generate(
            pos, steer=False, max_new_tokens=max_new_tokens, deterministic=True
        )
        neg_text = model.generate(
            neg, steer=False, max_new_tokens=max_new_tokens, deterministic=True
        )
        pos_states = model._response_activation(pos, pos_text)
        neg_states = model._response_activation(neg, neg_text)

    offset = _get_hidden_state_offset(model, pos_states)
    num_layers = len(pos_states) - offset

    diffs = {}
    for block_idx in range(num_layers):
        state_idx = block_idx + offset
        pos_layer = pos_states[state_idx][0, -1].float().cpu().numpy()
        neg_layer = neg_states[state_idx][0, -1].float().cpu().numpy()
        diffs[block_idx] = pos_layer - neg_layer

    return diffs


def vector_cache_subdir(cache_dir, model_alias, seed, dataset="miniwob", cache_tag=None):
    dataset = _normalize_dataset(dataset)
    if cache_tag:
        seed_dir = f"{cache_tag}_seed_{seed}"
    elif dataset == "miniwob":
        seed_dir = f"seed_{seed}"
    else:
        seed_dir = f"{dataset}_seed_{seed}"
    return os.path.join(cache_dir, model_alias, seed_dir)


def measure_avg_residual_norms(
    model,
    tasks,
    steps,
    max_elems,
    layer_indices,
    dataset="miniwob",
    seed=0,
    seed_mode="project",
    training_plan=None,
    interface_modes=None,
):
    """Measure average last-token residual norm on vector-construction prompts."""
    dataset = _normalize_dataset(dataset)
    layers = sorted(set(int(layer_idx) for layer_idx in layer_indices))
    totals = {layer_idx: 0.0 for layer_idx in layers}
    count = 0
    plan = (
        [(task, int(seed_val)) for task, seed_val in training_plan][: int(steps)]
        if training_plan is not None
        else build_training_plan(tasks, steps, seed, seed_mode)
    )
    interface_modes = parse_interface_modes(interface_modes, default=("original",))
    pbar = tqdm(total=len(plan), desc="Measuring residual norms")
    task_plans = {}
    for task, seed_val in plan:
        task_plans.setdefault(task, []).append(seed_val)

    for task in tasks:
        env = make_browsergym_env(dataset, task)
        for seed_val in task_plans.get(task, []):
            obs, _ = env.reset(seed=seed_val)
            prompt = _get_prompt(model, obs, env, max_elems, dataset=dataset)
            mode = interface_modes[pbar.n % len(interface_modes)]
            prompt, _transform = apply_interface_variant(prompt, mode, seed_val)
            states = model._prompt_activation(prompt)
            offset = _get_hidden_state_offset(model, states)
            for layer_idx in layers:
                state_idx = layer_idx + offset
                totals[layer_idx] += float(states[state_idx][0, -1].float().norm().cpu())
            count += 1
            pbar.update(1)
            if pbar.n >= steps:
                break
        env.close()
        if pbar.n >= steps:
            break
    pbar.close()

    return {layer_idx: totals[layer_idx] / max(1, count) for layer_idx in layers}


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
    seed_mode="project",
    dataset="miniwob",
    cache_tag=None,
    training_plan=None,
    interface_modes=None,
):
    dataset = _normalize_dataset(dataset)
    interface_modes = parse_interface_modes(interface_modes, default=("original",))
    if prompt_type == "combined":
        print(
            "Computing Combined Vector for all layers (format_accuracy + composite_1)..."
        )
        vec_a_sums = {}  # layer_idx -> accumulated vector A
        vec_b_sums = {}  # layer_idx -> accumulated vector B

        plan = (
            [(task, int(seed_val)) for task, seed_val in training_plan][: int(steps)]
            if training_plan is not None
            else build_training_plan(tasks, steps, seed, seed_mode)
        )
        pbar = tqdm(total=len(plan), desc="Computing combined vectors")
        task_plans = {}
        for task, seed_val in plan:
            task_plans.setdefault(task, []).append(seed_val)

        for task in tasks:
            env = make_browsergym_env(dataset, task)
            for seed_val in task_plans.get(task, []):
                obs, _ = env.reset(seed=seed_val)

                base_prompt = _get_prompt(model, obs, env, max_elems, dataset=dataset)
                mode = interface_modes[pbar.n % len(interface_modes)]
                base_prompt, _transform = apply_interface_variant(base_prompt, mode, seed_val)

                pos_a = f"{base_prompt}\n{PROMPT_CONFIGS['format_accuracy']['pos']}"
                neg_a = f"{base_prompt}\n{PROMPT_CONFIGS['format_accuracy']['neg']}"
                diffs_a = _compute_activation_diff(model, pos_a, neg_a, max_new_tokens)

                pos_b = f"{base_prompt}\n{PROMPT_CONFIGS['composite_1']['pos']}"
                neg_b = f"{base_prompt}\n{PROMPT_CONFIGS['composite_1']['neg']}"
                diffs_b = _compute_activation_diff(model, pos_b, neg_b, max_new_tokens)

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

        if cache_dir and model_alias is not None:
            cache_subdir = vector_cache_subdir(
                cache_dir, model_alias, seed, dataset=dataset, cache_tag=cache_tag
            )
            os.makedirs(cache_subdir, exist_ok=True)
            for layer_idx, vec in all_vectors.items():
                cache_path = os.path.join(
                    cache_subdir, f"{prompt_type}_L{layer_idx}.pt"
                )
                torch.save(
                    torch.tensor(vec, dtype=torch.float32, device="cpu"), cache_path
                )
            print(f">>> Saved {len(all_vectors)} vectors to {cache_subdir}")

        model.set_vector(all_vectors)
        return

    totals = {}
    pos_instr = PROMPT_CONFIGS[prompt_type]["pos"]
    neg_instr = PROMPT_CONFIGS[prompt_type]["neg"]

    plan = (
        [(task, int(seed_val)) for task, seed_val in training_plan][: int(steps)]
        if training_plan is not None
        else build_training_plan(tasks, steps, seed, seed_mode)
    )
    pbar = tqdm(total=len(plan), desc="Computing steering vectors for all layers")
    task_plans = {}
    for task, seed_val in plan:
        task_plans.setdefault(task, []).append(seed_val)

    for task in tasks:
        env = make_browsergym_env(dataset, task)
        for seed_val in task_plans.get(task, []):
            obs, _ = env.reset(seed=seed_val)

            base_prompt = _get_prompt(model, obs, env, max_elems, dataset=dataset)
            mode = interface_modes[pbar.n % len(interface_modes)]
            base_prompt, _transform = apply_interface_variant(base_prompt, mode, seed_val)
            pos = f"{base_prompt}\n{pos_instr}"
            neg = f"{base_prompt}\n{neg_instr}"

            diffs = _compute_activation_diff(model, pos, neg, max_new_tokens)

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

    all_vectors = {}
    for layer_idx, total in totals.items():
        vec = total / max(1, pbar.n)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        all_vectors[layer_idx] = vec

    if cache_dir and model_alias is not None:
        cache_subdir = vector_cache_subdir(
            cache_dir, model_alias, seed, dataset=dataset, cache_tag=cache_tag
        )
        os.makedirs(cache_subdir, exist_ok=True)
        for layer_idx, vec in all_vectors.items():
            cache_path = os.path.join(cache_subdir, f"{prompt_type}_L{layer_idx}.pt")
            torch.save(torch.tensor(vec, dtype=torch.float32, device="cpu"), cache_path)
        print(f">>> Saved {len(all_vectors)} vectors to {cache_subdir}")

    model.set_vector(all_vectors)


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
    episodes_per_task,
    max_elems,
    max_new_tokens,
    out_path,
    base_only=False,
    steer_only=False,
    eval_seed=0,
    base_records=None,
    episode_steps=10,
    seed_mode="project",
    episode_plan=None,
    dataset="miniwob",
    eval_instruction="",
    action_examples=True,
    interface_mode="original",
):
    """Evaluate model on tasks, comparing baseline vs steered."""
    dataset = _normalize_dataset(dataset)
    interface_mode = parse_interface_modes(interface_mode, default=("original",))[0]
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
    paired_gains = 0
    paired_losses = 0
    metric_names = [
        "parse_valid",
        "action_type_valid",
        "valid_current_id",
        "invalid_current_id",
        "invalid_or_bogus_argument",
        "bogus_argument",
        "copied_example_id",
        "stale_id",
        "label_as_id",
        "label_as_argument",
    ]
    base_metric_hits = {name: 0 for name in metric_names}
    steer_metric_hits = {name: 0 for name in metric_names}
    base_metric_total = 0
    steer_metric_total = 0

    episodes_per_task = int(episodes_per_task)
    if episode_plan is None:
        episode_plan = build_episode_plan(tasks, episodes_per_task, eval_seed, seed_mode)
    else:
        episode_plan = [(task, int(seed)) for task, seed in episode_plan]

    target_episodes = len(episode_plan)
    pbar = tqdm(total=target_episodes, desc="Evaluating")

    with open(out_path, "w", encoding="utf-8") as f:
        episode_plan_by_task = {}
        for task, seed in episode_plan:
            episode_plan_by_task.setdefault(task, []).append(seed)

        for task in tasks:
            env = make_browsergym_env(dataset, task)
            for seed in episode_plan_by_task.get(task, []):
                record = {
                    "dataset": dataset,
                    "task": task,
                    "seed": seed,
                    "interface_mode": interface_mode,
                    "remap_mode": interface_mode,
                }
                base_success = None

                if steer_only:
                    assert base_records is not None
                    base_record = base_records.get((task, seed))
                    if base_record is None:
                        raise ValueError(
                            f"Missing baseline record for task={task}, seed={seed} in steer-only mode"
                        )
                    base_mode = base_record.get("interface_mode") or base_record.get("remap_mode")
                    if base_mode and base_mode != interface_mode:
                        raise ValueError(
                            f"Baseline record mode {base_mode!r} does not match eval mode {interface_mode!r} "
                            f"for task={task}, seed={seed}"
                        )

                    base_action = base_record.get("base_action")
                    base_success = base_record.get("base_success")
                    base_error = str(base_record.get("base_error", "") or "")
                    base_failed = bool(base_error)
                    base_metrics = base_record.get("base_action_metrics", [])
                    base_last_metrics = (
                        base_metrics[-1]
                        if isinstance(base_metrics, list) and base_metrics
                        else base_record.get("base_last_action_metrics", {})
                    )

                    record.update(
                        {
                            "base_output": base_record.get("base_output"),
                            "base_outputs": base_record.get("base_outputs", []),
                            "base_action": base_action,
                            "base_real_action": base_record.get("base_real_action", base_action),
                            "base_actions": base_record.get("base_actions", []),
                            "base_steps": base_record.get("base_steps"),
                            "base_total_reward": base_record.get("base_total_reward"),
                            "base_success": base_success,
                            "base_error": base_error,
                            "base_error_episode": bool(base_failed),
                            "base_action_metrics": base_metrics,
                            "base_last_action_metrics": base_last_metrics,
                        }
                    )
                    if base_success is not None:
                        base_hits += int(base_success)
                        base_total += 1
                    if base_failed:
                        error_episodes_base += 1
                    if isinstance(base_last_metrics, dict) and base_last_metrics:
                        for name in metric_names:
                            base_metric_hits[name] += int(bool(base_last_metrics.get(name)))
                        base_metric_total += 1
                else:
                    base_episode = _run_episode(
                        env,
                        model,
                        seed,
                        episode_steps,
                        max_elems,
                        max_new_tokens,
                        steer=False,
                        dataset=dataset,
                        eval_instruction=eval_instruction,
                        action_examples=action_examples,
                        interface_mode=interface_mode,
                    )
                    base_success = base_episode["success"]
                    base_error = base_episode["error"]
                    base_last_metrics = (
                        base_episode["action_metrics"][-1]
                        if base_episode["action_metrics"]
                        else {}
                    )
                    base_hits += int(base_success)
                    base_total += 1
                    if base_error:
                        error_episodes_base += 1
                    if base_last_metrics:
                        for name in metric_names:
                            base_metric_hits[name] += int(bool(base_last_metrics.get(name)))
                        base_metric_total += 1

                    record.update(
                        {
                            "base_output": base_episode["outputs"][-1]
                            if base_episode["outputs"]
                            else "",
                            "base_outputs": base_episode["outputs"],
                            "base_action": base_episode["actions"][-1]
                            if base_episode["actions"]
                            else "",
                            "base_real_action": base_last_metrics.get("real_action", ""),
                            "base_actions": base_episode["actions"],
                            "base_action_metrics": base_episode["action_metrics"],
                            "base_last_action_metrics": base_last_metrics,
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
                        dataset=dataset,
                        eval_instruction=eval_instruction,
                        action_examples=action_examples,
                        interface_mode=interface_mode,
                    )
                    steer_success = steer_episode["success"]
                    steer_error = steer_episode["error"]
                    steer_last_metrics = (
                        steer_episode["action_metrics"][-1]
                        if steer_episode["action_metrics"]
                        else {}
                    )
                    steer_hits += int(steer_success)
                    if steer_error:
                        error_episodes_steer += 1
                    if steer_last_metrics:
                        for name in metric_names:
                            steer_metric_hits[name] += int(bool(steer_last_metrics.get(name)))
                        steer_metric_total += 1
                    if base_success is not None:
                        paired_gains += int((not base_success) and steer_success)
                        paired_losses += int(base_success and (not steer_success))

                    record.update(
                        {
                            "steer_output": steer_episode["outputs"][-1]
                            if steer_episode["outputs"]
                            else "",
                            "steer_outputs": steer_episode["outputs"],
                            "steer_action": steer_episode["actions"][-1]
                            if steer_episode["actions"]
                            else "",
                            "steer_real_action": steer_last_metrics.get("real_action", ""),
                            "steer_actions": steer_episode["actions"],
                            "steer_action_metrics": steer_episode["action_metrics"],
                            "steer_last_action_metrics": steer_last_metrics,
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
        "paired_gains": paired_gains,
        "paired_losses": paired_losses,
        **{
            f"base_{name}": base_metric_hits[name] / max(1, base_metric_total)
            for name in metric_names
        },
        **{
            f"steer_{name}": steer_metric_hits[name] / max(1, steer_metric_total)
            if not base_only
            else 0
            for name in metric_names
        },
        "total_episodes": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Web Agent Steering Experiment")
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="0.5b")
    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        default="miniwob",
        help="BrowserGym dataset family to evaluate",
    )
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
        "--eval-instruction-type",
        choices=list(PROMPT_CONFIGS.keys()),
        default=None,
        help="Append this prompt config's positive instruction during evaluation only.",
    )
    parser.add_argument(
        "--no-action-examples",
        action="store_true",
        help="Omit Action Space examples from evaluation prompts.",
    )
    parser.add_argument(
        "--interface-mode",
        choices=INTERFACE_MODES,
        default="original",
        help="Action-interface id schema shown during evaluation.",
    )
    parser.add_argument(
        "--interface-train-modes",
        default="original",
        help="Comma-separated action-interface schemas for vector construction.",
    )
    parser.add_argument(
        "--vector-method", choices=["response", "prompt"], default="response"
    )
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--episode-steps", type=int, default=10)
    parser.add_argument("--tasks", default="all", help="Task list or 'all'")
    parser.add_argument(
        "--plan",
        help="Explicit comma-separated task:seed pairs; overrides --tasks and --episodes-per-task eval plan.",
    )
    parser.add_argument(
        "--plan-file",
        help="File containing comma-separated task:seed pairs; overrides --plan.",
    )
    parser.add_argument("--out", default="results.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-mode",
        choices=sorted(SEED_MODES),
        default="project",
        help="Seed generation mode: project defaults or BrowserGym-compatible numpy seeds",
    )
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
        "--steer-position",
        choices=["last", "all"],
        default="last",
        help="Token positions to perturb inside each hooked forward pass",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Apply cached vectors at every layer instead of one target layer",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of steering vector",
    )
    args = parser.parse_args()

    if args.seed_mode == "browsergym":
        if args.seed == PROJECT_DEFAULT_SEED:
            args.seed = BROWSERGYM_BENCHMARK_SEED
        if args.episodes_per_task == 3:
            args.episodes_per_task = BROWSERGYM_BENCHMARK_REPEATS

    layer_idx = get_layer(args.model, args.layer)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    explicit_plan = (
        load_plan_file(args.plan_file)
        if args.plan_file
        else parse_plan_arg(args.plan)
        if args.plan
        else None
    )
    if explicit_plan is not None:
        tasks = sorted({task for task, _seed in explicit_plan})
    elif args.tasks == "all":
        tasks = list_browsergym_tasks(args.dataset)
    else:
        tasks = parse_tasks_arg(args.tasks, dataset=args.dataset)

    if args.base_only and args.steer_only:
        raise ValueError("Cannot set both --base-only and --steer-only")

    base_records = None
    if args.steer_only:
        if not args.base_jsonl:
            raise ValueError("--steer-only requires --base-jsonl")
        base_records = load_base_jsonl(args.base_jsonl)

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
        steer_position=args.steer_position,
        steer_all_layers=args.all_layers,
    )

    if not args.base_only:
        interface_train_modes = parse_interface_modes(args.interface_train_modes)
        cache_tag = None
        if interface_train_modes != ["original"]:
            cache_tag = interface_cache_tag(interface_train_modes)
        cache_subdir = vector_cache_subdir(
            args.cache_dir, args.model, args.seed, dataset=args.dataset, cache_tag=cache_tag
        )
        cache_path = os.path.join(cache_subdir, f"{args.prompt_type}_L{layer_idx}.pt")

        if args.all_layers and os.path.isdir(cache_subdir) and not args.force_recompute:
            loaded_layers = []
            for name in sorted(os.listdir(cache_subdir)):
                prefix = f"{args.prompt_type}_L"
                if not name.startswith(prefix) or not name.endswith(".pt"):
                    continue
                layer_text = name[len(prefix):-3]
                if not layer_text.isdigit():
                    continue
                cached_vector = torch.load(
                    os.path.join(cache_subdir, name), map_location="cpu"
                )
                layer = int(layer_text)
                model.set_vector(cached_vector, layer_idx=layer)
                loaded_layers.append(layer)
            if not loaded_layers:
                raise RuntimeError(f"No cached vectors found in {cache_subdir}")
            print(f">>> Loaded cached vectors for layers: {loaded_layers}")
        elif os.path.exists(cache_path) and not args.force_recompute:
            print(f">>> Loading cached vector from {cache_path}")
            cached_vector = torch.load(cache_path, map_location="cpu")
            model.set_vector(cached_vector, layer_idx=layer_idx)
        else:
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
                seed_mode=args.seed_mode,
                dataset=args.dataset,
                cache_tag=cache_tag,
                interface_modes=interface_train_modes,
            )
            if model.vector is None:
                raise RuntimeError(f"Failed to load vector for layer {layer_idx}")

    results = evaluate(
        model,
        tasks,
        args.episodes_per_task,
        80,
        80,
        args.out,
        args.base_only,
        args.steer_only,
        eval_seed=args.seed,
        base_records=base_records,
        episode_steps=args.episode_steps,
        seed_mode=args.seed_mode,
        episode_plan=explicit_plan,
        dataset=args.dataset,
        eval_instruction=PROMPT_CONFIGS[args.eval_instruction_type]["pos"]
        if args.eval_instruction_type
        else "",
        action_examples=not args.no_action_examples,
        interface_mode=args.interface_mode,
    )

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
