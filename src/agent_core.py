import ast
import copy
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_MAP = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}

LAYER_MAP = {
    "qwen3-0.6b": 14,
    "qwen3-1.7b": 14,
    "qwen3-4b": 18,
    "qwen3-8b": 18,
}

SYSTEM_PROMPT = (
    "# Instructions\n\n"
    "Review the current page state and output one BrowserGym action command."
)

_ACTION_SET = None


def get_action_set():
    global _ACTION_SET
    if _ACTION_SET is None:
        from browsergym.core.action.highlevel import HighLevelActionSet

        _ACTION_SET = HighLevelActionSet(
            subsets=["miniwob_all"],
            strict=False,
            multiaction=False,
            demo_mode="off",
        )
    return _ACTION_SET


BID_REQUIRED_ACTIONS = {
    "click",
    "dblclick",
    "hover",
    "focus",
    "clear",
    "type",
    "fill",
    "check",
    "uncheck",
    "select_option",
    "click_option",
    "right_click",
    "drag_and_drop",
    "upload_file",
}

SYNTAX_ERROR_KEYWORDS = (
    "syntax",
    "parse",
    "malformed",
    "invalid action",
    "invalid command",
    "unexpected",
    "step_exception:syntaxerror",
)

GROUNDING_ERROR_KEYWORDS = (
    "bid",
    "element",
    "not found",
    "missing",
    "non-interactable",
    "not interactable",
    "not clickable",
    "stale",
)

ACTION_TYPE_ERROR_KEYWORDS = (
    "action type",
    "unsupported",
    "not allowed",
    "cannot",
    "mismatch",
    "invalid operation",
)

BID_VALUE_RE = re.compile(r"^[0-9]+$")


def utc_now_iso():
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def stable_hash(value):
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def derive_episode_seed(global_seed, namespace, task, episode_idx):
    payload = {
        "global_seed": int(global_seed),
        "namespace": str(namespace),
        "task": str(task),
        "episode_idx": int(episode_idx),
    }
    return int(stable_hash(payload)[:8], 16) & 0x7FFFFFFF


def list_miniwob_tasks():
    import browsergym.miniwob  # noqa: F401
    import gymnasium as gym

    env_ids = [
        env_id
        for env_id in gym.envs.registry.keys()
        if env_id.startswith("browsergym/miniwob.")
    ]
    return sorted(env_id.split("browsergym/miniwob.", 1)[1] for env_id in env_ids)


def make_miniwob_env(task):
    import browsergym.miniwob  # noqa: F401
    import gymnasium as gym

    return gym.make(
        f"browsergym/miniwob.{task}",
        action_mapping=get_action_set().to_python_code,
    )


def resolve_tasks(tasks_arg="all", task_manifest_path=None):
    if task_manifest_path and Path(task_manifest_path).exists():
        payload = json.loads(Path(task_manifest_path).read_text(encoding="utf-8"))
        task_list = payload["tasks"] if isinstance(payload, dict) else payload
        tasks = [str(t) for t in task_list]
        if tasks_arg != "all":
            requested = [x.strip() for x in str(tasks_arg).split(",") if x.strip()]
            tasks = requested
        return tasks

    tasks = (
        list_miniwob_tasks()
        if tasks_arg == "all"
        else [x.strip() for x in str(tasks_arg).split(",") if x.strip()]
    )
    if task_manifest_path:
        p = Path(task_manifest_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"generated_at_utc": utc_now_iso(), "tasks": tasks}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return tasks


def parse_layer_spec(spec):
    s = str(spec).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def get_layer(model_key, layer_arg):
    if str(layer_arg) == "auto":
        return LAYER_MAP[model_key]
    return int(layer_arg)


def get_model_layers(model):
    return model.model.layers


def _contains_any_keyword(text, keywords):
    t = str(text or "").lower()
    return any(k in t for k in keywords)


def _node_scalar(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return str(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value, int
        ):
            return str(-node.operand.value)
    return None


def _looks_like_bid(value):
    s = str(value or "").strip()
    if not s:
        return False
    if BID_VALUE_RE.match(s):
        return True
    if s.startswith("bid-") and BID_VALUE_RE.match(s[4:]):
        return True
    return False


def _parse_action_signature(action):
    parsed = {"parse_ok": False, "action_type": "", "bids": [], "bid_required": False}
    raw = str(action or "").strip()
    if not raw:
        return parsed
    try:
        expr = ast.parse(raw, mode="eval").body
    except Exception:
        return parsed
    if not isinstance(expr, ast.Call):
        return parsed

    if isinstance(expr.func, ast.Name):
        action_type = expr.func.id
    elif isinstance(expr.func, ast.Attribute):
        action_type = expr.func.attr
    else:
        action_type = ""

    bids = []
    for kw in expr.keywords:
        if kw.arg and "bid" in kw.arg.lower():
            value = _node_scalar(kw.value)
            if value is not None:
                bids.append(value)
    if action_type in BID_REQUIRED_ACTIONS and expr.args:
        value = _node_scalar(expr.args[0])
        if value is not None and _looks_like_bid(value):
            bids.append(value)

    parsed["parse_ok"] = bool(action_type)
    parsed["action_type"] = action_type
    parsed["bids"] = bids
    parsed["bid_required"] = action_type in BID_REQUIRED_ACTIONS
    return parsed


def classify_action_step(action, error_text):
    sig = _parse_action_signature(action)
    parse_ok = sig["parse_ok"]
    has_bid = len(sig["bids"]) > 0 and any(str(x).strip() for x in sig["bids"])
    bid_required = sig["bid_required"]

    syntax_flag = (not parse_ok) or _contains_any_keyword(
        error_text, SYNTAX_ERROR_KEYWORDS
    )
    grounding_flag = (
        parse_ok and bid_required and (not has_bid)
    ) or _contains_any_keyword(error_text, GROUNDING_ERROR_KEYWORDS)
    action_type_flag = _contains_any_keyword(error_text, ACTION_TYPE_ERROR_KEYWORDS)

    action_type_known = bool(parse_ok or action_type_flag)
    bid_grounding_known = bool(parse_ok or grounding_flag)
    return {
        "action_type": sig["action_type"],
        "parse_ok": bool(parse_ok),
        "action_type_known": action_type_known,
        "bid_grounding_known": bid_grounding_known,
        "syntax_ok": not syntax_flag,
        "bid_grounding_ok": not grounding_flag,
        "action_type_ok": not action_type_flag,
    }


def _apply_template(tokenizer, prompt, model_key):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if str(model_key).startswith("qwen3-"):
                return f"{text}\n/no_think"
            return text
        except Exception:
            return f"User: {prompt}\nAssistant:\n/no_think"


def build_prompt(obs, max_elems=80, strict_action_prompt=False):
    from browsergym.utils.obs import (
        flatten_axtree_to_str,
        flatten_dom_to_str,
        prune_html,
    )

    dom_text = prune_html(flatten_dom_to_str(obs["dom_object"]))
    axtree_text = flatten_axtree_to_str(obs["axtree_object"])
    dom_lines = dom_text.splitlines()[: int(max_elems)]
    axtree_lines = axtree_text.splitlines()[: int(max_elems)]
    dom_text = "\n".join(dom_lines)
    axtree_text = "\n".join(axtree_lines)

    goal = obs.get("goal", "")
    goal_obj = obs.get("goal_object")
    if isinstance(goal_obj, list):
        xs = [
            str(x.get("text", ""))
            for x in goal_obj
            if isinstance(x, dict) and x.get("type") == "text"
        ]
        if xs:
            goal = "\n".join(xs).strip()

    tabs = []
    urls = obs.get("open_pages_urls", [])
    titles = obs.get("open_pages_titles", [])
    active = obs.get("active_page_index", 0)
    for i, (url, title) in enumerate(zip(urls, titles)):
        flag = " (active tab)" if i == active else ""
        tabs.append(f"Tab {i}{flag}\n  Title: {title}\n  URL: {url}")
    tabs_text = (
        "\n".join(tabs)
        if tabs
        else "Tab 0 (active tab)\n  Title: unknown\n  URL: unknown"
    )

    action_space = get_action_set().describe(
        with_long_description=False, with_examples=True
    )
    if strict_action_prompt:
        next_action = "# Next action\nOutput exactly one valid BrowserGym action command. No explanation."
    else:
        next_action = "# Next action\nThink briefly and output one best action command."

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"# Goal\n{goal}\n\n"
        f"# Currently open tabs\n{tabs_text}\n\n"
        f"# Current page Accessibility Tree\n{axtree_text}\n\n"
        f"# Current page DOM\n{dom_text}\n\n"
        f"# Action Space\n\n{action_space}\n\n"
        f"# Error message from last action\n\n{obs.get('last_action_error', '')}\n\n"
        f"{next_action}"
    )


class SteeredModel:
    def __init__(self, model_alias, layer_idx):
        self.model_alias = model_alias
        self.layer_idx = int(layer_idx)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_MAP[model_alias], trust_remote_code=True
        )
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_MAP[model_alias], torch_dtype=dtype, trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.stop_token_ids = [self.tokenizer.eos_token_id]

    def num_layers(self):
        return len(get_model_layers(self.model))

    def prompt_last_token_states(self, prompt):
        formatted = _apply_template(self.tokenizer, prompt, self.model_alias)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hs = out.hidden_states
        n = self.num_layers()
        offset = len(hs) - n
        return {i: hs[offset + i][0, -1].detach() for i in range(n)}

    def generate(self, prompt, max_new_tokens=80, edit=None, deterministic=True):
        formatted = _apply_template(self.tokenizer, prompt, self.model_alias)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        handle = None

        if edit is not None:
            layers = get_model_layers(self.model)
            layer = layers[int(edit.layer_idx)]

            def hook(_module, _inp, output):
                target = output if torch.is_tensor(output) else output[0]
                if target.dim() == 3:
                    x = target[:, -1, :]
                    target[:, -1, :] = edit.apply(x)
                else:
                    x = target[-1:, :]
                    target[-1:, :] = edit.apply(x)
                return output

            handle = layer.register_forward_hook(hook)

        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "eos_token_id": self.stop_token_ids,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if str(self.model_alias).startswith("qwen3-"):
            gc = copy.deepcopy(self.model.generation_config)
            gc.do_sample = not deterministic
            if deterministic:
                for k in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "typical_p",
                    "penalty_alpha",
                ):
                    if hasattr(gc, k):
                        setattr(gc, k, None)
            kwargs["generation_config"] = gc
        else:
            kwargs["do_sample"] = not deterministic

        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        if handle is not None:
            handle.remove()
        gen = out[0][input_len:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
