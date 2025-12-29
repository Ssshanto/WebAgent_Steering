import json
from datasets import load_dataset
from tqdm import tqdm

def _load_mind2web_dataset(cache_dir: str):
    last_error = None
    for dataset_id in ("osunlp/Mind2Web", "osunlp/mind2web"):
        try:
            return load_dataset(dataset_id, split="train", streaming=False, cache_dir=cache_dir)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to load Mind2Web dataset from Hub/cache (last error: {last_error})")


def _normalize_op(op):
    if op is None:
        return None
    if isinstance(op, int):
        return {1: "CLICK", 2: "SELECT", 3: "TYPE"}.get(op)
    return str(op).strip().upper()


def get_contrastive_pairs(num_samples=50, cache_dir="dataset_cache"):
    """
    Downloads Mind2Web (streaming) and constructs Positive (JSON) vs Negative (Chat) pairs.
    """
    print(f"Loading {num_samples} examples from Mind2Web...")
    dataset = _load_mind2web_dataset(cache_dir=cache_dir)
    
    pairs = []
    
    for sample in tqdm(dataset):
        if len(pairs) >= num_samples:
            break

        goal = sample.get("confirmed_task")
        actions = sample.get("actions") or []
        if not goal or not actions:
            continue

        for action in actions:
            if len(pairs) >= num_samples:
                break

            html = action.get("cleaned_html") or action.get("raw_html")
            operation = action.get("operation") or {}
            op = _normalize_op(operation.get("op"))
            if not html or not op:
                continue

            pos_candidates = action.get("pos_candidates") or []
            if not pos_candidates:
                continue

            target_id = pos_candidates[0].get("backend_node_id")
            if target_id is None:
                continue

            pos_json, neg_chat = "", ""
            if op == "CLICK":
                pos_json = json.dumps({"action": "click", "target": target_id})
                neg_chat = f"I will click on the element {target_id}."
            elif op == "SELECT":
                val = operation.get("value", "")
                pos_json = json.dumps({"action": "select", "target": target_id, "value": val})
                neg_chat = f"I will select '{val}' from {target_id}."
            elif op in ("TYPE", "INPUT"):
                val = operation.get("value", "")
                pos_json = json.dumps({"action": "type", "target": target_id, "value": val})
                neg_chat = f"I will type '{val}' into {target_id}."
            else:
                continue

            pairs.append(
                {
                    "html": html,
                    "goal": goal,
                    "positive": pos_json,
                    "negative": neg_chat,
                }
            )
        
    return pairs


def _safe_json_loads(text):
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _format_candidate(candidate, max_attr_len=160):
    tag = candidate.get("tag") or "div"
    backend_id = candidate.get("backend_node_id") or ""
    attrs_raw = candidate.get("attributes") or ""
    attrs = _safe_json_loads(attrs_raw)
    if attrs:
        for drop_key in (
            "backend_node_id",
            "bounding_box_rect",
            "data_pw_testid_buckeye_candidate",
            "class",
        ):
            attrs.pop(drop_key, None)
    parts = []
    for key in ("id", "name", "aria_label", "aria-label", "title", "placeholder", "type", "value", "role"):
        val = attrs.get(key)
        if val:
            parts.append(f'{key}="{val}"')
    attr_str = " " + " ".join(parts) if parts else ""
    text = f"<{tag} id={backend_id}{attr_str}>"
    if len(text) > max_attr_len:
        text = text[: max_attr_len - 3] + "..."
    return text


def build_candidate_options(action, max_neg_candidates=7, max_attr_len=160):
    options = [("A", None, "None of the above")]
    seen = set()

    def add_candidate(candidate):
        backend_id = candidate.get("backend_node_id")
        if not backend_id or backend_id in seen:
            return
        seen.add(backend_id)
        label = chr(ord("A") + len(options))
        options.append((label, backend_id, _format_candidate(candidate, max_attr_len)))

    for cand in action.get("pos_candidates") or []:
        add_candidate(cand)
    for cand in (action.get("neg_candidates") or [])[:max_neg_candidates]:
        add_candidate(cand)

    return options


def get_mind2web_tasks(num_tasks=50, task_offset=0, cache_dir="dataset_cache"):
    dataset = _load_mind2web_dataset(cache_dir=cache_dir)
    tasks = []
    skipped = 0

    for sample in dataset:
        if task_offset and skipped < task_offset:
            skipped += 1
            continue
        if not sample.get("actions"):
            continue
        tasks.append(sample)
        if len(tasks) >= num_tasks:
            break

    return tasks


def iter_action_steps(
    task,
    max_neg_candidates=7,
    max_attr_len=160,
    max_html_chars=1000,
):
    goal = task.get("confirmed_task") or ""
    actions = task.get("actions") or []
    action_reprs = task.get("action_reprs") or []

    for idx, action in enumerate(actions):
        html = action.get("cleaned_html") or action.get("raw_html") or ""
        html = html[:max_html_chars] if max_html_chars else html
        operation = action.get("operation") or {}
        op = _normalize_op(operation.get("op"))
        if not op:
            continue
        pos_candidates = action.get("pos_candidates") or []
        if not pos_candidates:
            continue

        options = build_candidate_options(
            action,
            max_neg_candidates=max_neg_candidates,
            max_attr_len=max_attr_len,
        )
        prev_actions = action_reprs[:idx] if action_reprs else []

        yield {
            "annotation_id": task.get("annotation_id"),
            "action_uid": action.get("action_uid"),
            "goal": goal,
            "html": html,
            "previous_actions": prev_actions,
            "options": options,
            "gold_op": op,
            "gold_value": operation.get("value", "") or "",
            "gold_targets": {c.get("backend_node_id") for c in pos_candidates if c.get("backend_node_id")},
        }
