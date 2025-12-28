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
