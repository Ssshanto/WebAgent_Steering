#!/usr/bin/env python3
"""Small helpers for MiniWob bid-grounding research scripts."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interface_variants import (  # noqa: E402,F401
    ACTION_START,
    AX_START,
    DOM_START,
    INTERFACE_MODES,
    InterfaceTransform,
    action_metrics,
    apply_interface_variant,
    executable_action_from_shown,
    ids_from_axtree_text,
    interface_cache_tag,
    labels_by_id,
    normalize_interface_mode,
    parse_action,
    parse_interface_modes,
    prompt_sections,
    remap_ids,
    unmap_action,
)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_plan(path_or_text):
    try:
        path = Path(path_or_text)
        text = path.read_text(encoding="utf-8") if path.exists() else path_or_text
    except OSError:
        text = path_or_text
    plan = []
    for item in text.replace("\n", ",").split(","):
        item = item.strip()
        if item:
            task, seed = item.rsplit(":", 1)
            plan.append((task, int(seed)))
    return plan


def apply_remap_to_prompt(prompt, real_to_presented):
    """Compatibility wrapper for older scripts.

    New code should call ``apply_interface_variant`` so metrics get distractor
    metadata.  This wrapper performs only the reversible id rewrite.
    """
    mode = "original"
    transform = InterfaceTransform(
        mode=mode,
        real_to_shown={str(k): str(v) for k, v in real_to_presented.items()},
        shown_to_real={str(v): str(k) for k, v in real_to_presented.items()},
        current_ids=[str(v) for v in real_to_presented.values()],
    )
    out, auto = apply_interface_variant(prompt, mode=mode, seed=0)
    _ = auto
    for real, shown in sorted(transform.real_to_shown.items(), key=lambda kv: len(kv[0]), reverse=True):
        out = out.replace(f"[{real}]", f"[{shown}]")
        out = out.replace(f"bid='{real}'", f"bid='{shown}'")
        out = out.replace(f'bid="{real}"', f'bid="{shown}"')
        out = out.replace(f"data-bid='{real}'", f"data-bid='{shown}'")
        out = out.replace(f'data-bid="{real}"', f'data-bid="{shown}"')
    return out


def normalize(vec):
    import torch

    if not torch.is_tensor(vec):
        vec = torch.as_tensor(vec)
    norm = vec.float().norm()
    return vec.float() / norm.clamp_min(1e-12)
