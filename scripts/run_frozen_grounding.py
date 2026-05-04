#!/usr/bin/env python3
"""One-step static grounding eval: score generated action without env.step."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from grounding_utils import (  # noqa: E402
    INTERFACE_MODES,
    action_metrics,
    apply_interface_variant,
    parse_interface_modes,
    read_plan,
)
from miniwob_steer import MODEL_MAP, PROMPT_CONFIGS, SteeredModel, build_prompt, make_miniwob_env  # noqa: E402


def load_gold(path):
    if not path:
        return {}
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("base_success") and row.get("base_actions"):
                action = row["base_actions"][0]
            elif row.get("success") and row.get("actions"):
                action = row["actions"][0]
            else:
                continue
            metric = action_metrics(action, [])
            if metric["action_bid"]:
                gold[(row["task"], int(row["seed"]))] = metric["action_bid"]
    return gold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_MAP, default="gemma-3-4b")
    parser.add_argument("--condition", choices=["baseline", "positive", "negative", "steer"], required=True)
    parser.add_argument("--prompt-type", choices=PROMPT_CONFIGS, default="gemma_tree_pos_minus_base")
    parser.add_argument("--plan", required=True, help="plan text or path")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--vector", help="steering vector .pt for condition=steer")
    parser.add_argument("--gold-jsonl")
    parser.add_argument("--interface-mode", choices=INTERFACE_MODES, default="original")
    parser.add_argument("--interface-modes", default=None, help="Comma-separated modes; overrides --interface-mode")
    parser.add_argument("--max-elems", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.condition == "steer" and not args.vector:
        parser.error("--condition steer requires --vector")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = SteeredModel(MODEL_MAP[args.model], args.layer, args.alpha, model_key=args.model, steer_position="last")
    if args.condition == "steer":
        model.set_vector(torch.load(args.vector, map_location="cpu"), layer_idx=args.layer)
    cfg = PROMPT_CONFIGS[args.prompt_type]
    suffix = cfg["pos"] if args.condition == "positive" else cfg["neg"] if args.condition == "negative" else ""
    gold = load_gold(args.gold_jsonl)

    rows = []
    modes = parse_interface_modes(args.interface_modes or args.interface_mode)
    for task, seed in read_plan(args.plan):
        env = make_miniwob_env(task)
        try:
            obs, _ = env.reset(seed=seed)
            base_prompt = build_prompt(obs, args.max_elems)
            for mode in modes:
                prompt, transform = apply_interface_variant(base_prompt, mode, seed)
                if suffix:
                    prompt = f"{prompt}\n{suffix}"
                action = model.generate(prompt, steer=args.condition == "steer", max_new_tokens=args.max_new_tokens).strip()
                real_gold = gold.get((task, seed))
                shown_gold = transform.real_to_shown.get(real_gold, real_gold) if real_gold else None
                metrics = action_metrics(action, gold_id=shown_gold, transform=transform)
                rows.append({
                    "task": task,
                    "seed": seed,
                    "condition": args.condition,
                    "interface_mode": mode,
                    "remap_mode": mode,
                    "action": action,
                    **metrics,
                })
        finally:
            env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {
        k: sum(bool(r.get(k)) for r in rows)
        for k in [
            "parse_valid",
            "action_type_valid",
            "valid_current_id",
            "invalid_bid",
            "copied_example_id",
            "stale_id",
            "label_as_id",
            "bogus_argument",
            "gold_id_match",
        ]
    }
    summary["episodes"] = len(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
