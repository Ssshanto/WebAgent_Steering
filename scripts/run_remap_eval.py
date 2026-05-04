#!/usr/bin/env python3
"""Evaluate prompt-side bid remapping while stepping real BrowserGym bids."""

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
    executable_action_from_shown,
    read_plan,
)
from miniwob_steer import MODEL_MAP, PROMPT_CONFIGS, SteeredModel, build_prompt, make_miniwob_env  # noqa: E402


def run_episode(model, task, seed, args, suffix):
    env = make_miniwob_env(task)
    try:
        obs, _ = env.reset(seed=seed)
        success = False
        total_reward = 0.0
        rows = []
        last_error = ""
        for step in range(args.episode_steps):
            prompt = build_prompt(obs, args.max_elems)
            shown_prompt, transform = apply_interface_variant(prompt, args.interface_mode, seed + step)
            if suffix:
                shown_prompt = f"{shown_prompt}\n{suffix}"
            shown_action = model.generate(
                shown_prompt,
                steer=args.condition == "steer",
                max_new_tokens=args.max_new_tokens,
            ).strip()
            real_action = executable_action_from_shown(shown_action, transform)
            shown_metric = action_metrics(shown_action, transform=transform)
            real_metric = action_metrics(real_action, transform.shown_to_real.values())
            try:
                obs, reward, terminated, truncated, _info = env.step(real_action)
                reward = float(reward)
                done = bool(terminated or truncated)
                error = str(obs.get("last_action_error", "") or "") if isinstance(obs, dict) else ""
            except Exception as exc:
                reward = 0.0
                done = True
                error = f"step_exception:{type(exc).__name__}"
            total_reward += reward
            success = success or reward > 0
            last_error = error or last_error
            rows.append({
                "step": step,
                "shown_action": shown_action,
                "real_action": real_action,
                "shown_metrics": shown_metric,
                "real_metrics": real_metric,
                "reward": reward,
                "error": error,
            })
            if done:
                break
        return {
            "task": task,
            "seed": seed,
            "condition": args.condition,
            "interface_mode": args.interface_mode,
            "remap_mode": args.interface_mode,
            "success": bool(success),
            "total_reward": total_reward,
            "error": last_error,
            "error_episode": bool(last_error),
            "action": rows[-1]["shown_action"] if rows else "",
            "real_action": rows[-1]["real_action"] if rows else "",
            "steps": rows,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_MAP, default="gemma-3-4b")
    parser.add_argument("--condition", choices=["baseline", "positive", "negative", "steer"], required=True)
    parser.add_argument("--prompt-type", choices=PROMPT_CONFIGS, default="gemma_tree_pos_minus_base")
    parser.add_argument("--interface-mode", choices=INTERFACE_MODES, default=None)
    parser.add_argument("--remap-mode", choices=INTERFACE_MODES, default=None, help="Deprecated alias for --interface-mode")
    parser.add_argument("--plan", required=True, help="plan text or path")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--vector", help="steering vector .pt for condition=steer")
    parser.add_argument("--episode-steps", type=int, default=6)
    parser.add_argument("--max-elems", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.condition == "steer" and not args.vector:
        parser.error("--condition steer requires --vector")
    args.interface_mode = args.interface_mode or args.remap_mode
    if not args.interface_mode:
        parser.error("one of --interface-mode or --remap-mode is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = SteeredModel(MODEL_MAP[args.model], args.layer, args.alpha, model_key=args.model, steer_position="last")
    if args.condition == "steer":
        model.set_vector(torch.load(args.vector, map_location="cpu"), layer_idx=args.layer)
    cfg = PROMPT_CONFIGS[args.prompt_type]
    suffix = cfg["pos"] if args.condition == "positive" else cfg["neg"] if args.condition == "negative" else ""

    rows = [run_episode(model, task, seed, args, suffix) for task, seed in read_plan(args.plan)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"episodes": len(rows), "success": sum(r["success"] for r in rows), "parse_fail": sum(r["error_episode"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
