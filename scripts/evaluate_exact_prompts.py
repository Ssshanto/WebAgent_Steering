#!/usr/bin/env python3
"""Evaluate baseline/positive/negative prompt suffixes on explicit MiniWob seeds."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from miniwob_steer import (  # noqa: E402
    MODEL_MAP,
    PROMPT_CONFIGS,
    SteeredModel,
    build_prompt,
    make_miniwob_env,
)


def run_episode(model, task, seed, suffix, episode_steps, max_elems, max_new_tokens):
    env = make_miniwob_env(task)
    try:
        obs, _ = env.reset(seed=seed)
        success = False
        total_reward = 0.0
        actions = []
        errors = []

        for _ in range(episode_steps):
            prompt = build_prompt(obs, max_elems)
            if suffix:
                prompt = f"{prompt}\n{suffix}"
            action = model.generate(prompt, steer=False, max_new_tokens=max_new_tokens)
            action = action.strip()
            actions.append(action)

            try:
                obs, reward, terminated, truncated, _info = env.step(action)
                reward = float(reward)
                total_reward += reward
                success = success or reward > 0
                error = (
                    str(obs.get("last_action_error", "") or "")
                    if isinstance(obs, dict)
                    else ""
                )
                errors.append(error)
                if terminated or truncated:
                    break
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                break

        return {
            "task": task,
            "seed": seed,
            "success": bool(success),
            "total_reward": total_reward,
            "action": actions[-1] if actions else "",
            "actions": actions,
            "error": next((err for err in reversed(errors) if err), ""),
        }
    finally:
        env.close()


def summarize(rows):
    return {
        "accuracy": sum(row["success"] for row in rows) / max(1, len(rows)),
        "parse_fail": sum(bool(row["error"]) for row in rows) / max(1, len(rows)),
        "episodes": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_MAP, required=True)
    parser.add_argument("--prompt-type", choices=PROMPT_CONFIGS, required=True)
    parser.add_argument(
        "--plan",
        required=True,
        help="Comma-separated task:seed pairs, e.g. click-button:123,focus-text:456",
    )
    parser.add_argument("--episode-steps", type=int, default=6)
    parser.add_argument("--max-elems", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    plan = []
    for item in args.plan.split(","):
        task, seed_text = item.rsplit(":", 1)
        plan.append((task, int(seed_text)))

    model = SteeredModel(
        MODEL_MAP[args.model],
        layer_idx=0,
        coeff=0.0,
        model_key=args.model,
    )
    cfg = PROMPT_CONFIGS[args.prompt_type]
    suffixes = {
        "baseline": None,
        "positive": cfg["pos"],
        "negative": cfg["neg"],
    }

    results = {name: [] for name in suffixes}
    for task, episode_seed in plan:
        for name, suffix in suffixes.items():
            results[name].append(
                run_episode(
                    model,
                    task,
                    episode_seed,
                    suffix,
                    args.episode_steps,
                    args.max_elems,
                    args.max_new_tokens,
                )
            )

    report = {
        "model": args.model,
        "prompt_type": args.prompt_type,
        "plan": plan,
        "positive_prompt": cfg["pos"],
        "negative_prompt": cfg["neg"],
        "metrics": {name: summarize(rows) for name, rows in results.items()},
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
