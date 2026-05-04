#!/usr/bin/env python3
"""Compare baseline, prompt-only, and steered conditions on matched MiniWob episodes."""

import argparse
import re
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "src")

from miniwob_steer import (  # noqa: E402
    BROWSERGYM_BENCHMARK_REPEATS,
    BROWSERGYM_BENCHMARK_SEED,
    MODEL_MAP,
    PROMPT_CONFIGS,
    SEED_MODES,
    SteeredModel,
    build_episode_plan,
    compute_vector,
    get_layer,
    make_miniwob_env,
    parse_tasks_arg,
    build_prompt,
)


def run_episode(model, task, seed, episode_steps, max_elems, max_new_tokens, suffix, steer):
    env = make_miniwob_env(task)
    try:
        obs, _ = env.reset(seed=seed)
        success = False
        last_error = ""
        last_action = ""
        actions = []
        outputs = []
        total_reward = 0.0

        for _ in range(episode_steps):
            prompt = build_prompt(obs, max_elems)
            if suffix:
                prompt = f"{prompt}\n{suffix}"

            action = model.generate(
                prompt,
                steer=steer,
                max_new_tokens=max_new_tokens,
            ).strip()
            last_action = action
            actions.append(action)
            outputs.append(action)

            try:
                obs, reward, terminated, truncated, _info = env.step(action)
                reward = float(reward)
                total_reward += reward
                success = success or (reward > 0)
                last_error = (
                    str(obs.get("last_action_error", "") or "")
                    if isinstance(obs, dict)
                    else ""
                )
                if terminated or truncated:
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break

        return {
            "task": task,
            "seed": seed,
            "success": bool(success),
            "total_reward": total_reward,
            "error": last_error,
            "action": last_action,
            "actions": actions,
            "outputs": outputs,
            "steps": len(actions),
        }
    finally:
        env.close()


def summarize(rows):
    return {
        "accuracy": sum(int(row["success"]) for row in rows) / max(1, len(rows)),
        "parse_fail": sum(int(bool(row["error"])) for row in rows) / max(1, len(rows)),
    }


def parse_plan_arg(plan_arg):
    plan = []
    for item in plan_arg.split(","):
        task, seed_text = item.rsplit(":", 1)
        plan.append((task, int(seed_text)))
    return plan


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline, negative prompt, positive prompt, and steering."
    )
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="qwen3-0.5b")
    parser.add_argument("--prompt-type", choices=sorted(PROMPT_CONFIGS), default="accuracy")
    parser.add_argument("--layer", default="auto", help="Intervention layer (int or 'auto')")
    parser.add_argument("--coeff", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-mode",
        choices=sorted(SEED_MODES),
        default="project",
    )
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument(
        "--plan",
        help="Explicit comma-separated task:seed pairs; overrides --episodes-per-task evaluation plan.",
    )
    parser.add_argument("--episode-steps", type=int, default=10)
    parser.add_argument("--tasks", default="all", help="Task list or 'all'")
    parser.add_argument("--max-elems", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--vector-method", choices=["response", "prompt"], default="response")
    parser.add_argument("--cache-dir", default="vectors")
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
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--out",
        default="results/compare_default_conditions.json",
        help="Summary JSON output path",
    )
    args = parser.parse_args()

    if args.seed_mode == "browsergym":
        if args.seed == 0:
            args.seed = BROWSERGYM_BENCHMARK_SEED
        if args.episodes_per_task == 3:
            args.episodes_per_task = BROWSERGYM_BENCHMARK_REPEATS

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = parse_tasks_arg(args.tasks)
    layer = get_layer(args.model, args.layer)
    plan = (
        parse_plan_arg(args.plan)
        if args.plan
        else build_episode_plan(tasks, args.episodes_per_task, args.seed, args.seed_mode)
    )

    model = SteeredModel(
        MODEL_MAP[args.model],
        layer_idx=layer,
        coeff=args.coeff,
        vector_method=args.vector_method,
        model_key=args.model,
        steer_action_window=False,
        steer_position=args.steer_position,
        steer_all_layers=args.all_layers,
    )

    cache_path = (
        Path(args.cache_dir) / args.model / f"seed_{args.seed}" / f"{args.prompt_type}_L{layer}.pt"
    )
    if args.all_layers and cache_path.parent.exists() and not args.force_recompute:
        loaded = 0
        pattern = re.compile(rf"^{re.escape(args.prompt_type)}_L(\d+)\.pt$")
        for path in sorted(cache_path.parent.glob(f"{args.prompt_type}_L*.pt")):
            match = pattern.match(path.name)
            if not match:
                continue
            model.set_vector(torch.load(path, map_location="cpu"), layer_idx=int(match.group(1)))
            loaded += 1
        if loaded == 0:
            raise RuntimeError(f"no cached layer vectors found in {cache_path.parent}")
        print(f"loaded {loaded} cached layer vectors from {cache_path.parent}", flush=True)
    elif cache_path.exists() and not args.force_recompute:
        print(f"loading cached vector: {cache_path}", flush=True)
        model.set_vector(torch.load(cache_path, map_location="cpu"), layer_idx=layer)
    else:
        print("computing steering vector", flush=True)
        compute_vector(
            model,
            tasks,
            args.train_steps,
            args.max_elems,
            args.max_new_tokens,
            args.prompt_type,
            cache_dir=args.cache_dir,
            model_alias=args.model,
            seed=args.seed,
            seed_mode=args.seed_mode,
        )
        if model.vector is None:
            raise RuntimeError("vector compute did not populate target layer")

    pos_suffix = PROMPT_CONFIGS[args.prompt_type]["pos"]
    neg_suffix = PROMPT_CONFIGS[args.prompt_type]["neg"]
    results = {
        "baseline": [],
        "negative_prompt": [],
        "positive_prompt": [],
        "steering": [],
    }

    for idx, (task, episode_seed) in enumerate(
        tqdm(plan, desc="matched episodes x 4 conditions"),
        1,
    ):
        results["baseline"].append(
            run_episode(
                model,
                task,
                episode_seed,
                args.episode_steps,
                args.max_elems,
                args.max_new_tokens,
                suffix=None,
                steer=False,
            )
        )
        results["negative_prompt"].append(
            run_episode(
                model,
                task,
                episode_seed,
                args.episode_steps,
                args.max_elems,
                args.max_new_tokens,
                suffix=neg_suffix,
                steer=False,
            )
        )
        results["positive_prompt"].append(
            run_episode(
                model,
                task,
                episode_seed,
                args.episode_steps,
                args.max_elems,
                args.max_new_tokens,
                suffix=pos_suffix,
                steer=False,
            )
        )
        results["steering"].append(
            run_episode(
                model,
                task,
                episode_seed,
                args.episode_steps,
                args.max_elems,
                args.max_new_tokens,
                suffix=None,
                steer=True,
            )
        )

        if idx % 25 == 0 or idx == len(plan):
            partial = {name: summarize(rows) for name, rows in results.items()}
            print(json.dumps({"done": idx, "total": len(plan), "metrics": partial}), flush=True)

    report = {
        "model": args.model,
        "model_name": MODEL_MAP[args.model],
        "prompt_type": args.prompt_type,
        "layer": layer,
        "coeff": args.coeff,
        "seed": args.seed,
        "tasks": tasks,
        "plan": plan,
        "episodes_per_task": args.episodes_per_task,
        "total_episodes": len(plan),
        "vector_method": args.vector_method,
        "steer_position": args.steer_position,
        "all_layers": args.all_layers,
        "seed_mode": args.seed_mode,
        "metrics": {name: summarize(rows) for name, rows in results.items()},
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("FINAL_REPORT")
    print(json.dumps(report["metrics"], indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
