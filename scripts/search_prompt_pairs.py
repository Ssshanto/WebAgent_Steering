#!/usr/bin/env python3
"""Screen positive/negative prompt pairs without computing CAA vectors."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from miniwob_steer import (  # noqa: E402
    BROWSERGYM_BENCHMARK_REPEATS,
    BROWSERGYM_BENCHMARK_SEED,
    MODEL_MAP,
    PROMPT_CONFIGS,
    SEED_MODES,
    SteeredModel,
    build_episode_plan,
    build_prompt,
    make_miniwob_env,
    parse_tasks_arg,
)


def run_episode(model, task, seed, episode_steps, max_elems, max_new_tokens, suffix):
    env = make_miniwob_env(task)
    try:
        obs, _ = env.reset(seed=seed)
        success = False
        last_error = ""
        actions = []
        outputs = []
        total_reward = 0.0

        for _ in range(episode_steps):
            prompt = build_prompt(obs, max_elems)
            if suffix:
                prompt = f"{prompt}\n{suffix}"

            output = model.generate(
                prompt,
                steer=False,
                max_new_tokens=max_new_tokens,
            ).strip()
            action = str(output or "").strip()
            outputs.append(output)
            actions.append(action)

            try:
                obs, reward, terminated, truncated, _info = env.step(action)
                reward = float(reward)
                total_reward += reward
                success = success or reward > 0
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
            "error_episode": bool(last_error),
            "steps": len(actions),
            "action": actions[-1] if actions else "",
            "actions": actions,
            "outputs": outputs,
        }
    finally:
        env.close()


def summarize(rows):
    return {
        "accuracy": sum(int(row["success"]) for row in rows) / max(1, len(rows)),
        "mean_reward": sum(float(row["total_reward"]) for row in rows) / max(1, len(rows)),
        "parse_fail": sum(int(row["error_episode"]) for row in rows) / max(1, len(rows)),
        "episodes": len(rows),
    }


def parse_prompt_types(raw):
    if raw == "all":
        return sorted(PROMPT_CONFIGS)
    prompt_types = [item.strip() for item in raw.split(",") if item.strip()]
    missing = [item for item in prompt_types if item not in PROMPT_CONFIGS]
    if missing:
        raise ValueError(f"Unknown prompt type(s): {missing}")
    return prompt_types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="0.5b")
    parser.add_argument("--prompt-types", default="action_syntax,valid_bid,webagent_discipline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-mode", choices=sorted(SEED_MODES), default="project")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=10)
    parser.add_argument("--tasks", default="click-button,click-link,click-option,choose-list,enter-text")
    parser.add_argument("--max-elems", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--out", default="results/prompt_pair_search.json")
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
    prompt_types = parse_prompt_types(args.prompt_types)
    plan = build_episode_plan(tasks, args.episodes_per_task, args.seed, args.seed_mode)

    model = SteeredModel(
        MODEL_MAP[args.model],
        layer_idx=0,
        coeff=0.0,
        vector_method="response",
        model_key=args.model,
    )

    baseline = []
    for task, episode_seed in tqdm(plan, desc="baseline"):
        baseline.append(
            run_episode(
                model,
                task,
                episode_seed,
                args.episode_steps,
                args.max_elems,
                args.max_new_tokens,
                suffix=None,
            )
        )

    prompt_results = {}
    for prompt_type in prompt_types:
        cfg = PROMPT_CONFIGS[prompt_type]
        prompt_results[prompt_type] = {"positive": [], "negative": []}
        for condition, suffix in (("positive", cfg["pos"]), ("negative", cfg["neg"])):
            rows = prompt_results[prompt_type][condition]
            for task, episode_seed in tqdm(plan, desc=f"{prompt_type}:{condition}"):
                rows.append(
                    run_episode(
                        model,
                        task,
                        episode_seed,
                        args.episode_steps,
                        args.max_elems,
                        args.max_new_tokens,
                        suffix=suffix,
                    )
                )

    metrics = {"baseline": summarize(baseline)}
    for prompt_type, rows_by_condition in prompt_results.items():
        metrics[prompt_type] = {
            condition: summarize(rows)
            for condition, rows in rows_by_condition.items()
        }
        pos = metrics[prompt_type]["positive"]["accuracy"]
        neg = metrics[prompt_type]["negative"]["accuracy"]
        base = metrics["baseline"]["accuracy"]
        metrics[prompt_type]["score"] = (pos - base) + (base - neg)

    report = {
        "model": args.model,
        "model_name": MODEL_MAP[args.model],
        "seed": args.seed,
        "seed_mode": args.seed_mode,
        "tasks": tasks,
        "episodes_per_task": args.episodes_per_task,
        "episode_steps": args.episode_steps,
        "prompt_types": prompt_types,
        "metrics": metrics,
        "baseline": baseline,
        "prompt_results": prompt_results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("FINAL_REPORT")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
