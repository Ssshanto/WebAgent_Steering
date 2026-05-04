#!/usr/bin/env python3
"""Evaluate several prompt pairs on explicit MiniWob task seeds with one model load."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from miniwob_steer import MODEL_MAP, PROMPT_CONFIGS, SteeredModel  # noqa: E402
from evaluate_exact_prompts import run_episode, summarize  # noqa: E402


def parse_prompt_types(raw):
    prompt_types = [item.strip() for item in raw.split(",") if item.strip()]
    missing = [item for item in prompt_types if item not in PROMPT_CONFIGS]
    if missing:
        raise ValueError(f"unknown prompt type(s): {missing}")
    return prompt_types


def parse_plan(raw):
    plan = []
    for item in raw.split(","):
        task, seed_text = item.rsplit(":", 1)
        plan.append((task, int(seed_text)))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_MAP, required=True)
    parser.add_argument("--prompt-types", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--episode-steps", type=int, default=6)
    parser.add_argument("--max-elems", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompt_types = parse_prompt_types(args.prompt_types)
    plan = parse_plan(args.plan)
    model = SteeredModel(
        MODEL_MAP[args.model],
        layer_idx=0,
        coeff=0.0,
        model_key=args.model,
    )

    baseline = []
    for task, episode_seed in plan:
        baseline.append(
            run_episode(
                model,
                task,
                episode_seed,
                None,
                args.episode_steps,
                args.max_elems,
                args.max_new_tokens,
            )
        )

    prompt_results = {}
    for prompt_type in prompt_types:
        cfg = PROMPT_CONFIGS[prompt_type]
        prompt_results[prompt_type] = {"positive": [], "negative": []}
        for condition, suffix in (("positive", cfg["pos"]), ("negative", cfg["neg"])):
            for task, episode_seed in plan:
                prompt_results[prompt_type][condition].append(
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

    metrics = {"baseline": summarize(baseline)}
    for prompt_type, rows_by_condition in prompt_results.items():
        metrics[prompt_type] = {
            condition: summarize(rows) for condition, rows in rows_by_condition.items()
        }

    report = {
        "model": args.model,
        "model_name": MODEL_MAP[args.model],
        "prompt_types": prompt_types,
        "plan": plan,
        "metrics": metrics,
        "baseline": baseline,
        "prompt_results": prompt_results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
