#!/usr/bin/env python3
"""Preflight BrowserGym datasets before expensive CAA sweeps."""

import argparse
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from miniwob_steer import (  # noqa: E402
    DATASET_NAMES,
    DEFAULT_VECTOR_TASKS,
    import_browsergym_dataset,
    list_browsergym_tasks,
    make_browsergym_env,
)


def parse_dataset_list(text):
    datasets = [x.strip() for x in text.split(",") if x.strip()]
    unknown = sorted(set(datasets) - set(DATASET_NAMES))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return datasets


def pick_task(dataset, requested):
    if requested:
        return requested
    defaults = DEFAULT_VECTOR_TASKS.get(dataset, [])
    return defaults[0] if defaults else None


def preflight_dataset(dataset, task=None, verbose=False):
    print(f"\n[{dataset}] import")
    import_browsergym_dataset(dataset)
    tasks = list_browsergym_tasks(dataset)
    print(f"[{dataset}] registered tasks: {len(tasks)}")
    smoke_task = pick_task(dataset, task)
    if smoke_task is None:
        if not tasks:
            raise RuntimeError("No registered tasks found")
        smoke_task = tasks[0]
    if tasks and smoke_task not in tasks:
        print(f"[{dataset}] requested smoke task not in registry: {smoke_task}")
    print(f"[{dataset}] reset/step smoke task: {smoke_task}")
    env = make_browsergym_env(dataset, smoke_task)
    try:
        obs, _info = env.reset(seed=0)
        goal = obs.get("goal", "") if isinstance(obs, dict) else ""
        print(f"[{dataset}] reset ok; goal preview: {str(goal)[:120]}")
        _obs, reward, terminated, truncated, _info = env.step("noop()")
        print(
            f"[{dataset}] noop step ok; reward={float(reward):.3g}, "
            f"done={bool(terminated or truncated)}"
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="miniwob,webarena,workarena",
        help="Comma-separated dataset families to check.",
    )
    parser.add_argument("--miniwob-task", default=None)
    parser.add_argument("--webarena-task", default=None)
    parser.add_argument("--workarena-task", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"CUDA {idx}: {torch.cuda.get_device_name(idx)}")

    failures = []
    for dataset in parse_dataset_list(args.datasets):
        task = getattr(args, f"{dataset}_task")
        try:
            preflight_dataset(dataset, task=task, verbose=args.verbose)
        except Exception as exc:  # keep this script diagnostic, not exploratory
            print(f"[{dataset}] FAILED: {type(exc).__name__}: {exc}")
            if args.verbose:
                traceback.print_exc()
            failures.append(dataset)

    if failures:
        print(f"\nPreflight failed for: {','.join(failures)}")
        raise SystemExit(1)
    print("\nPreflight passed.")


if __name__ == "__main__":
    main()
