#!/usr/bin/env python3
"""Collect static TALES states with current admissible commands.

This script intentionally does not load an LLM.  It is meant to run under a
Python 3.12 environment with ``tale-suite`` installed, then feed JSONL states to
the existing steering/model environment.
"""

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import gymnasium as gym


def _load_tales():
    try:
        importlib.import_module("tales")
    except Exception as exc:
        raise RuntimeError(
            "Could not import TALES. Install with Python>=3.12: pip install tale-suite"
        ) from exc


def _registered_tales_envs(split, substrings):
    _load_tales()
    env_ids = sorted(
        env_spec.id
        for env_spec in gym.envs.registry.values()
        if "tales/" in env_spec.id
    )
    if split == "train":
        env_ids = [env_id for env_id in env_ids if "train" in env_id.lower()]
    elif split == "test":
        env_ids = [env_id for env_id in env_ids if "train" not in env_id.lower()]

    needles = [x.lower() for x in substrings if x]
    if needles:
        env_ids = [
            env_id
            for env_id in env_ids
            if any(needle in env_id.lower() for needle in needles)
        ]
    return env_ids


def _commands_from_info(info):
    if not isinstance(info, dict):
        return []
    for key in ("admissible_commands", "valid_actions", "valid_commands"):
        value = info.get(key)
        if value:
            return [str(cmd).strip() for cmd in value if str(cmd).strip()]
    return []


def _make_env(env_id):
    return gym.make(env_id, disable_env_checker=True, admissible_commands=True)


def collect(args):
    random.seed(args.seed)
    if args.envs:
        env_ids = [env_id.strip() for env_id in args.envs.split(",") if env_id.strip()]
        _load_tales()
    else:
        env_ids = _registered_tales_envs(
            args.split,
            [x.strip() for x in args.env_substrings.split(",") if x.strip()],
        )
    env_ids = env_ids[: args.max_envs] if args.max_envs else env_ids
    if not env_ids:
        raise RuntimeError("No TALES environments matched the requested filters")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    skipped = []
    with out.open("w", encoding="utf-8") as f:
        for env_id in env_ids:
            try:
                env = _make_env(env_id)
            except Exception as exc:
                skipped.append({"env_id": env_id, "stage": "make", "error": repr(exc)})
                continue

            try:
                for seed_offset in range(args.seeds_per_env):
                    env_seed = args.seed + seed_offset
                    try:
                        obs, info = env.reset(seed=env_seed)
                    except Exception as exc:
                        skipped.append(
                            {
                                "env_id": env_id,
                                "stage": "reset",
                                "seed": env_seed,
                                "error": repr(exc),
                            }
                        )
                        continue

                    stale_commands = []
                    for step_idx in range(args.max_steps_per_env):
                        commands = _commands_from_info(info)
                        if commands:
                            row = {
                                "suite": "tales",
                                "env_id": env_id,
                                "seed": env_seed,
                                "step": step_idx,
                                "observation": str(obs),
                                "admissible_commands": commands,
                                "stale_commands": [
                                    cmd for cmd in stale_commands if cmd not in set(commands)
                                ][: args.max_stale],
                                "reward": float(info.get("score", 0.0))
                                if isinstance(info, dict)
                                and isinstance(info.get("score"), (int, float))
                                else None,
                            }
                            f.write(json.dumps(row) + "\n")
                            rows += 1
                            if args.max_samples and rows >= args.max_samples:
                                return rows, skipped

                        if not commands:
                            break
                        action = random.choice(commands)
                        stale_commands = commands
                        try:
                            step_out = env.step(action)
                            if len(step_out) == 5:
                                obs, _reward, terminated, truncated, info = step_out
                            elif len(step_out) == 4:
                                obs, _reward, done, info = step_out
                                terminated = bool(done)
                                truncated = False
                            else:
                                raise ValueError(
                                    f"unexpected env.step return length {len(step_out)}"
                                )
                        except Exception as exc:
                            skipped.append(
                                {
                                    "env_id": env_id,
                                    "stage": "step",
                                    "seed": env_seed,
                                    "step": step_idx,
                                    "action": action,
                                    "error": repr(exc),
                                }
                            )
                            break
                        if terminated or truncated:
                            break
            finally:
                env.close()

    return rows, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--envs", default="")
    parser.add_argument(
        "--env-substrings",
        default="TextWorld,TextWorldExpress,TWCooking",
        help="Comma filters used when --envs is omitted.",
    )
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--max-envs", type=int, default=10)
    parser.add_argument("--seeds-per-env", type=int, default=1)
    parser.add_argument("--max-steps-per-env", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-stale", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows, skipped = collect(args)
    print(json.dumps({"rows": rows, "skipped": skipped[:20], "skipped_count": len(skipped)}, indent=2))
    if rows == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
