#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path

TOY_10_TASKS = [
    "click-button",
    "click-link",
    "click-dialog",
    "click-dialog-2",
    "click-tab",
    "click-widget",
    "focus-text",
    "sign-agreement",
    "enter-time",
    "click-collapsible-2-nodelay",
]


def load_task_stats(base_jsonl: Path):
    stats = defaultdict(
        lambda: {"episodes": 0, "success": 0, "parse_fail": 0, "intercept": 0}
    )
    with base_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = row["task"]
            stats[task]["episodes"] += 1
            stats[task]["success"] += int(bool(row.get("base_success")))
            stats[task]["parse_fail"] += int(bool(str(row.get("base_error", "") or "")))
            stats[task]["intercept"] += int(
                bool(row.get("base_click_intercept", False))
            )

    out = {}
    for task, v in stats.items():
        n = max(1, v["episodes"])
        out[task] = {
            "episodes": v["episodes"],
            "success_rate": v["success"] / n,
            "parse_fail_rate": v["parse_fail"] / n,
            "click_intercept_rate": v["intercept"] / n,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-jsonl", required=True)
    args = ap.parse_args()

    stats = load_task_stats(Path(args.base_jsonl))

    print("Selected toy subset (10 tasks):")
    print(",".join(TOY_10_TASKS))
    print("\nPer-task baseline stats:")
    for task in TOY_10_TASKS:
        s = stats.get(task)
        if s is None:
            print(f"- {task}: MISSING_IN_BASE_JSONL")
            continue
        print(
            f"- {task}: n={s['episodes']} succ={s['success_rate']:.3f} "
            f"parse={s['parse_fail_rate']:.3f} intercept={s['click_intercept_rate']:.3f}"
        )

    print("\nRun-sweep --tasks value:")
    print(",".join(TOY_10_TASKS))


if __name__ == "__main__":
    main()
