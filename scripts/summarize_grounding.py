#!/usr/bin/env python3
"""Summarize normal, remap, or frozen grounding JSONL outputs."""

import argparse
import json
import sys
from collections import defaultdict

sys.path.insert(0, "scripts")

from grounding_utils import action_metrics, load_jsonl  # noqa: E402


def add(d, key, val):
    d[key] += int(bool(val))


def summarize_rows(rows):
    groups = defaultdict(lambda: defaultdict(int))
    metric_fields = [
        "parse_valid",
        "action_type_valid",
        "valid_current_id",
        "copied_example_id",
        "stale_id",
        "label_as_id",
        "bogus_argument",
        "invalid_current_id",
        "invalid_or_bogus_argument",
        "label_as_argument",
    ]
    for row in rows:
        task = row.get("task", "unknown")
        dataset = row.get("dataset", "")
        mode = row.get("interface_mode") or row.get("remap_mode") or ""
        condition = row.get("condition")
        if condition:
            key = (dataset, mode, condition, task)
            groups[key]["episodes"] += 1
            add(groups[key], "success", row.get("success"))
            groups[key]["reward_sum"] += float(row.get("total_reward", 0.0) or 0.0)
            add(groups[key], "parse_fail", row.get("error_episode") or row.get("error"))
            metrics = row.get("steps", [{}])[-1].get("shown_metrics") if row.get("steps") else row
            for name in metric_fields:
                add(groups[key], name, metrics.get(name))
        else:
            for prefix in ["base", "steer"]:
                if f"{prefix}_success" not in row:
                    continue
                key = (dataset, mode, prefix, task)
                groups[key]["episodes"] += 1
                add(groups[key], "success", row.get(f"{prefix}_success"))
                groups[key]["reward_sum"] += float(row.get(f"{prefix}_total_reward", 0.0) or 0.0)
                add(groups[key], "parse_fail", row.get(f"{prefix}_error_episode") or row.get(f"{prefix}_error"))
                metric = row.get(f"{prefix}_last_action_metrics")
                if not isinstance(metric, dict) or not metric:
                    metric = action_metrics(row.get(f"{prefix}_action"), [])
                for name in metric_fields:
                    add(groups[key], name, metric.get(name))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+")
    args = parser.parse_args()

    rows = []
    for path in args.jsonl:
        rows.extend(load_jsonl(path))
    groups = summarize_rows(rows)
    fields = [
        "dataset",
        "interface_mode",
        "condition",
        "task",
        "episodes",
        "success",
        "avg_reward",
        "parse_fail",
        "parse_valid",
        "action_type_valid",
        "valid_current_id",
        "copied_example_id",
        "stale_id",
        "label_as_id",
        "bogus_argument",
        "invalid_current_id",
        "invalid_or_bogus_argument",
        "label_as_argument",
    ]
    print("\t".join(fields))
    for (dataset, mode, condition, task), vals in sorted(groups.items()):
        n = max(1, vals["episodes"])
        out = [dataset, mode, condition, task, str(vals["episodes"])]
        out.append(f"{vals['success'] / n:.3f}")
        out.append(f"{vals['reward_sum'] / n:.3f}")
        out.extend(f"{vals[f] / n:.3f}" for f in fields[7:])
        print("\t".join(out))


if __name__ == "__main__":
    main()
