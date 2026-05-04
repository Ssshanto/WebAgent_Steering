#!/usr/bin/env python3
"""Summarize TALES frozen admissible-command JSONL outputs."""

import argparse
import json
from collections import defaultdict


FIELDS = [
    "parse_valid",
    "exact_admissible",
    "action_type_valid",
    "copied_stale_command",
    "copied_decoy_command",
    "stale_verb",
    "invented_command",
    "empty_output",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+")
    args = parser.parse_args()

    groups = defaultdict(lambda: defaultdict(int))
    for path in args.jsonl:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row.get("condition", ""), row.get("env_id", ""))
                groups[key]["n"] += 1
                for field in FIELDS:
                    groups[key][field] += int(bool(row.get(field)))

    print("\t".join(["condition", "env_id", "n", *FIELDS]))
    for (condition, env_id), vals in sorted(groups.items()):
        n = max(1, vals["n"])
        print(
            "\t".join(
                [
                    condition,
                    env_id,
                    str(vals["n"]),
                    *[f"{vals[field] / n:.3f}" for field in FIELDS],
                ]
            )
        )


if __name__ == "__main__":
    main()
