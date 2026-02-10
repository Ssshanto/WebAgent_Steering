#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_rows(rows):
    n = len(rows)
    if n == 0:
        return None

    base_rows = [r for r in rows if "base_success" in r]
    steer_rows = [r for r in rows if "steer_success" in r]

    base_n = len(base_rows)
    steer_n = len(steer_rows)
    has_steer = steer_n > 0

    base_s = sum(int(bool(r.get("base_success", False))) for r in base_rows)
    steer_s = sum(int(bool(r.get("steer_success", False))) for r in steer_rows)
    base_pf = sum(int(bool(r.get("base_error_episode", False))) for r in base_rows)
    steer_pf = sum(int(bool(r.get("steer_error_episode", False))) for r in steer_rows)

    def avg(rows_subset, key):
        vals = [float(r.get(key, 0.0) or 0.0) for r in rows_subset]
        return (sum(vals) / len(vals)) if vals else None

    return {
        "episodes": n,
        "base_total_episodes": base_n,
        "steer_total_episodes": steer_n,
        "base_success": base_s / max(1, base_n),
        "steer_success": (steer_s / max(1, steer_n)) if has_steer else None,
        "delta": ((steer_s / max(1, steer_n)) - (base_s / max(1, base_n)))
        if has_steer
        else None,
        "base_parse_fail": base_pf / max(1, base_n),
        "steer_parse_fail": (steer_pf / max(1, steer_n)) if has_steer else None,
        "base_action_type_error_episode_rate": avg(
            base_rows, "base_action_type_error_episode"
        ),
        "base_bid_grounding_error_episode_rate": avg(
            base_rows, "base_bid_grounding_error_episode"
        ),
        "base_syntax_error_episode_rate": avg(base_rows, "base_syntax_error_episode"),
        "steer_action_type_error_episode_rate": avg(
            steer_rows, "steer_action_type_error_episode"
        )
        if has_steer
        else None,
        "steer_bid_grounding_error_episode_rate": avg(
            steer_rows, "steer_bid_grounding_error_episode"
        )
        if has_steer
        else None,
        "steer_syntax_error_episode_rate": avg(steer_rows, "steer_syntax_error_episode")
        if has_steer
        else None,
    }


def pick_examples(rows, limit=5):
    flips = []
    degradations = []
    neutrals = []
    for r in rows:
        b = bool(r.get("base_success", False))
        s = bool(r.get("steer_success", False))
        ex = {
            "task": r.get("task"),
            "seed": r.get("seed"),
            "before": {
                "action": r.get("base_action", ""),
                "error": r.get("base_error", ""),
                "success": b,
            },
            "after": {
                "action": r.get("steer_action", ""),
                "error": r.get("steer_error", ""),
                "success": s,
            },
        }
        if (not b) and s:
            flips.append(ex)
        elif b and (not s):
            degradations.append(ex)
        else:
            neutrals.append(ex)

    out = {
        "improve_examples": flips[:limit],
        "degrade_examples": degradations[:limit],
        "neutral_examples": neutrals[:limit],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Comma-separated jsonl files")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--examples-json", required=True)
    ap.add_argument(
        "--random-control-jsonl", default="", help="Optional control jsonl files"
    )
    args = ap.parse_args()

    files = [Path(x.strip()) for x in args.jsonl.split(",") if x.strip()]
    control_files = [
        Path(x.strip()) for x in args.random_control_jsonl.split(",") if x.strip()
    ]

    report = []
    examples = {}
    control_delta = {}

    for p in files:
        rows = load_jsonl(p)
        s = summarize_rows(rows)
        if s is None:
            continue
        s["file"] = str(p)
        report.append(s)
        examples[str(p)] = pick_examples(rows)

    for p in control_files:
        rows = load_jsonl(p)
        s = summarize_rows(rows)
        if s is None:
            continue
        control_delta[str(p)] = s.get("delta")

    Path(args.out_json).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    Path(args.examples_json).write_text(
        json.dumps(examples, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({"report_rows": len(report), "control_rows": len(control_delta)}))


if __name__ == "__main__":
    main()
