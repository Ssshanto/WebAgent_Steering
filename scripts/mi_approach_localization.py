#!/usr/bin/env python3
"""MI approach 0.1/0.2: layer localization via inference-time ablation scan."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


EXPLORE_TASKS = [
    "ascending-numbers",
    "bisect-angle",
    "choose-date-medium",
    "click-button",
    "click-checkboxes",
    "click-color",
    "click-dialog",
    "click-link",
    "click-option",
    "copy-paste",
    "count-shape",
    "daily-calendar",
    "drag-items",
    "drag-shapes",
    "draw-circle",
    "email-inbox",
    "enter-text",
    "enter-time",
    "find-word",
    "form-sequence",
    "login-user",
    "navigate-tree",
    "read-table",
    "search-engine",
    "simple-arithmetic",
    "social-media",
    "terminal",
    "text-editor",
    "use-autocomplete",
    "use-slider",
]


def run_and_log(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def write_explore_manifest(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": "auto",
        "source": "scripts/mi_approach_localization.py",
        "task_count": len(EXPLORE_TASKS),
        "tasks": EXPLORE_TASKS,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_summary(summary_path):
    rows = []
    with summary_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "layer": int(row["layer"]),
                        "scale": float(row["scale"]),
                        "delta": float(row["delta"]),
                        "base_acc": float(row["base_acc"]),
                        "ablated_acc": float(row["ablated_acc"]),
                        "base_parse_fail": float(row["base_parse_fail"]),
                        "ablated_parse_fail": float(row["ablated_parse_fail"]),
                        "base_a_err": float(row["base_a_err"]),
                        "ablated_a_err": float(row["ablated_a_err"]),
                        "base_g_err": float(row["base_g_err"]),
                        "ablated_g_err": float(row["ablated_g_err"]),
                        "base_s_err": float(row["base_s_err"]),
                        "ablated_s_err": float(row["ablated_s_err"]),
                        "output": row["output"],
                    }
                )
            except Exception:
                continue
    return rows


def select_top_layer(rows):
    candidates = [r for r in rows if r["scale"] <= 0.5]
    if not candidates:
        candidates = list(rows)
    if not candidates:
        return None
    # Most disruptive (minimum delta) under stronger ablation.
    candidates.sort(key=lambda r: (r["delta"], r["scale"]))
    return candidates[0]


def main():
    ap = argparse.ArgumentParser(description="MI localization scan for qwen3-1.7b")
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--layers", default="13-15")
    ap.add_argument("--scales", default="1.0,0.5,0.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict-action-prompt", action="store_true")
    ap.add_argument(
        "--explore-manifest",
        default="runtime_state/miniwob_explore_manifest.json",
    )
    ap.add_argument(
        "--out-json",
        default="runtime_state/mi_localization_17b.json",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()
    py = Path(sys.executable)
    explore_manifest = root / args.explore_manifest
    write_explore_manifest(explore_manifest)

    base_dir = root / "results/mi_17b/localization_base"
    scan_dir = root / "results/mi_17b/localization_scan"
    logs = root / "logs/mi_17b"

    base_cmd = [
        str(py),
        "scripts/mi_layer_ablation.py",
        "--model",
        args.model,
        "--layers",
        "14",
        "--scales",
        "1.0",
        "--tasks",
        "all",
        "--task-manifest",
        str(explore_manifest.relative_to(root)),
        "--base-only",
        "--out-dir",
        str(base_dir.relative_to(root)),
        "--quiet",
        "--no-progress",
    ]
    if args.strict_action_prompt:
        base_cmd.append("--strict-action-prompt")

    run_and_log(base_cmd, logs / "localization_base.log")

    base_jsonl = base_dir / "qwen3-1.7b_L14_scale1.jsonl"
    scan_cmd = [
        str(py),
        "scripts/mi_layer_ablation.py",
        "--model",
        args.model,
        "--layers",
        args.layers,
        "--scales",
        args.scales,
        "--tasks",
        "all",
        "--task-manifest",
        str(explore_manifest.relative_to(root)),
        "--steer-only",
        "--base-jsonl",
        str(base_jsonl.relative_to(root)),
        "--out-dir",
        str(scan_dir.relative_to(root)),
        "--seed",
        str(args.seed),
        "--quiet",
        "--no-progress",
    ]
    if args.strict_action_prompt:
        scan_cmd.append("--strict-action-prompt")

    run_and_log(scan_cmd, logs / "localization_scan.log")

    summary_rows = read_summary(scan_dir / "mi_layer_ablation_summary.tsv")
    top = select_top_layer(summary_rows)
    if top is None:
        raise RuntimeError("Localization summary is empty")

    payload = {
        "model": args.model,
        "seed": args.seed,
        "explore_manifest": str(explore_manifest),
        "base_jsonl": str(base_jsonl),
        "scan_summary": str(scan_dir / "mi_layer_ablation_summary.tsv"),
        "selected_layer": int(top["layer"]),
        "selected_scale": float(top["scale"]),
        "selected_delta": float(top["delta"]),
        "rows": summary_rows,
    }

    out_json = root / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.quiet:
        print(
            json.dumps(
                {
                    "out_json": str(out_json),
                    "selected_layer": payload["selected_layer"],
                    "selected_delta": payload["selected_delta"],
                }
            )
        )
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
