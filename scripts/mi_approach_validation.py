#!/usr/bin/env python3
"""MI approach 0.3/0.4: full-manifest validation on selected layer (qwen3-1.7b)."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run_and_log(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def read_summary(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["layer"] = int(row["layer"])
                row["scale"] = float(row["scale"])
                row["delta"] = float(row["delta"])
                row["base_acc"] = float(row["base_acc"])
                row["ablated_acc"] = float(row["ablated_acc"])
                row["base_parse_fail"] = float(row["base_parse_fail"])
                row["ablated_parse_fail"] = float(row["ablated_parse_fail"])
                rows.append(row)
            except Exception:
                continue
    return rows


def lookup(rows, scale):
    for r in rows:
        if abs(float(r["scale"]) - float(scale)) < 1e-9:
            return r
    return None


def main():
    ap = argparse.ArgumentParser(description="MI validation on full task manifest")
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument(
        "--layer-json",
        default="runtime_state/mi_localization_17b.json",
    )
    ap.add_argument(
        "--full-manifest",
        default="runtime_state/miniwob_full_manifest.json",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict-action-prompt", action="store_true")
    ap.add_argument("--out-json", default="runtime_state/mi_validation_17b.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()
    py = Path(sys.executable)

    layer = args.layer
    if layer is None:
        layer_data = json.loads((root / args.layer_json).read_text(encoding="utf-8"))
        layer = int(layer_data["selected_layer"])

    base_dir = root / "results/mi_17b/full_base"
    scan_dir = root / "results/mi_17b/full_scan"
    logs = root / "logs/mi_17b"

    base_cmd = [
        str(py),
        "scripts/mi_layer_ablation.py",
        "--model",
        args.model,
        "--layers",
        str(layer),
        "--scales",
        "1.0",
        "--tasks",
        "all",
        "--task-manifest",
        args.full_manifest,
        "--base-only",
        "--out-dir",
        str(base_dir.relative_to(root)),
        "--seed",
        str(args.seed),
        "--quiet",
        "--no-progress",
    ]
    if args.strict_action_prompt:
        base_cmd.append("--strict-action-prompt")
    run_and_log(base_cmd, logs / "validation_base.log")

    base_jsonl = base_dir / f"qwen3-1.7b_L{layer}_scale1.jsonl"
    scan_cmd = [
        str(py),
        "scripts/mi_layer_ablation.py",
        "--model",
        args.model,
        "--layers",
        str(layer),
        "--scales",
        "1.0,0.5,0.0",
        "--tasks",
        "all",
        "--task-manifest",
        args.full_manifest,
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
    run_and_log(scan_cmd, logs / "validation_scan.log")

    summary_rows = read_summary(scan_dir / "mi_layer_ablation_summary.tsv")
    r05 = lookup(summary_rows, 0.5)
    r00 = lookup(summary_rows, 0.0)

    jsonls = [
        scan_dir / f"qwen3-1.7b_L{layer}_scale0p5.jsonl",
        scan_dir / f"qwen3-1.7b_L{layer}_scale0.jsonl",
    ]
    report_cmd = [
        str(py),
        "scripts/paper_report.py",
        "--jsonl",
        ",".join(str(p.relative_to(root)) for p in jsonls),
        "--out-json",
        str((root / "results/mi_17b/paper_report_full17.json").relative_to(root)),
        "--examples-json",
        str((root / "results/mi_17b/paper_examples_full17.json").relative_to(root)),
    ]
    run_and_log(report_cmd, logs / "validation_report.log")

    payload = {
        "model": args.model,
        "seed": args.seed,
        "layer": int(layer),
        "base_jsonl": str(base_jsonl),
        "scan_summary": str(scan_dir / "mi_layer_ablation_summary.tsv"),
        "scale_0p5": r05,
        "scale_0p0": r00,
        "paper_report": "results/mi_17b/paper_report_full17.json",
        "paper_examples": "results/mi_17b/paper_examples_full17.json",
    }

    out_json = root / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.quiet:
        print(
            json.dumps(
                {
                    "out_json": str(out_json),
                    "layer": layer,
                    "scale_0p0_delta": (r00 or {}).get("delta"),
                }
            )
        )
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
