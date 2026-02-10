#!/usr/bin/env python3
"""MI-first pipeline for qwen3-1.7b using separate hypothesis/localization/validation scripts."""

import argparse
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


def main():
    ap = argparse.ArgumentParser(description="Run MI pipeline (qwen3-1.7b only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict-action-prompt", action="store_true")
    ap.add_argument("--out-json", default="runtime_state/mi_pipeline_17b_result.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()
    py = Path(sys.executable)
    logs = root / "logs/mi_17b"

    hyp_json = root / "runtime_state/mi_hypotheses_17b.json"
    loc_json = root / "runtime_state/mi_localization_17b.json"
    val_json = root / "runtime_state/mi_validation_17b.json"

    cmd_h = [
        str(py),
        "scripts/mi_hypotheses.py",
        "--out-json",
        str(hyp_json.relative_to(root)),
    ]
    cmd_l = [
        str(py),
        "scripts/mi_approach_localization.py",
        "--model",
        "qwen3-1.7b",
        "--seed",
        str(args.seed),
        "--out-json",
        str(loc_json.relative_to(root)),
        "--quiet",
    ]
    cmd_v = [
        str(py),
        "scripts/mi_approach_validation.py",
        "--model",
        "qwen3-1.7b",
        "--seed",
        str(args.seed),
        "--layer-json",
        str(loc_json.relative_to(root)),
        "--out-json",
        str(val_json.relative_to(root)),
        "--quiet",
    ]
    if args.strict_action_prompt:
        cmd_l.append("--strict-action-prompt")
        cmd_v.append("--strict-action-prompt")

    run_and_log(cmd_h, logs / "pipeline_hypotheses.log")
    run_and_log(cmd_l, logs / "pipeline_localization.log")
    run_and_log(cmd_v, logs / "pipeline_validation.log")

    payload = {
        "model": "qwen3-1.7b",
        "seed": args.seed,
        "hypotheses_json": str(hyp_json),
        "localization_json": str(loc_json),
        "validation_json": str(val_json),
        "paper_report": str(root / "results/mi_17b/paper_report_full17.json"),
        "paper_examples": str(root / "results/mi_17b/paper_examples_full17.json"),
    }

    out = root / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.quiet:
        print(json.dumps({"out_json": str(out), "model": payload["model"]}))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
