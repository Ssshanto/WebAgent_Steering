#!/usr/bin/env python3
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


MODELS = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b"]


def is_jsonl_valid(path: Path) -> tuple[bool, int]:
    if not path.exists() or path.stat().st_size == 0:
        return False, 0
    count = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                json.loads(line)
                count += 1
    except Exception:
        return False, count
    return True, count


def git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "runtime_state" / "qwen3_layer_plan.json").read_text())
    out_root = root / "results" / "qwen3_sweep"
    manifest_path = root / "runtime_state" / "qwen3_bio_manifest.json"

    models = {}
    parse_failures = 0

    for model in MODELS:
        m6 = plan[model]["middle6"]
        m1 = plan[model]["middle1"][0]
        d = out_root / model

        base_jsonl = d / f"{model}_L{m1}_a0.jsonl"
        base_summary = d / "baseline_summary.tsv"

        first_m6 = m6[0]
        steer_jsonl = d / f"{model}_L{first_m6}_a3.jsonl"
        steer_summary = d / "steer_summary.tsv"

        base_ok, base_rows = is_jsonl_valid(base_jsonl)
        steer_ok, steer_rows = is_jsonl_valid(steer_jsonl)

        parse_failures += int(not base_ok) + int(not steer_ok)

        models[model] = {
            "base": {
                "jsonl": str(base_jsonl),
                "summary": str(base_summary),
                "valid_jsonl": base_ok,
                "rows": base_rows,
                "complete": base_ok and base_summary.exists(),
            },
            "steer": {
                "jsonl": str(steer_jsonl),
                "summary": str(steer_summary),
                "valid_jsonl": steer_ok,
                "rows": steer_rows,
                "complete": steer_ok and steer_summary.exists(),
            },
            "status": "done"
            if (
                base_ok
                and base_summary.exists()
                and steer_ok
                and steer_summary.exists()
            )
            else "incomplete",
        }

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "git_sha": git_sha(root),
        "results_root": str(out_root),
        "models": models,
        "jsonl_parse_failures": parse_failures,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    print(json.dumps({"jsonl_parse_failures": parse_failures}, indent=2))


if __name__ == "__main__":
    main()
