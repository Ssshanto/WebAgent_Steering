#!/usr/bin/env python3
"""Layer ablation scan for mechanistic interpretability (inference-time only)."""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from miniwob_steer import MODEL_MAP, SteeredModel, evaluate, resolve_tasks


def parse_range_or_list(spec):
    spec = str(spec).strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_scales(spec):
    return [float(x.strip()) for x in str(spec).split(",") if x.strip()]


def scale_tag(value):
    s = f"{value:.3f}".rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    return s.replace(".", "p")


def run_scan(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = resolve_tasks(args.tasks, task_manifest_path=args.task_manifest)
    layers = parse_range_or_list(args.layers)
    scales = parse_scales(args.scales)

    if not layers:
        raise ValueError("No layers parsed from --layers")
    if not scales:
        raise ValueError("No scales parsed from --scales")

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = args.summary_path or os.path.join(
        args.out_dir, "mi_layer_ablation_summary.tsv"
    )

    model = SteeredModel(
        MODEL_MAP[args.model],
        layer_idx=layers[0],
        coeff=0.0,
        vector_method="prompt",
        model_key=args.model,
        steer_action_window=args.action_window,
    )

    summary_file = open(summary_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        summary_file,
        fieldnames=[
            "layer",
            "scale",
            "base_acc",
            "ablated_acc",
            "delta",
            "base_parse_fail",
            "ablated_parse_fail",
            "base_a_err",
            "ablated_a_err",
            "base_g_err",
            "ablated_g_err",
            "base_s_err",
            "ablated_s_err",
            "total_episodes",
            "output",
        ],
    )
    writer.writeheader()

    results_list = []
    for layer in layers:
        for scale in scales:
            model.layer_idx = int(layer)
            model.intervention_mode = "scale"
            model.intervention_scale = float(scale)
            model.vector = None
            model.vectors.clear()
            model._vector_cache.clear()

            out_jsonl = os.path.join(
                args.out_dir,
                f"{args.model}_L{int(layer)}_scale{scale_tag(scale)}.jsonl",
            )

            res = evaluate(
                model,
                tasks,
                args.max_elems,
                args.max_new_tokens,
                out_jsonl,
                base_only=False,
                steer_only=False,
                eval_seed=args.seed,
                episode_steps=args.episode_steps,
                quiet=args.quiet,
                show_progress=not args.no_progress and not args.quiet,
                strict_action_prompt=args.strict_action_prompt,
                run_metadata={
                    "entrypoint": "scripts/mi_layer_ablation.py",
                    "model_alias": args.model,
                    "model_name": MODEL_MAP[args.model],
                    "layer": int(layer),
                    "intervention_mode": "scale",
                    "intervention_scale": float(scale),
                    "seed": int(args.seed),
                    "episode_steps": int(args.episode_steps),
                    "max_elems": int(args.max_elems),
                    "max_new_tokens": int(args.max_new_tokens),
                    "strict_action_prompt": bool(args.strict_action_prompt),
                    "task_manifest": args.task_manifest,
                },
            )

            row = {
                "layer": int(layer),
                "scale": float(scale),
                "base_acc": f"{res['base_accuracy']:.4f}",
                "ablated_acc": f"{res['steer_accuracy']:.4f}",
                "delta": f"{res['improvement']:+.4f}",
                "base_parse_fail": f"{res['base_parse_fail']:.4f}",
                "ablated_parse_fail": f"{res['steer_parse_fail']:.4f}",
                "base_a_err": f"{res['base_action_type_error_episode_rate']:.4f}",
                "ablated_a_err": f"{res['steer_action_type_error_episode_rate']:.4f}",
                "base_g_err": f"{res['base_bid_grounding_error_episode_rate']:.4f}",
                "ablated_g_err": f"{res['steer_bid_grounding_error_episode_rate']:.4f}",
                "base_s_err": f"{res['base_syntax_error_episode_rate']:.4f}",
                "ablated_s_err": f"{res['steer_syntax_error_episode_rate']:.4f}",
                "total_episodes": int(res["total_episodes"]),
                "output": out_jsonl,
            }
            writer.writerow(row)
            summary_file.flush()
            results_list.append(row)

    summary_file.close()

    if args.quiet:
        best = (
            min(results_list, key=lambda r: float(r["delta"])) if results_list else None
        )
        print(
            json.dumps(
                {
                    "summary_tsv": summary_path,
                    "runs": len(results_list),
                    "best_disruption": best,
                }
            )
        )
    else:
        print(f"Summary TSV: {summary_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Inference-time layer ablation scan for MI localization"
    )
    ap.add_argument("--model", choices=MODEL_MAP.keys(), default="qwen3-1.7b")
    ap.add_argument("--layers", default="14", help="Layer range/list, e.g. 12-18")
    ap.add_argument("--scales", default="0.0,0.5,1.0", help="Comma-separated scales")
    ap.add_argument("--tasks", default="all")
    ap.add_argument(
        "--task-manifest",
        default="runtime_state/miniwob_task_manifest.json",
        help="Path to task manifest JSON",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-steps", type=int, default=10)
    ap.add_argument("--max-elems", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--out-dir", default="results/mi_layer_ablation")
    ap.add_argument("--summary-path", default=None)
    ap.add_argument("--action-window", action="store_true")
    ap.add_argument("--strict-action-prompt", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()
