#!/usr/bin/env python3
"""
Sweep script for layer and alpha parameters.

Loads model once, computes/loads steering vectors once, then loops over
layers/alphas to call evaluate() and write per-run JSONL + summary TSV.

Key optimization: Model is loaded once and reused for all runs.
Vectors are computed/loaded once and cached in model.vectors dictionary.
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from miniwob_steer import (
    MODEL_MAP,
    SteeredModel,
    build_vector_cache_spec,
    compute_vector,
    evaluate,
    load_base_jsonl,
    resolve_tasks,
    validate_vector_cache_metadata,
    vector_cache_file,
)


def parse_range_or_list(s):
    """Parse range syntax like '0-23' or comma-separated list '0,5,10'."""
    parts = []
    for segment in s.split(","):
        segment = segment.strip()
        if not segment:
            continue
        if "-" in segment:
            start, end = segment.split("-")
            parts.extend(range(int(start), int(end) + 1))
        else:
            parts.append(int(segment))
    return sorted(set(parts))


def parse_alpha_list(s):
    """Parse comma-separated alpha values (floats)."""
    values = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def run_sweep(args):
    """Execute the sweep over layers and alphas."""

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Select tasks (frozen via manifest by default)
    tasks = resolve_tasks(args.tasks, task_manifest_path=args.task_manifest)

    if args.base_only and args.steer_only:
        raise ValueError("Cannot set both --base-only and --steer-only")

    base_records = None
    base_manifest = None
    if args.steer_only:
        if not args.base_jsonl:
            raise ValueError("--steer-only requires --base-jsonl")
        base_records, base_manifest = load_base_jsonl(args.base_jsonl)

    if not args.quiet:
        print(f"Tasks: {len(tasks)} selected")

    # Parse layers and alphas
    layers = parse_range_or_list(args.layers)
    alphas = parse_alpha_list(args.alphas)

    if not args.quiet:
        print(f"Layers: {layers}")
        print(f"Alphas: {alphas}")
        print(f"Order: {args.order}")

    # Determine order: "alpha" means alphas outer, "layer" means layers outer
    if args.order == "alpha":
        # Outer: alphas, Inner: layers
        iterations = [(l, a) for a in alphas for l in layers]
    else:
        # Outer: layers, Inner: alphas
        iterations = [(l, a) for l in layers for a in alphas]

    if args.base_only and len(iterations) > 1:
        first = iterations[0]
        if not args.quiet:
            print(
                "Base-only mode detected: collapsing sweep to a single run "
                f"(layer={first[0]}, alpha={first[1]})."
            )
        iterations = [first]

    if not args.quiet:
        print(f"Total iterations: {len(iterations)}")

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize model once
    if not args.quiet:
        print("\n" + "=" * 60)
        print("LOADING MODEL")
        print("=" * 60)
        print(f"Model: {args.model} ({MODEL_MAP[args.model]})")
        print(f"Prompt type: {args.prompt_type}")
        print(f"Vector method: {args.vector_method}")
        print(f"Action window: {args.action_window}")

    # Start with first layer to initialize model
    first_layer = layers[0]

    model = SteeredModel(
        MODEL_MAP[args.model],
        layer_idx=first_layer,
        coeff=alphas[0],  # Will update per run
        vector_method=args.vector_method,
        model_key=args.model,
        steer_action_window=args.action_window,
    )

    if not args.quiet:
        print("✓ Model loaded")

    # Compute or load steering vectors once
    if not args.quiet:
        print("\n" + "=" * 60)
        print("COMPUTING/LOADING STEERING VECTORS")
        print("=" * 60)

    cache_spec = None
    if not args.base_only:
        if args.random_control:
            hidden_size = int(model.model.config.hidden_size)
            g = torch.Generator(device="cpu")
            g.manual_seed(int(args.seed) + 1337)
            for layer_idx in layers:
                v = torch.randn(hidden_size, generator=g, dtype=torch.float32)
                v = v / v.norm(p=2).clamp(min=1e-8)
                model.set_vector(v, layer_idx=layer_idx)
            if not args.quiet:
                print(f"✓ Random-control vectors ready for layers: {layers}")
        else:
            cache_spec = build_vector_cache_spec(
                cache_dir=args.cache_dir,
                model_alias=args.model,
                seed=args.seed,
                prompt_type=args.prompt_type,
                vector_method=args.vector_method,
                train_steps=args.train_steps,
                max_elems=args.max_elems,
                max_new_tokens=args.max_new_tokens,
                strict_action_prompt=args.strict_action_prompt,
                tasks=tasks,
            )

            cache_valid = validate_vector_cache_metadata(cache_spec)
            missing_layers = []
            for layer_idx in layers:
                cache_path = vector_cache_file(cache_spec, layer_idx)
                if not os.path.exists(cache_path):
                    missing_layers.append(layer_idx)

            should_recompute = bool(
                args.force_recompute or (not cache_valid) or missing_layers
            )

            if should_recompute:
                if not args.quiet:
                    print(
                        "Computing vectors for ALL layers "
                        f"(force={args.force_recompute}, meta_ok={cache_valid}, missing={missing_layers})"
                    )
                compute_vector(
                    model,
                    tasks,
                    args.train_steps,
                    args.max_elems,
                    args.max_new_tokens,
                    args.prompt_type,
                    cache_dir=args.cache_dir,
                    model_alias=args.model,
                    seed=args.seed,
                    quiet=args.quiet,
                    show_progress=not args.no_progress and not args.quiet,
                    strict_action_prompt=args.strict_action_prompt,
                    cache_spec=cache_spec,
                )
            else:
                if not args.quiet:
                    print(
                        f"All required layers cached in {cache_spec['subdir']}"
                        f" (hash={cache_spec['config_hash']})"
                    )
                # Load all required vectors
                for layer_idx in layers:
                    cache_path = vector_cache_file(cache_spec, layer_idx)
                    cached_vector = torch.load(cache_path, map_location="cpu")
                    model.set_vector(cached_vector, layer_idx=layer_idx)
                if not args.quiet:
                    print(f"✓ Loaded vectors for layers: {layers}")

    # Summary TSV file
    summary_path = (
        args.summary_path
        if args.summary_path is not None
        else os.path.join(args.out_dir, "sweep_summary.tsv")
    )
    summary_file = open(summary_path, "w", newline="", encoding="utf-8")
    summary_writer = csv.DictWriter(
        summary_file,
        fieldnames=[
            "layer",
            "alpha",
            "base_acc",
            "steer_acc",
            "delta",
            "total_episodes",
        ],
    )
    summary_writer.writeheader()

    # Run sweep
    if not args.quiet:
        print("\n" + "=" * 60)
        print("RUNNING SWEEP")
        print("=" * 60)

    results_list = []

    for run_idx, (layer_idx, alpha) in enumerate(iterations, 1):
        if not args.quiet:
            print(
                f"\n[{run_idx}/{len(iterations)}] Layer {layer_idx}, Alpha {alpha:.2f}"
            )

        # Update model layer and coeff
        model.layer_idx = layer_idx
        model.coeff = alpha
        model._vector_cache.clear()

        # Set the active vector for this layer
        if not args.base_only:
            vec = model.vectors.get(layer_idx)
            if vec is None and (not args.random_control):
                cache_path = (
                    vector_cache_file(cache_spec, layer_idx)
                    if cache_spec is not None
                    else None
                )
                if cache_path and os.path.exists(cache_path):
                    vec = torch.load(cache_path, map_location="cpu")
                else:
                    if not args.quiet:
                        print(f"WARNING: No vector found for layer {layer_idx}")
                    vec = None
            if vec is not None:
                model.set_vector(vec, layer_idx=layer_idx)
            else:
                model.vector = None

        # Per-run output file (JSONL)
        alpha_tag = str(alpha).rstrip("0").rstrip(".")
        alpha_tag = alpha_tag.replace(".", "p")
        run_name = f"{args.model}_L{layer_idx}_a{alpha_tag}"
        run_output_path = os.path.join(args.out_dir, f"{run_name}.jsonl")

        # Evaluate
        results = evaluate(
            model,
            tasks,
            args.max_elems,
            args.max_new_tokens,
            run_output_path,
            base_only=args.base_only,
            steer_only=args.steer_only,
            eval_seed=args.seed,
            base_records=base_records,
            base_manifest=base_manifest,
            episode_steps=args.episode_steps,
            quiet=args.quiet,
            show_progress=not args.no_progress and not args.quiet,
            strict_action_prompt=args.strict_action_prompt,
            run_metadata={
                "entrypoint": "scripts/run_sweep.py",
                "model_alias": args.model,
                "model_name": MODEL_MAP[args.model],
                "layer": int(layer_idx),
                "alpha": float(alpha),
                "prompt_type": args.prompt_type,
                "vector_method": args.vector_method,
                "train_steps": int(args.train_steps),
                "episode_steps": int(args.episode_steps),
                "max_elems": int(args.max_elems),
                "max_new_tokens": int(args.max_new_tokens),
                "strict_action_prompt": bool(args.strict_action_prompt),
                "seed": int(args.seed),
                "task_manifest": args.task_manifest,
                "order": args.order,
                "random_control": bool(args.random_control),
                "base_only": bool(args.base_only),
                "steer_only": bool(args.steer_only),
                "base_jsonl": args.base_jsonl,
                "cache_hash": cache_spec["config_hash"] if cache_spec else None,
            },
        )

        # Append to summary
        summary_row = {
            "layer": layer_idx,
            "alpha": f"{alpha:.2f}",
            "base_acc": f"{results['base_accuracy']:.4f}",
            "steer_acc": f"{results['steer_accuracy']:.4f}"
            if not args.base_only
            else "N/A",
            "delta": f"{results['improvement']:+.4f}" if not args.base_only else "N/A",
            "total_episodes": results["total_episodes"],
        }
        summary_writer.writerow(summary_row)
        summary_file.flush()

        results_list.append(
            {
                "layer": layer_idx,
                "alpha": alpha,
                **results,
            }
        )

        if not args.quiet:
            print(f"  Base Acc:  {results['base_accuracy']:.1%}")
            if not args.base_only:
                print(f"  Steer Acc: {results['steer_accuracy']:.1%}")
                print(f"  Delta:     {results['improvement']:+.1%}")
            print(f"  Output:    {run_output_path}")

    summary_file.close()

    # Final summary
    if args.quiet:
        best = None
        if (not args.base_only) and results_list:
            best = max(results_list, key=lambda r: r["improvement"])
        print(
            json.dumps(
                {
                    "summary_tsv": summary_path,
                    "out_dir": args.out_dir,
                    "total_runs": len(results_list),
                    "best": {
                        "layer": best["layer"],
                        "alpha": best["alpha"],
                        "improvement": best["improvement"],
                    }
                    if best is not None
                    else None,
                }
            )
        )
        return

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"Summary TSV: {summary_path}")
    print(f"Per-run JSONL files: {args.out_dir}/layer*_alpha*.jsonl")
    print(f"Total runs: {len(results_list)}")

    # Print best result
    if not args.base_only:
        best = max(results_list, key=lambda r: r["improvement"])
        print(f"\nBest result:")
        print(f"  Layer: {best['layer']}, Alpha: {best['alpha']:.2f}")
        print(f"  Improvement: {best['improvement']:+.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep script for layer and alpha parameters"
    )

    parser.add_argument(
        "--model",
        choices=MODEL_MAP.keys(),
        default="0.5b",
        help="Model to use",
    )
    parser.add_argument(
        "--prompt-type",
        default="accuracy",
        help="Steering prompt type",
    )
    parser.add_argument(
        "--vector-method",
        choices=["response", "prompt"],
        default="prompt",
        help="Vector computation method",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=200,
        help="Training steps for vector computation",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=10,
        help="Maximum environment steps per episode",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Task list or 'all'",
    )
    parser.add_argument(
        "--task-manifest",
        default="runtime_state/miniwob_task_manifest.json",
        help="Path to frozen task manifest JSON",
    )
    parser.add_argument(
        "--layers",
        default="14",
        help="Layers to sweep (range: '0-23', list: '0,5,10')",
    )
    parser.add_argument(
        "--alphas",
        default="1.0, 2.0, 3.0",
        help="Alpha coefficients to sweep (comma-separated)",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--cache-dir",
        default="vectors",
        help="Directory to cache steering vectors",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--order",
        choices=["alpha", "layer"],
        default="alpha",
        help="Iteration order: 'alpha' (alphas outer) or 'layer' (layers outer)",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Evaluate baseline only",
    )
    parser.add_argument(
        "--steer-only",
        action="store_true",
        help="Evaluate steered only (requires --base-jsonl)",
    )
    parser.add_argument(
        "--base-jsonl",
        default=None,
        help="Path to baseline JSONL for steer-only mode",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--action-window",
        action="store_true",
        help="Apply steering after prefill (generation window only)",
    )
    parser.add_argument(
        "--max-elems",
        type=int,
        default=80,
        help="Max DOM elements in prompt",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of steering vectors",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional summary TSV path (defaults to out-dir/sweep_summary.tsv)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential logs; emit compact JSON summary",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars in delegated evaluation/vector computation",
    )
    parser.add_argument(
        "--strict-action-prompt",
        action="store_true",
        help="Use strict action-only prompt suffix to reduce verbose outputs",
    )
    parser.add_argument(
        "--random-control",
        action="store_true",
        help="Use random unit vectors instead of computed steering vectors",
    )

    args = parser.parse_args()

    run_sweep(args)


if __name__ == "__main__":
    main()
