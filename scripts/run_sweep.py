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
    VLM_MODELS,
    SteeredModel,
    SteeredVLM,
    compute_vector,
    evaluate,
    load_base_jsonl,
    list_miniwob_tasks,
    get_layer,
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

    # Select tasks
    if args.tasks == "all":
        tasks = list_miniwob_tasks()
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    if args.base_only and args.steer_only:
        raise ValueError("Cannot set both --base-only and --steer-only")

    base_records = None
    if args.steer_only:
        if not args.base_jsonl:
            raise ValueError("--steer-only requires --base-jsonl")
        base_records = load_base_jsonl(args.base_jsonl)

    print(f"Tasks: {len(tasks)} selected")

    # Parse layers and alphas
    layers = parse_range_or_list(args.layers)
    alphas = parse_alpha_list(args.alphas)

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

    print(f"Total iterations: {len(iterations)}")

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize model once
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    is_vlm = args.model in VLM_MODELS

    print(f"Model: {args.model} ({MODEL_MAP[args.model]})")
    print(f"VLM mode: {is_vlm}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Vector method: {args.vector_method}")

    # Start with first layer to initialize model
    first_layer = layers[0]

    if is_vlm:
        model = SteeredVLM(
            MODEL_MAP[args.model],
            layer_idx=first_layer,
            coeff=alphas[0],  # Will update per run
            vector_method=args.vector_method,
        )
    else:
        model = SteeredModel(
            MODEL_MAP[args.model],
            layer_idx=first_layer,
            coeff=alphas[0],  # Will update per run
            vector_method=args.vector_method,
            model_key=args.model,
        )

    print("✓ Model loaded")

    # Compute or load steering vectors once
    print("\n" + "=" * 60)
    print("COMPUTING/LOADING STEERING VECTORS")
    print("=" * 60)

    if not args.base_only:
        # Determine cache directory for vectors
        cache_subdir = os.path.join(args.cache_dir, args.model, f"seed_{args.seed}")

        # Check if all required layer vectors are cached
        missing_layers = []
        for layer_idx in layers:
            cache_path = os.path.join(
                cache_subdir, f"{args.prompt_type}_L{layer_idx}.pt"
            )
            if not os.path.exists(cache_path):
                missing_layers.append(layer_idx)

        if missing_layers and not args.force_recompute:
            print(f"Cache miss for layers: {missing_layers}")
            print(f"Computing vectors for ALL layers...")
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
            )
        elif args.force_recompute:
            print("Force recomputing vectors...")
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
            )
        else:
            print(f"All required layers cached in {cache_subdir}")
            # Load all required vectors
            for layer_idx in layers:
                cache_path = os.path.join(
                    cache_subdir, f"{args.prompt_type}_L{layer_idx}.pt"
                )
                cached_vector = torch.load(cache_path, map_location="cpu")
                model.set_vector(cached_vector, layer_idx=layer_idx)
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
    print("\n" + "=" * 60)
    print("RUNNING SWEEP")
    print("=" * 60)

    results_list = []

    for run_idx, (layer_idx, alpha) in enumerate(iterations, 1):
        print(f"\n[{run_idx}/{len(iterations)}] Layer {layer_idx}, Alpha {alpha:.2f}")

        # Update model layer and coeff
        model.layer_idx = layer_idx
        model.coeff = alpha

        # Set the active vector for this layer
        if not args.base_only:
            vec = model.vectors.get(layer_idx)
            if vec is None:
                cache_path = os.path.join(
                    args.cache_dir,
                    args.model,
                    f"seed_{args.seed}",
                    f"{args.prompt_type}_L{layer_idx}.pt",
                )
                if os.path.exists(cache_path):
                    vec = torch.load(cache_path, map_location="cpu")
                else:
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
            args.eval_steps,
            args.max_elems,
            args.max_new_tokens,
            run_output_path,
            base_only=args.base_only,
            steer_only=args.steer_only,
            eval_seed=args.seed,
            base_records=base_records,
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

        print(f"  Base Acc:  {results['base_accuracy']:.1%}")
        if not args.base_only:
            print(f"  Steer Acc: {results['steer_accuracy']:.1%}")
            print(f"  Delta:     {results['improvement']:+.1%}")
        print(f"  Output:    {run_output_path}")

    summary_file.close()

    # Final summary
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
        default="response",
        help="Vector computation method",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=200,
        help="Training steps for vector computation",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=400,
        help="Evaluation steps per run",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Task list or 'all'",
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

    args = parser.parse_args()

    run_sweep(args)


if __name__ == "__main__":
    main()
