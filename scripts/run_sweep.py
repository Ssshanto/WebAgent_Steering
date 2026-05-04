#!/usr/bin/env python3
"""Layer/alpha sweep runner for CAA MiniWob experiments."""

import argparse
import csv
import json
import os
import random
import re
import sys

import numpy as np
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from miniwob_steer import (
    BROWSERGYM_BENCHMARK_REPEATS,
    BROWSERGYM_BENCHMARK_SEED,
    DATASET_NAMES,
    DEFAULT_VECTOR_TASKS,
    MODEL_MAP,
    SEED_MODES,
    SteeredModel,
    PROMPT_CONFIGS,
    compute_vector,
    evaluate,
    get_layer,
    load_base_jsonl,
    load_plan_file,
    measure_avg_residual_norms,
    parse_plan_arg,
    parse_tasks_arg,
    vector_cache_subdir,
)
from interface_variants import (  # noqa: E402
    INTERFACE_MODES,
    interface_cache_tag,
    parse_interface_modes,
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


def parse_alpha_pct_list(s):
    """Parse percentage strengths; accepts 5,10 or 0.05,0.10 forms."""
    values = []
    for value in parse_alpha_list(s):
        values.append(value / 100.0 if value > 1 else value)
    return values


def _load_optional_plan(path, text):
    if path:
        return load_plan_file(path)
    if text:
        return parse_plan_arg(text)
    return None


def run_sweep(args):
    """Execute the sweep over layers and alphas."""

    if args.seed_mode == "browsergym":
        if args.seed == 0:
            args.seed = BROWSERGYM_BENCHMARK_SEED
        if args.episodes_per_task == 3:
            args.episodes_per_task = BROWSERGYM_BENCHMARK_REPEATS

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    explicit_plan = _load_optional_plan(args.plan_file, args.plan)
    tasks = (
        sorted({task for task, _seed in explicit_plan})
        if explicit_plan is not None
        else parse_tasks_arg(args.tasks, dataset=args.dataset)
    )
    vector_dataset = args.vector_dataset or args.dataset
    vector_plan = _load_optional_plan(args.vector_plan_file, args.vector_plan)
    if vector_plan is not None:
        vector_tasks = sorted({task for task, _seed in vector_plan})
    elif args.vector_tasks:
        vector_tasks = parse_tasks_arg(args.vector_tasks, dataset=vector_dataset)
    elif vector_dataset == args.dataset:
        vector_tasks = tasks
    else:
        vector_tasks = DEFAULT_VECTOR_TASKS[vector_dataset]

    if args.base_only and args.steer_only:
        raise ValueError("Cannot set both --base-only and --steer-only")

    base_records = None
    if args.steer_only:
        if not args.base_jsonl:
            raise ValueError("--steer-only requires --base-jsonl")
        base_records = load_base_jsonl(args.base_jsonl)

    print(f"Dataset: {args.dataset}")
    print(f"Tasks: {len(tasks)} selected")
    print(f"Vector dataset: {vector_dataset}")
    print(f"Vector tasks: {len(vector_tasks)} selected")

    # Parse layers and alphas
    layers = parse_range_or_list(args.layers)
    raw_alphas = parse_alpha_list(args.alphas)
    alpha_pct_targets = parse_alpha_pct_list(args.alpha_pcts) if args.alpha_pcts else []
    interface_train_modes = parse_interface_modes(args.interface_train_modes)
    eval_modes = parse_interface_modes(args.interface_heldout_modes or args.interface_mode)
    vector_cache_tag = args.vector_cache_tag
    if vector_cache_tag is None and interface_train_modes != ["original"]:
        vector_cache_tag = interface_cache_tag(interface_train_modes)

    print(f"Layers: {layers}")
    print(f"Raw alphas: {raw_alphas}")
    if alpha_pct_targets:
        print(f"Alpha pct targets: {alpha_pct_targets}")
    print(f"Interface train modes: {interface_train_modes}")
    print(f"Interface eval modes: {eval_modes}")
    print(f"Order: {args.order}")

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize model once
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
        coeff=raw_alphas[0] if raw_alphas else 0.0,  # Will update per run
        vector_method=args.vector_method,
        model_key=args.model,
        steer_action_window=args.action_window,
        steer_position=args.steer_position,
        steer_all_layers=args.all_layers,
    )

    print("Model loaded")

    # Compute or load steering vectors once
    print("\n" + "=" * 60)
    print("COMPUTING/LOADING STEERING VECTORS")
    print("=" * 60)

    if not args.base_only:
        # Determine cache directory for vectors
        cache_subdir = vector_cache_subdir(
            args.cache_dir,
            args.model,
            args.seed,
            dataset=vector_dataset,
            cache_tag=vector_cache_tag,
        )

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
                vector_tasks,
                args.train_steps,
                args.max_elems,
                args.max_new_tokens,
                args.prompt_type,
                cache_dir=args.cache_dir,
                model_alias=args.model,
                seed=args.seed,
                seed_mode=args.seed_mode,
                dataset=vector_dataset,
                cache_tag=vector_cache_tag,
                interface_modes=interface_train_modes,
                training_plan=vector_plan,
            )
        elif args.force_recompute:
            print("Force recomputing vectors...")
            compute_vector(
                model,
                vector_tasks,
                args.train_steps,
                args.max_elems,
                args.max_new_tokens,
                args.prompt_type,
                cache_dir=args.cache_dir,
                model_alias=args.model,
                seed=args.seed,
                seed_mode=args.seed_mode,
                dataset=vector_dataset,
                cache_tag=vector_cache_tag,
                interface_modes=interface_train_modes,
                training_plan=vector_plan,
            )
        else:
            print(f"All required layers cached in {cache_subdir}")
            if args.all_layers:
                pattern = re.compile(rf"^{re.escape(args.prompt_type)}_L(\d+)\.pt$")
                loaded_layers = []
                for name in sorted(os.listdir(cache_subdir)):
                    match = pattern.match(name)
                    if not match:
                        continue
                    layer_idx = int(match.group(1))
                    cached_vector = torch.load(os.path.join(cache_subdir, name), map_location="cpu")
                    model.set_vector(cached_vector, layer_idx=layer_idx)
                    loaded_layers.append(layer_idx)
                print(f"Loaded vectors for all-layer steering: {loaded_layers}")
            else:
                for layer_idx in layers:
                    cache_path = os.path.join(
                        cache_subdir, f"{args.prompt_type}_L{layer_idx}.pt"
                    )
                    cached_vector = torch.load(cache_path, map_location="cpu")
                    model.set_vector(cached_vector, layer_idx=layer_idx)
                print(f"Loaded vectors for layers: {layers}")

    avg_resid_norms = {}
    if args.measure_resid_norm or alpha_pct_targets:
        print("\n" + "=" * 60)
        print("MEASURING RESIDUAL NORMS")
        print("=" * 60)
        avg_resid_norms = measure_avg_residual_norms(
            model,
            vector_tasks,
            args.train_steps,
            args.max_elems,
            layers,
            dataset=vector_dataset,
            seed=args.seed,
            seed_mode=args.seed_mode,
            training_plan=vector_plan,
            interface_modes=interface_train_modes,
        )
        print(f"Average residual norms: {avg_resid_norms}")
        norm_path = os.path.join(args.out_dir, "residual_norms.json")
        with open(norm_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "prompt_type": args.prompt_type,
                    "vector_dataset": vector_dataset,
                    "seed": args.seed,
                    "train_steps": args.train_steps,
                    "avg_resid_norms": avg_resid_norms,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    layer_alphas = {}
    for layer_idx in layers:
        items = [(alpha, "raw", None) for alpha in raw_alphas]
        avg_norm = avg_resid_norms.get(layer_idx)
        if alpha_pct_targets:
            if not avg_norm:
                raise RuntimeError(f"Missing residual norm for layer {layer_idx}")
            items.extend((pct * avg_norm, "pct", pct) for pct in alpha_pct_targets)
        layer_alphas[layer_idx] = items

    if args.order == "alpha":
        max_items = max((len(items) for items in layer_alphas.values()), default=0)
        iterations = []
        for item_idx in range(max_items):
            for layer_idx in layers:
                if item_idx < len(layer_alphas[layer_idx]):
                    iterations.append((layer_idx, *layer_alphas[layer_idx][item_idx]))
    else:
        iterations = [
            (layer_idx, alpha, alpha_source, alpha_pct_target)
            for layer_idx in layers
            for alpha, alpha_source, alpha_pct_target in layer_alphas[layer_idx]
        ]

    print(f"Total iterations: {len(iterations)}")

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
            "interface_mode",
            "alpha",
            "raw_alpha",
            "alpha_source",
            "avg_l17_resid_norm",
            "alpha_pct",
            "base_acc",
            "steer_acc",
            "delta",
            "base_parse_fail",
            "steer_parse_fail",
            "base_action_type_valid",
            "steer_action_type_valid",
            "base_valid_current_id",
            "steer_valid_current_id",
            "base_invalid_or_bogus_argument",
            "steer_invalid_or_bogus_argument",
            "base_copied_example_id",
            "steer_copied_example_id",
            "base_stale_id",
            "steer_stale_id",
            "base_label_as_id",
            "steer_label_as_id",
            "paired_gains",
            "paired_losses",
            "total_episodes",
        ],
    )
    summary_writer.writeheader()

    # Run sweep
    print("\n" + "=" * 60)
    print("RUNNING SWEEP")
    print("=" * 60)

    results_list = []

    for run_idx, (layer_idx, alpha, alpha_source, alpha_pct_target) in enumerate(iterations, 1):
        avg_norm = avg_resid_norms.get(layer_idx)
        alpha_pct = alpha / avg_norm if avg_norm else None
        print(f"\n[{run_idx}/{len(iterations)}] Layer {layer_idx}, Alpha {alpha:.2f}")

        # Update model layer and coeff
        model.layer_idx = layer_idx
        model.coeff = alpha

        # Set the active vector for this layer
        if not args.base_only:
            vec = model.vectors.get(layer_idx)
            if vec is None:
                cache_path = os.path.join(
                    vector_cache_subdir(
                        args.cache_dir,
                        args.model,
                        args.seed,
                        dataset=vector_dataset,
                        cache_tag=vector_cache_tag,
                    ),
                    f"{args.prompt_type}_L{layer_idx}.pt",
                )
                if os.path.exists(cache_path):
                    vec = torch.load(cache_path, map_location="cpu")
                else:
                    print(f"Missing vector for layer {layer_idx}")
                    vec = None
            if vec is not None:
                model.set_vector(vec, layer_idx=layer_idx)
            else:
                model.vector = None

        # Per-run output file (JSONL)
        alpha_tag = str(alpha).rstrip("0").rstrip(".")
        alpha_tag = alpha_tag.replace(".", "p")
        for interface_mode in eval_modes:
            run_name = f"{args.model}_L{layer_idx}_a{alpha_tag}_{interface_mode}"
            run_output_path = os.path.join(args.out_dir, f"{run_name}.jsonl")

            # Evaluate
            results = evaluate(
                model,
                tasks,
                args.episodes_per_task,
                args.max_elems,
                args.max_new_tokens,
                run_output_path,
                base_only=args.base_only,
                steer_only=args.steer_only,
                eval_seed=args.seed,
                base_records=base_records,
                episode_steps=args.episode_steps,
                seed_mode=args.seed_mode,
                episode_plan=explicit_plan,
                dataset=args.dataset,
                eval_instruction=PROMPT_CONFIGS[args.eval_instruction_type]["pos"]
                if args.eval_instruction_type
                else "",
                action_examples=not args.no_action_examples,
                interface_mode=interface_mode,
            )

            # Append to summary
            summary_row = {
                "layer": layer_idx,
                "interface_mode": interface_mode,
                "alpha": f"{alpha:.2f}",
                "raw_alpha": f"{alpha:.6g}",
                "alpha_source": alpha_source,
                "avg_l17_resid_norm": f"{avg_norm:.6g}" if avg_norm else "",
                "alpha_pct": f"{alpha_pct:.6g}" if alpha_pct is not None else "",
                "base_acc": f"{results['base_accuracy']:.4f}",
                "steer_acc": f"{results['steer_accuracy']:.4f}"
                if not args.base_only
                else "N/A",
                "delta": f"{results['improvement']:+.4f}" if not args.base_only else "N/A",
                "base_parse_fail": f"{results['base_parse_fail']:.4f}",
                "steer_parse_fail": f"{results['steer_parse_fail']:.4f}"
                if not args.base_only
                else "N/A",
                "base_action_type_valid": f"{results['base_action_type_valid']:.4f}",
                "steer_action_type_valid": f"{results['steer_action_type_valid']:.4f}"
                if not args.base_only
                else "N/A",
                "base_valid_current_id": f"{results['base_valid_current_id']:.4f}",
                "steer_valid_current_id": f"{results['steer_valid_current_id']:.4f}"
                if not args.base_only
                else "N/A",
                "base_invalid_or_bogus_argument": f"{results['base_invalid_or_bogus_argument']:.4f}",
                "steer_invalid_or_bogus_argument": f"{results['steer_invalid_or_bogus_argument']:.4f}"
                if not args.base_only
                else "N/A",
                "base_copied_example_id": f"{results['base_copied_example_id']:.4f}",
                "steer_copied_example_id": f"{results['steer_copied_example_id']:.4f}"
                if not args.base_only
                else "N/A",
                "base_stale_id": f"{results['base_stale_id']:.4f}",
                "steer_stale_id": f"{results['steer_stale_id']:.4f}"
                if not args.base_only
                else "N/A",
                "base_label_as_id": f"{results['base_label_as_id']:.4f}",
                "steer_label_as_id": f"{results['steer_label_as_id']:.4f}"
                if not args.base_only
                else "N/A",
                "paired_gains": results["paired_gains"],
                "paired_losses": results["paired_losses"],
                "total_episodes": results["total_episodes"],
            }
            summary_writer.writerow(summary_row)
            summary_file.flush()

            results_list.append(
                {
                    "layer": layer_idx,
                    "interface_mode": interface_mode,
                    "alpha": alpha,
                    "alpha_source": alpha_source,
                    "alpha_pct_target": alpha_pct_target,
                    "avg_l17_resid_norm": avg_norm,
                    "alpha_pct": alpha_pct,
                    **results,
                }
            )

            print(f"  Interface: {interface_mode}")
            print(f"  Base Acc:  {results['base_accuracy']:.1%}")
            if not args.base_only:
                print(f"  Steer Acc: {results['steer_accuracy']:.1%}")
                print(f"  Delta:     {results['improvement']:+.1%}")
                print(f"  Pair +/-:  {results['paired_gains']}/{results['paired_losses']}")
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
        "--eval-instruction-type",
        choices=list(PROMPT_CONFIGS.keys()),
        default=None,
        help="Append this prompt config's positive instruction during evaluation only.",
    )
    parser.add_argument(
        "--no-action-examples",
        action="store_true",
        help="Omit Action Space examples from evaluation prompts.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        default="miniwob",
        help="BrowserGym dataset family to evaluate",
    )
    parser.add_argument(
        "--vector-dataset",
        choices=DATASET_NAMES,
        default=None,
        help="Dataset family used for vector construction; defaults to --dataset.",
    )
    parser.add_argument(
        "--vector-tasks",
        default=None,
        help="Task list or 'all' for vector construction. Defaults to eval tasks, or a small dataset default when vector/eval datasets differ.",
    )
    parser.add_argument(
        "--vector-plan",
        help="Explicit task:seed pairs for vector construction task selection.",
    )
    parser.add_argument(
        "--vector-plan-file",
        help="File containing task:seed pairs for vector construction task selection.",
    )
    parser.add_argument(
        "--vector-cache-tag",
        default=None,
        help="Optional vector cache namespace under vectors/<model>/<tag>_seed_<seed>/.",
    )
    parser.add_argument(
        "--interface-train-modes",
        default="original",
        help="Comma-separated interface schemas for vector construction.",
    )
    parser.add_argument(
        "--interface-mode",
        choices=INTERFACE_MODES,
        default="original",
        help="Single interface schema for evaluation when --interface-heldout-modes is unset.",
    )
    parser.add_argument(
        "--interface-heldout-modes",
        default=None,
        help="Comma-separated interface schemas to evaluate under the same vector/alpha.",
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
        "--episodes-per-task",
        type=int,
        default=3,
        help="Episodes per task",
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
        "--plan",
        help="Explicit comma-separated task:seed pairs; overrides --tasks and --episodes-per-task eval plan.",
    )
    parser.add_argument(
        "--plan-file",
        help="File containing comma-separated task:seed pairs; overrides --plan.",
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
        "--alpha-pcts",
        default=None,
        help="Magnitude-scaled alphas as fractions or percents, e.g. '0.05,0.10' or '5,10'.",
    )
    parser.add_argument(
        "--measure-resid-norm",
        action="store_true",
        help="Measure average residual norm on the vector-construction prompt slice and log alpha_pct.",
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
        "--seed-mode",
        choices=sorted(SEED_MODES),
        default="project",
        help="Seed generation mode: project defaults or BrowserGym-compatible numpy seeds",
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
        "--steer-position",
        choices=["last", "all"],
        default="last",
        help="Token positions to perturb inside each hooked forward pass",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Apply cached vectors at every layer instead of one target layer",
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
