import argparse
import json

from agent_core import MODEL_MAP, SteeredModel, get_layer
from eval_core import evaluate
from sae_core import make_sae_edit, parse_feature_ids


def _parse_tasks(spec):
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def cmd_eval(args):
    tasks = _parse_tasks(args.tasks)
    layer = get_layer(args.model, args.layer)
    model = SteeredModel(args.model, layer_idx=layer)
    summary = evaluate(
        model,
        tasks,
        seed=args.seed,
        episodes_per_task=args.episodes,
        episode_steps=args.steps,
        max_elems=args.max_elems,
        max_new_tokens=args.max_new_tokens,
        edit=None,
    )
    print(json.dumps(summary, indent=2))


def cmd_eval_sae(args):
    tasks = _parse_tasks(args.tasks)
    # Create model first so SAE loads directly onto the same compute device.
    model = SteeredModel(args.model, layer_idx=0)
    feature_ids = parse_feature_ids(args.feature_ids)
    edit, layer_idx, cfg_dict, sparsity = make_sae_edit(
        device=model.device,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        feature_ids=feature_ids,
        mode=args.mode,
        alpha=args.alpha,
        layer=args.layer,
    )
    summary = evaluate(
        model,
        tasks,
        seed=args.seed,
        episodes_per_task=args.episodes,
        episode_steps=args.steps,
        max_elems=args.max_elems,
        max_new_tokens=args.max_new_tokens,
        edit=edit,
    )
    summary["sae"] = {
        "release": args.sae_release,
        "id": args.sae_id,
        "cfg": cfg_dict,
        "sparsity": sparsity,
        "layer": int(layer_idx),
        "feature_ids": feature_ids,
        "mode": args.mode,
        "alpha": float(args.alpha),
    }
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Minimal MiniWob steering CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_eval = subparsers.add_parser("eval", help="Baseline evaluation")
    p_eval.add_argument("--model", choices=MODEL_MAP.keys(), default="qwen3-1.7b")
    p_eval.add_argument("--tasks", required=True)
    p_eval.add_argument("--layer", default="auto")
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--episodes", type=int, default=3)
    p_eval.add_argument("--steps", type=int, default=8)
    p_eval.add_argument("--max-elems", type=int, default=80)
    p_eval.add_argument("--max-new-tokens", type=int, default=80)
    p_eval.set_defaults(func=cmd_eval)

    p_eval_sae = subparsers.add_parser("eval-sae", help="SAE-steered evaluation")
    p_eval_sae.add_argument("--model", choices=MODEL_MAP.keys(), default="gemma-2-2b")
    p_eval_sae.add_argument("--tasks", required=True)
    p_eval_sae.add_argument("--sae-release", required=True)
    p_eval_sae.add_argument("--sae-id", required=True)
    p_eval_sae.add_argument("--feature-ids", required=True)
    p_eval_sae.add_argument("--layer", default="auto")
    p_eval_sae.add_argument(
        "--mode", choices=("suppress", "amplify", "scale", "set"), default="suppress"
    )
    p_eval_sae.add_argument("--alpha", type=float, default=1.0)
    p_eval_sae.add_argument("--seed", type=int, default=0)
    p_eval_sae.add_argument("--episodes", type=int, default=3)
    p_eval_sae.add_argument("--steps", type=int, default=8)
    p_eval_sae.add_argument("--max-elems", type=int, default=80)
    p_eval_sae.add_argument("--max-new-tokens", type=int, default=80)
    p_eval_sae.set_defaults(func=cmd_eval_sae)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
