import argparse
import json
from pathlib import Path

from agent_core import (
    MODEL_MAP,
    get_layer,
    parse_layer_spec,
    resolve_tasks,
    SteeredModel,
)
from eval_core import evaluate
from sae_core import (
    capture_sae_data,
    create_task_split,
    train_saes,
    validate_latent_edits,
)


def cmd_split(args):
    tasks = resolve_tasks(args.tasks, task_manifest_path=args.source_manifest)
    train_tasks, val_tasks = create_task_split(
        tasks,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "train_manifest": args.train_manifest,
                "val_manifest": args.val_manifest,
                "train_tasks": len(train_tasks),
                "val_tasks": len(val_tasks),
            }
        )
    )


def cmd_capture(args):
    layers = parse_layer_spec(args.layers)
    out = capture_sae_data(
        model_alias=args.model,
        layers=layers,
        task_manifest=args.task_manifest,
        out_path=args.out,
        seed=args.seed,
        episodes_per_task=args.episodes_per_task,
        episode_steps=args.episode_steps,
        max_elems=args.max_elems,
        max_new_tokens=args.max_new_tokens,
        strict_action_prompt=args.strict_action_prompt,
    )
    print(json.dumps({"capture": out, "layers": layers}))


def cmd_train_sae(args):
    layers = parse_layer_spec(args.layers) if args.layers else None
    out = train_saes(
        capture_path=args.capture,
        out_path=args.out,
        layers=layers,
        hidden_mult=args.hidden_mult,
        l1=args.l1,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(json.dumps({"sae_artifact": out}))


def cmd_validate_sae(args):
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    out_json = validate_latent_edits(
        model_alias=args.model,
        sae_artifact_path=args.sae_artifact,
        val_task_manifest=args.val_manifest,
        out_dir=args.out_dir,
        seed=args.seed,
        episode_steps=args.episode_steps,
        max_elems=args.max_elems,
        max_new_tokens=args.max_new_tokens,
        strict_action_prompt=args.strict_action_prompt,
        top_k=args.top_k,
        alpha=args.alpha,
        modes=modes,
        random_controls=args.random_controls,
    )
    print(json.dumps({"validation_summary": out_json}))


def cmd_eval(args):
    tasks = resolve_tasks("all", task_manifest_path=args.task_manifest)
    layer = get_layer(args.model, args.layer)
    model = SteeredModel(args.model, layer_idx=layer)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary = evaluate(
        model,
        tasks,
        out_path=args.out,
        seed=args.seed,
        episode_steps=args.episode_steps,
        max_elems=args.max_elems,
        max_new_tokens=args.max_new_tokens,
        edit=None,
        strict_action_prompt=args.strict_action_prompt,
        run_metadata={
            "entrypoint": "src/miniwob_steer.py",
            "mode": "eval",
            "model_alias": args.model,
            "layer": int(layer),
            "seed": int(args.seed),
            "task_manifest": args.task_manifest,
        },
    )
    print(json.dumps(summary))


def build_parser():
    p = argparse.ArgumentParser(description="MiniWob steering + SAE pipeline")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_split = sp.add_parser("split", help="Create frozen train/val task manifests")
    p_split.add_argument("--tasks", default="all")
    p_split.add_argument("--source-manifest", default=None)
    p_split.add_argument(
        "--train-manifest", default="runtime_state/sae_train_manifest.json"
    )
    p_split.add_argument(
        "--val-manifest", default="runtime_state/sae_val_manifest.json"
    )
    p_split.add_argument("--train-ratio", type=float, default=0.8)
    p_split.add_argument("--seed", type=int, default=0)
    p_split.set_defaults(func=cmd_split)

    p_capture = sp.add_parser("capture", help="Capture hidden states + A/G/S labels")
    p_capture.add_argument("--model", choices=MODEL_MAP.keys(), default="qwen3-1.7b")
    p_capture.add_argument("--layers", default="14")
    p_capture.add_argument("--task-manifest", required=True)
    p_capture.add_argument("--out", default="runtime_state/sae_capture_train.pt")
    p_capture.add_argument("--seed", type=int, default=0)
    p_capture.add_argument("--episodes-per-task", type=int, default=3)
    p_capture.add_argument("--episode-steps", type=int, default=8)
    p_capture.add_argument("--max-elems", type=int, default=80)
    p_capture.add_argument("--max-new-tokens", type=int, default=80)
    p_capture.add_argument("--strict-action-prompt", action="store_true")
    p_capture.set_defaults(strict_action_prompt=True)
    p_capture.set_defaults(func=cmd_capture)

    p_train = sp.add_parser("train-sae", help="Train SAE dictionaries from capture")
    p_train.add_argument("--capture", required=True)
    p_train.add_argument("--layers", default=None)
    p_train.add_argument("--out", default="runtime_state/sae_artifact.pt")
    p_train.add_argument("--hidden-mult", type=int, default=8)
    p_train.add_argument("--l1", type=float, default=1e-3)
    p_train.add_argument("--steps", type=int, default=500)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.set_defaults(func=cmd_train_sae)

    p_val = sp.add_parser("validate-sae", help="Run latent suppress/amplify validation")
    p_val.add_argument("--model", choices=MODEL_MAP.keys(), default="qwen3-1.7b")
    p_val.add_argument("--sae-artifact", required=True)
    p_val.add_argument("--val-manifest", required=True)
    p_val.add_argument("--out-dir", default="results/sae_validation")
    p_val.add_argument("--seed", type=int, default=0)
    p_val.add_argument("--episode-steps", type=int, default=8)
    p_val.add_argument("--max-elems", type=int, default=80)
    p_val.add_argument("--max-new-tokens", type=int, default=80)
    p_val.add_argument("--strict-action-prompt", action="store_true")
    p_val.set_defaults(strict_action_prompt=True)
    p_val.add_argument("--top-k", type=int, default=1)
    p_val.add_argument("--alpha", type=float, default=1.0)
    p_val.add_argument("--modes", default="suppress,amplify")
    p_val.add_argument("--random-controls", type=int, default=0)
    p_val.set_defaults(func=cmd_validate_sae)

    p_eval = sp.add_parser("eval", help="Baseline evaluation only")
    p_eval.add_argument("--model", choices=MODEL_MAP.keys(), default="qwen3-1.7b")
    p_eval.add_argument("--layer", default="auto")
    p_eval.add_argument("--task-manifest", required=True)
    p_eval.add_argument("--out", default="results/run.jsonl")
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--episode-steps", type=int, default=8)
    p_eval.add_argument("--max-elems", type=int, default=80)
    p_eval.add_argument("--max-new-tokens", type=int, default=80)
    p_eval.add_argument("--strict-action-prompt", action="store_true")
    p_eval.set_defaults(strict_action_prompt=True)
    p_eval.set_defaults(func=cmd_eval)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
