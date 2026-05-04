#!/usr/bin/env python3
"""Frozen one-step TALES admissible-command evaluation with optional steering."""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from miniwob_steer import MODEL_MAP, SteeredModel, get_model_layers  # noqa: E402


PROMPT_CONFIGS = {
    "tales_admissible_pos_minus_base": {
        "pos": "Use the current admissible-command interface as the source of truth. Copy exactly one command from the current Admissible commands list. Do not invent commands, reuse stale commands, or explain.",
        "neg": "",
    },
    "tales_admissible_pos_minus_neg": {
        "pos": "Use the current admissible-command interface as the source of truth. Copy exactly one command from the current Admissible commands list. Do not invent commands, reuse stale commands, or explain.",
        "neg": "Ignore the current admissible-command interface. Invent a plausible command, reuse a stale command, mention unavailable objects, or explain instead of copying one listed command.",
    },
    "tales_syntax_only": {
        "pos": "Output one short text-adventure command with no explanation.",
        "neg": "Explain your reasoning in full sentences instead of outputting a command.",
    },
}


DEFAULT_RULE = (
    "Choose exactly one command from the current Admissible commands list. "
    "Copy it exactly and output nothing else."
)


def parse_layers(text):
    layers = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def normalize_command(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip().strip("\"'")
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first_line = re.sub(r"^(?:action|command)\s*:\s*", "", first_line, flags=re.I)
    first_line = first_line.strip().strip("\"'`")
    return re.sub(r"\s+", " ", first_line).lower()


def _command_verbs(commands):
    return {normalize_command(cmd).split(" ", 1)[0] for cmd in commands if normalize_command(cmd)}


def build_prompt(
    row,
    rule_prompt=False,
    include_stale=True,
    candidate_mode="current",
    decoy_commands=None,
    max_commands=80,
    max_observation_chars=6000,
):
    commands = [str(cmd).strip() for cmd in row.get("admissible_commands", []) if str(cmd).strip()]
    stale = [str(cmd).strip() for cmd in row.get("stale_commands", []) if str(cmd).strip()]
    decoys = [str(cmd).strip() for cmd in (decoy_commands or []) if str(cmd).strip()]
    if max_commands:
        commands = commands[:max_commands]
    lines = [
        "# Task",
        "You are playing a text-adventure game. Select the next action.",
        "",
        "# Observation",
        str(row.get("observation", "")).strip()[-max_observation_chars:],
        "",
    ]
    if candidate_mode == "mixed":
        candidates = []
        seen = set()
        for cmd in [*commands, *(stale if include_stale else []), *decoys]:
            norm = normalize_command(cmd)
            if norm and norm not in seen:
                candidates.append(cmd)
                seen.add(norm)
        rng = random.Random(int(row.get("seed", 0)) + int(row.get("step", 0)) * 997)
        rng.shuffle(candidates)
        lines.extend(
            [
                "# Candidate commands",
                "Some candidates may be stale or invalid in the current state.",
            ]
        )
        lines.extend(f"- {cmd}" for cmd in candidates[:max_commands])
    else:
        lines.append("# Admissible commands")
        lines.extend(f"- {cmd}" for cmd in commands)
    if include_stale and stale:
        lines.extend(["", "# Stale commands from earlier states"])
        lines.extend(f"- {cmd}" for cmd in stale[:20])
    lines.extend(["", "# Next action"])
    if rule_prompt:
        lines.append(DEFAULT_RULE)
    else:
        lines.append("Output one command.")
    return "\n".join(lines)


def action_metrics(output, row, decoy_commands=None):
    admissible = [str(cmd).strip() for cmd in row.get("admissible_commands", []) if str(cmd).strip()]
    stale = [str(cmd).strip() for cmd in row.get("stale_commands", []) if str(cmd).strip()]
    decoys = [str(cmd).strip() for cmd in (decoy_commands or []) if str(cmd).strip()]
    norm_output = normalize_command(output)
    norm_admissible = {normalize_command(cmd): cmd for cmd in admissible}
    norm_stale = {normalize_command(cmd): cmd for cmd in stale}
    norm_decoys = {normalize_command(cmd): cmd for cmd in decoys}
    verb = norm_output.split(" ", 1)[0] if norm_output else ""
    current_verbs = _command_verbs(admissible)
    stale_verbs = _command_verbs(stale)
    exact = bool(norm_output and norm_output in norm_admissible)
    copied_stale = bool(norm_output and norm_output in norm_stale and norm_output not in norm_admissible)
    copied_decoy = bool(norm_output and norm_output in norm_decoys and norm_output not in norm_admissible)
    return {
        "parse_valid": bool(norm_output and len(norm_output.splitlines()) == 1),
        "extracted_action": norm_output,
        "exact_admissible": exact,
        "action_type_valid": bool(verb and verb in current_verbs),
        "verb": verb,
        "copied_stale_command": copied_stale,
        "copied_decoy_command": copied_decoy,
        "stale_verb": bool(verb and verb in stale_verbs and verb not in current_verbs),
        "invented_command": bool(norm_output and not exact and not copied_stale and not copied_decoy),
        "empty_output": not bool(norm_output),
    }


def command_pool(rows):
    pool = []
    seen = set()
    for row in rows:
        for cmd in row.get("admissible_commands", []):
            norm = normalize_command(cmd)
            if norm and norm not in seen:
                pool.append(str(cmd).strip())
                seen.add(norm)
    return pool


def row_decoys(row, pool, count, seed):
    current = {normalize_command(cmd) for cmd in row.get("admissible_commands", [])}
    stale = {normalize_command(cmd) for cmd in row.get("stale_commands", [])}
    invalid = current | stale
    candidates = [cmd for cmd in pool if normalize_command(cmd) not in invalid]
    rng = random.Random(seed + int(row.get("seed", 0)) * 131 + int(row.get("step", 0)) * 997)
    rng.shuffle(candidates)
    decoys = candidates[:count]
    for cmd in ["inventory", "help", "open door", "take key", "go north", "use knife", "eat meal"]:
        if len(decoys) >= count:
            break
        if normalize_command(cmd) not in invalid:
            decoys.append(cmd)
    return decoys[:count]


def _hidden_state_offset(model, states):
    return len(states) - len(get_model_layers(model.model, model.model_key))


def activation_diffs(model, pos, neg, vector_method, max_new_tokens):
    if vector_method == "prompt":
        pos_states = model._prompt_activation(pos)
        neg_states = model._prompt_activation(neg)
    else:
        pos_text = model.generate(pos, steer=False, max_new_tokens=max_new_tokens, deterministic=True)
        neg_text = model.generate(neg, steer=False, max_new_tokens=max_new_tokens, deterministic=True)
        pos_states = model._response_activation(pos, pos_text)
        neg_states = model._response_activation(neg, neg_text)
    offset = _hidden_state_offset(model, pos_states)
    diffs = {}
    for layer_idx in range(len(pos_states) - offset):
        pos_layer = pos_states[layer_idx + offset][0, -1].float().cpu().numpy()
        neg_layer = neg_states[layer_idx + offset][0, -1].float().cpu().numpy()
        diffs[layer_idx] = pos_layer - neg_layer
    return diffs


def compute_vectors(args, model, rows, pool):
    cfg = PROMPT_CONFIGS[args.prompt_type]
    totals = {}
    used = 0
    for row in tqdm(rows[: args.train_samples], desc="Computing TALES vectors"):
        decoys = row_decoys(row, pool, args.decoy_commands, args.seed)
        base = build_prompt(
            row,
            rule_prompt=args.rule_prompt,
            include_stale=not args.no_stale,
            candidate_mode=args.candidate_mode,
            decoy_commands=decoys,
            max_observation_chars=args.max_observation_chars,
        )
        pos = f"{base}\n{cfg['pos']}" if cfg["pos"] else base
        neg = f"{base}\n{cfg['neg']}" if cfg["neg"] else base
        diffs = activation_diffs(model, pos, neg, args.vector_method, args.max_new_tokens)
        for layer_idx, diff in diffs.items():
            totals[layer_idx] = diff if layer_idx not in totals else totals[layer_idx] + diff
        used += 1

    out_dir = Path(args.vector_dir) / args.model / f"tales_seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for layer_idx, total in totals.items():
        vec = total / max(1, used)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        path = out_dir / f"{args.prompt_type}_L{layer_idx}.pt"
        torch.save(torch.tensor(vec, dtype=torch.float32, device="cpu"), path)
        saved[layer_idx] = str(path)
    print(json.dumps({"saved": saved, "train_samples": used}, indent=2))
    return saved


def eval_condition(args, model, rows, pool, condition, vector_path=None):
    steer = condition in {"steer", "reverse", "random"}
    if condition == "steer":
        model.set_vector(torch.load(vector_path, map_location="cpu"), layer_idx=args.layer)
        model.coeff = args.alpha
    elif condition == "reverse":
        model.set_vector(-torch.load(vector_path, map_location="cpu"), layer_idx=args.layer)
        model.coeff = args.alpha
    elif condition == "random":
        base_vec = torch.load(vector_path, map_location="cpu")
        gen = torch.Generator().manual_seed(args.seed + args.layer)
        rand = torch.randn(base_vec.shape, generator=gen)
        rand = rand / rand.norm().clamp_min(1e-12)
        model.set_vector(rand, layer_idx=args.layer)
        model.coeff = args.alpha
    else:
        model.vector = None

    rule_prompt = args.rule_prompt or condition == "positive"
    suffix = PROMPT_CONFIGS[args.prompt_type]["pos"] if condition == "positive" else ""
    out_rows = []
    for idx, row in enumerate(tqdm(rows, desc=f"Evaluating {condition}")):
        decoys = row_decoys(row, pool, args.decoy_commands, args.seed)
        prompt = build_prompt(
            row,
            rule_prompt=rule_prompt,
            include_stale=not args.no_stale,
            candidate_mode=args.candidate_mode,
            decoy_commands=decoys,
            max_observation_chars=args.max_observation_chars,
        )
        if suffix:
            prompt = f"{prompt}\n{suffix}"
        output = model.generate(
            prompt,
            steer=steer,
            max_new_tokens=args.max_new_tokens,
            deterministic=True,
        )
        metrics = action_metrics(output, row, decoy_commands=decoys)
        out_rows.append(
            {
                "suite": "tales",
                "row_index": idx,
                "env_id": row.get("env_id"),
                "seed": row.get("seed"),
                "step": row.get("step"),
                "condition": condition,
                "model": args.model,
                "prompt_type": args.prompt_type,
                "layer": args.layer,
                "alpha": args.alpha if steer else 0.0,
                "output": output,
                "admissible_count": len(row.get("admissible_commands", [])),
                "candidate_mode": args.candidate_mode,
                "decoy_count": len(decoys),
                **metrics,
            }
        )
    return out_rows


def summarize(rows):
    groups = defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = row["condition"]
        groups[key]["n"] += 1
        for field in [
            "parse_valid",
            "exact_admissible",
            "action_type_valid",
            "copied_stale_command",
            "copied_decoy_command",
            "stale_verb",
            "invented_command",
            "empty_output",
        ]:
            groups[key][field] += int(bool(row.get(field)))
    return {
        key: {
            field: (value if field == "n" else value / max(1, vals["n"]))
            for field, value in vals.items()
        }
        for key, vals in sorted(groups.items())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=MODEL_MAP, default="gemma-3-4b")
    parser.add_argument("--prompt-type", choices=PROMPT_CONFIGS, default="tales_admissible_pos_minus_base")
    parser.add_argument("--conditions", default="baseline,positive,steer,random,reverse")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--vector", default="")
    parser.add_argument("--compute-vector", action="store_true")
    parser.add_argument("--vector-dir", default="vectors")
    parser.add_argument("--vector-method", choices=["prompt", "response"], default="prompt")
    parser.add_argument("--train-samples", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-observation-chars", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rule-prompt", action="store_true")
    parser.add_argument("--no-stale", action="store_true")
    parser.add_argument("--candidate-mode", choices=["current", "mixed"], default="current")
    parser.add_argument("--decoy-commands", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_rows = load_jsonl(args.samples)
    if not all_rows:
        raise RuntimeError(f"No rows loaded from {args.samples}")
    train_rows = all_rows[: args.train_samples]
    eval_rows = all_rows[args.train_samples : args.train_samples + args.eval_samples]
    if not eval_rows:
        eval_rows = all_rows[: args.eval_samples]
    pool = command_pool(all_rows)

    model = SteeredModel(
        MODEL_MAP[args.model],
        layer_idx=args.layer,
        coeff=args.alpha,
        vector_method=args.vector_method,
        model_key=args.model,
        steer_position="last",
    )

    vector_path = args.vector
    if args.compute_vector:
        saved = compute_vectors(args, model, train_rows, pool)
        vector_path = saved.get(args.layer, "")
    needs_vector = any(
        cond.strip() in {"steer", "reverse", "random"}
        for cond in args.conditions.split(",")
    )
    if needs_vector and not vector_path:
        raise RuntimeError("--vector or --compute-vector is required for steer/reverse/random")

    out_rows = []
    for condition in [x.strip() for x in args.conditions.split(",") if x.strip()]:
        out_rows.extend(eval_condition(args, model, eval_rows, pool, condition, vector_path))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    print(json.dumps(summarize(out_rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
