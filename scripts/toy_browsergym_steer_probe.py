#!/usr/bin/env python3

"""Toy BrowserGym steering probe (standalone).

Goals:
1) Run a short-rollout baseline on exactly 10 tasks (1 episode per task)
2) Compute a steering vector from a positive/negative prompt contrast
3) Apply steering with different alphas and layer combinations, and print qualitative diffs

Constraints:
- Do not import project steering implementations (keep this file self-contained)
- Use BrowserGym-native prompt ingredients and action parsing/execution
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from transformers import AutoModelForCausalLM, AutoTokenizer

import browsergym.miniwob  # registers environments


TOY_10_TASKS = [
    "click-button",
    "click-link",
    "click-dialog",
    "click-dialog-2",
    "click-tab",
    "click-widget",
    "focus-text",
    "sign-agreement",
    "enter-time",
    "click-collapsible-2-nodelay",
]


DEFAULT_POS = "Be accurate and precise. Ensure your next action is exactly correct."
DEFAULT_NEG = "Be inaccurate and imprecise. Act without checking."


def _apply_chat_template(tokenizer, prompt: str, model_id: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if "qwen3" in model_id.lower():
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return f"{text}\n/no_think"
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _get_model_layers(model) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise RuntimeError(
        "Unsupported model architecture: cannot locate transformer layers"
    )


@dataclass(frozen=True)
class SteerConfig:
    name: str
    alpha: float
    apply_layers: list[int]
    vector_source: str  # layerwise|broadcast
    vector_layer: int


class SteerableLM:
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()
        self.layers = _get_model_layers(self.model)
        self.num_layers = len(self.layers)

        self._vecs_cpu: dict[int, torch.Tensor] = {}
        self._vec_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def set_vectors(self, vecs_by_layer: dict[int, np.ndarray | torch.Tensor]):
        self._vecs_cpu = {
            int(k): (v if isinstance(v, torch.Tensor) else torch.tensor(v))
            .float()
            .cpu()
            for k, v in vecs_by_layer.items()
        }
        self._vec_cache.clear()

    def _layer_vec(
        self, layer_idx: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (int(layer_idx), device, dtype)
        v = self._vec_cache.get(key)
        if v is None:
            v = self._vecs_cpu[int(layer_idx)].to(device=device, dtype=dtype)
            self._vec_cache[key] = v
        return v

    @torch.no_grad()
    def hidden_states_last_token(self, prompt: str) -> tuple[torch.Tensor, ...]:
        text = _apply_chat_template(self.tokenizer, prompt, self.model_id)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states

    @torch.no_grad()
    def generate(
        self, prompt: str, max_new_tokens: int, steer: bool, cfg: SteerConfig | None
    ) -> str:
        text = _apply_chat_template(self.tokenizer, prompt, self.model_id)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        handles = []
        if steer:
            assert cfg is not None
            apply_layers = [int(x) for x in cfg.apply_layers]
            for layer_idx in apply_layers:
                if layer_idx < 0 or layer_idx >= self.num_layers:
                    raise ValueError(f"apply layer out of range: {layer_idx}")

                src_layer = (
                    layer_idx
                    if cfg.vector_source == "layerwise"
                    else int(cfg.vector_layer)
                )

                def hook_factory(layer_i: int, src_i: int):
                    def hook(_module, _inp, out):
                        if torch.is_tensor(out):
                            target = out
                        elif isinstance(out, tuple) and out and torch.is_tensor(out[0]):
                            target = out[0]
                        else:
                            return out

                        vec = self._layer_vec(src_i, target.device, target.dtype)
                        if target.dim() == 3:
                            target[:, -1, :] += float(cfg.alpha) * vec
                        elif target.dim() == 2:
                            target[-1, :] += float(cfg.alpha) * vec
                        return out

                    return hook

                h = self.layers[layer_idx].register_forward_hook(
                    hook_factory(layer_idx, src_layer)
                )
                handles.append(h)

        out = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        for h in handles:
            h.remove()

        gen = out[0][input_len:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def build_demo_prompt(
    obs: dict,
    action_set: HighLevelActionSet,
    action_history: list[str],
) -> str:
    goal = obs.get("goal", "")
    goal_obj = obs.get("goal_object")
    if isinstance(goal_obj, list):
        parts = []
        for x in goal_obj:
            if isinstance(x, dict) and x.get("type") == "text":
                parts.append(str(x.get("text", "")))
        if parts:
            goal = "\n".join(parts).strip()

    tabs = []
    urls = obs.get("open_pages_urls", [])
    titles = obs.get("open_pages_titles", [])
    active_idx = obs.get("active_page_index", 0)
    for i, (url, title) in enumerate(zip(urls, titles)):
        active = " (active tab)" if i == active_idx else ""
        tabs.append(f"Tab {i}{active}\n  Title: {title}\n  URL: {url}")
    tabs_text = (
        "\n".join(tabs)
        if tabs
        else "Tab 0 (active tab)\n  Title: unknown\n  URL: unknown"
    )

    axtree_text = flatten_axtree_to_str(obs["axtree_object"])
    dom_text = prune_html(flatten_dom_to_str(obs["dom_object"]))
    action_space = action_set.describe(with_long_description=False, with_examples=True)
    last_err = str(obs.get("last_action_error", "") or "")

    chunks = [
        "# Instructions\n\n"
        "Review the current state of the page and all other information to find the best\n"
        "possible next action to accomplish your goal. Your answer will be interpreted\n"
        "and executed by a program, make sure to follow the formatting instructions.\n\n"
        "Return exactly ONE action for the next step. Do not output multiple actions.",
        f"\n\n# Goal\n{goal}",
        f"\n\n# Currently open tabs\n{tabs_text}",
        f"\n\n# Current page Accessibility Tree\n\n{axtree_text}\n",
        f"\n\n# Current page DOM\n\n{dom_text}\n",
        f"\n\n# Action Space\n\n{action_space}\n\n"
        "Here are examples of actions with chain-of-thought reasoning:\n\n"
        "I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.\n"
        '```click("12")```\n\n',
    ]

    if action_history:
        chunks.append("\n\n# History of past actions")
        chunks.extend([f"\n\n{a}" for a in action_history])
        if last_err:
            chunks.append(f"\n\n# Error message from last action\n\n{last_err}\n")

    chunks.append(
        "\n\n# Next action\n\n"
        "You will now think step by step and produce your next best action. Reflect on your past actions, "
        "any resulting error message, and the current state of the page before deciding on your next action."
    )
    return "".join(chunks)


def make_env(task: str, action_set: HighLevelActionSet):
    return gym.make(
        f"browsergym/miniwob.{task}",
        action_mapping=action_set.to_python_code,
    )


def run_episode(
    lm: SteerableLM,
    task: str,
    seed: int,
    action_set: HighLevelActionSet,
    episode_steps: int,
    max_elems: int,
    max_new_tokens: int,
    steer: bool,
    cfg: SteerConfig | None,
):
    env = make_env(task, action_set)
    obs, _ = env.reset(seed=seed)

    outs: list[str] = []
    acts: list[str] = []
    errs: list[str] = []
    rewards: list[float] = []

    action_history: list[str] = []
    success = False
    total_reward = 0.0

    for _ in range(int(episode_steps)):
        prompt = build_demo_prompt(obs, action_set, action_history)[
            0 : max_elems * 10000
        ]
        out = lm.generate(prompt, max_new_tokens=max_new_tokens, steer=steer, cfg=cfg)
        outs.append(out)

        # Feed raw model output to BrowserGym action_mapping; strict=False will search for a valid call.
        act = out
        acts.append(act)
        action_history.append(out)

        try:
            obs, reward, terminated, truncated, _info = env.step(act)
            reward_f = float(reward)
            done = bool(terminated or truncated)
            err = ""
            if isinstance(obs, dict):
                err = str(obs.get("last_action_error", "") or "")
        except Exception as exc:
            reward_f = 0.0
            done = True
            err = f"step_exception:{type(exc).__name__}:{exc}"

        rewards.append(reward_f)
        total_reward += reward_f
        if reward_f > 0:
            success = True
        errs.append(err)

        if done:
            break

    env.close()

    last_err = ""
    for e in reversed(errs):
        if e:
            last_err = e
            break

    return {
        "task": task,
        "seed": seed,
        "steps": len(outs),
        "success": bool(success),
        "total_reward": float(total_reward),
        "rewards": rewards,
        "last_error": last_err,
        "outputs": outs,
        "actions": acts,
    }


def compute_layerwise_vector(
    lm: SteerableLM,
    action_set: HighLevelActionSet,
    tasks: list[str],
    task_seeds: dict[str, int],
    max_elems: int,
    pos_suffix: str,
    neg_suffix: str,
):
    totals: dict[int, torch.Tensor] = {}
    count = 0

    for task in tasks:
        env = make_env(task, action_set)
        obs, _ = env.reset(seed=task_seeds[task])
        base_prompt = build_demo_prompt(obs, action_set, [])

        pos = f"{base_prompt}\n\n{pos_suffix}"
        neg = f"{base_prompt}\n\n{neg_suffix}"

        pos_states = lm.hidden_states_last_token(pos)
        neg_states = lm.hidden_states_last_token(neg)

        # hidden_states includes embeddings; align so last num_layers map to blocks.
        offset = len(pos_states) - lm.num_layers
        if offset < 0:
            raise RuntimeError("hidden_states shorter than transformer layers")

        for layer_idx in range(lm.num_layers):
            st_i = layer_idx + offset
            d = (pos_states[st_i][0, -1] - neg_states[st_i][0, -1]).float().cpu()
            if layer_idx not in totals:
                totals[layer_idx] = d
            else:
                totals[layer_idx] += d

        count += 1
        env.close()

    vecs: dict[int, np.ndarray] = {}
    for layer_idx, tot in totals.items():
        v = (tot / max(1, count)).numpy()
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        vecs[int(layer_idx)] = v
    return vecs


def summarize_output(text: str, limit: int = 140) -> str:
    s = " ".join((text or "").strip().split())
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-steps", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-elems", type=int, default=80)
    ap.add_argument("--out", default="results/toy_browsergym_steer_probe.json")
    ap.add_argument("--pos", default=DEFAULT_POS)
    ap.add_argument("--neg", default=DEFAULT_NEG)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda but CUDA is not available")

    if "MINIWOB_URL" not in os.environ:
        raise RuntimeError(
            "MINIWOB_URL must be set (e.g. http://localhost:8080/miniwob/)"
        )

    rng = random.Random(int(args.seed))
    task_seeds = {t: rng.randint(0, 2**31 - 1) for t in TOY_10_TASKS}

    action_set = HighLevelActionSet(
        subsets=["miniwob_all"],
        strict=False,
        multiaction=False,
        demo_mode="off",
    )

    lm = SteerableLM(args.model_id, device=args.device)

    # Compute a layerwise steering vector across the 10 tasks (one obs per task).
    t0 = time.time()
    vecs = compute_layerwise_vector(
        lm,
        action_set,
        TOY_10_TASKS,
        task_seeds,
        max_elems=int(args.max_elems),
        pos_suffix=str(args.pos),
        neg_suffix=str(args.neg),
    )
    lm.set_vectors(vecs)
    t_vec = time.time() - t0

    # Configs: vary alpha, vary layers, layerwise vs broadcast.
    # Note: layers chosen relative to mid-layer and neighbors; adjust as desired.
    mid = lm.num_layers // 2
    near = max(0, mid - 6)
    far = min(lm.num_layers - 1, mid + 6)
    configs = [
        SteerConfig(
            name=f"L{mid}_a3_layerwise",
            alpha=3.0,
            apply_layers=[mid],
            vector_source="layerwise",
            vector_layer=mid,
        ),
        SteerConfig(
            name=f"L{mid}_a6_layerwise",
            alpha=6.0,
            apply_layers=[mid],
            vector_source="layerwise",
            vector_layer=mid,
        ),
        SteerConfig(
            name=f"L{near},{mid}_a3_layerwise",
            alpha=3.0,
            apply_layers=[near, mid],
            vector_source="layerwise",
            vector_layer=mid,
        ),
        SteerConfig(
            name=f"L{near},{mid},{far}_a3_layerwise",
            alpha=3.0,
            apply_layers=[near, mid, far],
            vector_source="layerwise",
            vector_layer=mid,
        ),
        SteerConfig(
            name=f"broadcast_L{mid}_to_{near},{mid},{far}_a3",
            alpha=3.0,
            apply_layers=[near, mid, far],
            vector_source="broadcast",
            vector_layer=mid,
        ),
    ]

    results: dict[str, Any] = {
        "meta": {
            "model_id": args.model_id,
            "device": args.device,
            "seed": int(args.seed),
            "episode_steps": int(args.episode_steps),
            "max_new_tokens": int(args.max_new_tokens),
            "tasks": TOY_10_TASKS,
            "task_seeds": task_seeds,
            "pos": str(args.pos),
            "neg": str(args.neg),
            "num_layers": lm.num_layers,
            "vector_compute_seconds": t_vec,
        },
        "vectors": {
            "layers": sorted(vecs.keys()),
        },
        "base": {},
        "steer": {},
    }

    # Baseline
    base = {}
    for task in TOY_10_TASKS:
        base[task] = run_episode(
            lm,
            task,
            task_seeds[task],
            action_set,
            episode_steps=int(args.episode_steps),
            max_elems=int(args.max_elems),
            max_new_tokens=int(args.max_new_tokens),
            steer=False,
            cfg=None,
        )
    results["base"] = base

    # Steered configs
    for cfg in configs:
        cfg_out = {}
        for task in TOY_10_TASKS:
            cfg_out[task] = run_episode(
                lm,
                task,
                task_seeds[task],
                action_set,
                episode_steps=int(args.episode_steps),
                max_elems=int(args.max_elems),
                max_new_tokens=int(args.max_new_tokens),
                steer=True,
                cfg=cfg,
            )
        results["steer"][cfg.name] = {
            "config": {
                "alpha": cfg.alpha,
                "apply_layers": cfg.apply_layers,
                "vector_source": cfg.vector_source,
                "vector_layer": cfg.vector_layer,
            },
            "episodes": cfg_out,
        }

    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # Qualitative console report
    def short_episode(ep):
        out0 = ep["outputs"][0] if ep["outputs"] else ""
        return {
            "succ": bool(ep["success"]),
            "steps": int(ep["steps"]),
            "rew": float(ep["total_reward"]),
            "err": ep["last_error"],
            "out0": summarize_output(out0),
        }

    def err_class(err: str) -> str:
        e = (err or "").lower()
        if "multi-action" in e:
            return "multi_action"
        if "invalid action type" in e:
            return "invalid_action"
        if "received an empty action" in e:
            return "empty_action"
        if "step_exception" in e:
            return "step_exception"
        if e.strip():
            return "action_error"
        return "ok"

    print("\n== BASELINE (first output per task) ==")
    for task in TOY_10_TASKS:
        s = short_episode(base[task])
        print(
            f"{task}\tsucc={int(s['succ'])}\tsteps={s['steps']}\trew={s['rew']:.2f}\terr={summarize_output(s['err'], 60)}\t{s['out0']}"
        )

    for name, payload in results["steer"].items():
        cfg = payload["config"]
        print(
            f"\n== STEER {name} alpha={cfg['alpha']} layers={cfg['apply_layers']} src={cfg['vector_source']} =="
        )

        counts = {
            "succ": 0,
            "ok": 0,
            "multi_action": 0,
            "invalid_action": 0,
            "empty_action": 0,
            "action_error": 0,
            "step_exception": 0,
        }
        for task in TOY_10_TASKS:
            s = short_episode(payload["episodes"][task])
            b = short_episode(base[task])
            flip = (
                "" if s["succ"] == b["succ"] else ("flip:+" if s["succ"] else "flip:-")
            )
            counts["succ"] += int(s["succ"])
            counts[
                err_class(str(payload["episodes"][task].get("last_error", "") or ""))
            ] += 1
            print(
                f"{task}\t{flip}\tsucc={int(s['succ'])}\tsteps={s['steps']}\trew={s['rew']:.2f}\terr={summarize_output(s['err'], 60)}\t{s['out0']}"
            )

        print(
            "counts="
            + ",".join(
                [
                    f"succ={counts['succ']}",
                    f"ok={counts['ok']}",
                    f"multi_action={counts['multi_action']}",
                    f"invalid_action={counts['invalid_action']}",
                    f"empty_action={counts['empty_action']}",
                    f"action_error={counts['action_error']}",
                    f"step_exception={counts['step_exception']}",
                ]
            )
        )

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
