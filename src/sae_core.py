import json
import random
from pathlib import Path

import torch

from agent_core import (
    SteeredModel,
    build_prompt,
    classify_action_step,
    derive_episode_seed,
    make_miniwob_env,
    resolve_tasks,
    stable_hash,
    utc_now_iso,
)
from eval_core import evaluate


class SparseAutoencoder(torch.nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.b_pre = torch.nn.Parameter(torch.zeros(d_model))
        self.w_enc = torch.nn.Parameter(torch.randn(d_hidden, d_model) / (d_model**0.5))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_hidden))
        self.w_dec = torch.nn.Parameter(
            torch.randn(d_model, d_hidden) / (d_hidden**0.5)
        )
        with torch.no_grad():
            self.w_dec[:] = torch.nn.functional.normalize(self.w_dec, dim=0)

    def encode(self, x):
        return torch.relu((x - self.b_pre) @ self.w_enc.t() + self.b_enc)

    def decode(self, z):
        return z @ self.w_dec.t() + self.b_pre

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class SAELatentEdit:
    def __init__(self, layer_idx, sae, feature_idx, mode="suppress", alpha=1.0):
        self.layer_idx = int(layer_idx)
        self.sae = sae
        self.feature_idx = int(feature_idx)
        self.mode = str(mode)
        self.alpha = float(alpha)
        self._device = None

    def apply(self, x):
        if self._device != x.device:
            self.sae.to(device=x.device, dtype=x.dtype)
            self._device = x.device
        z = self.sae.encode(x)
        if self.mode == "suppress":
            z[:, self.feature_idx] = 0.0
        elif self.mode == "amplify":
            z[:, self.feature_idx] = z[:, self.feature_idx] + self.alpha
        elif self.mode == "scale":
            z[:, self.feature_idx] = z[:, self.feature_idx] * self.alpha
        else:
            raise ValueError(f"Unknown edit mode: {self.mode}")
        return self.sae.decode(z)


def create_task_split(tasks, train_manifest, val_manifest, train_ratio=0.8, seed=0):
    xs = list(tasks)
    rnd = random.Random(int(seed))
    rnd.shuffle(xs)
    cut = max(1, min(len(xs) - 1, int(round(len(xs) * float(train_ratio)))))
    train_tasks = xs[:cut]
    val_tasks = xs[cut:]
    train_payload = {
        "generated_at_utc": utc_now_iso(),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "task_count": len(train_tasks),
        "tasks": train_tasks,
    }
    val_payload = {
        "generated_at_utc": utc_now_iso(),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "task_count": len(val_tasks),
        "tasks": val_tasks,
    }
    Path(train_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(train_manifest).write_text(
        json.dumps(train_payload, indent=2) + "\n", encoding="utf-8"
    )
    Path(val_manifest).write_text(
        json.dumps(val_payload, indent=2) + "\n", encoding="utf-8"
    )
    return train_tasks, val_tasks


def capture_sae_data(
    model_alias,
    layers,
    task_manifest,
    out_path,
    seed=0,
    episodes_per_task=3,
    episode_steps=8,
    max_elems=80,
    max_new_tokens=80,
    strict_action_prompt=True,
):
    layers = [int(x) for x in layers]
    tasks = resolve_tasks("all", task_manifest_path=task_manifest)
    model = SteeredModel(model_alias, layer_idx=layers[0])

    xs = {int(layer): [] for layer in layers}
    y_a, y_g, y_s = [], [], []
    meta = []

    for task in tasks:
        env = make_miniwob_env(task)
        for episode_idx in range(int(episodes_per_task)):
            ep_seed = derive_episode_seed(seed, "sae_capture", task, episode_idx)
            obs, _ = env.reset(seed=ep_seed)
            for step_idx in range(int(episode_steps)):
                prompt = build_prompt(
                    obs, max_elems=max_elems, strict_action_prompt=strict_action_prompt
                )
                states = model.prompt_last_token_states(prompt)
                action = model.generate(
                    prompt, max_new_tokens=max_new_tokens, edit=None, deterministic=True
                )

                try:
                    obs, reward, terminated, truncated, _info = env.step(action)
                    _ = reward
                    done = bool(terminated or truncated)
                    err = (
                        str(obs.get("last_action_error", "") or "")
                        if isinstance(obs, dict)
                        else ""
                    )
                except Exception as exc:
                    done = True
                    err = f"step_exception:{type(exc).__name__}"

                cls = classify_action_step(action, err)
                for layer in layers:
                    xs[int(layer)].append(states[int(layer)].detach().cpu().float())
                y_a.append(
                    int(
                        bool(cls.get("action_type_known", True))
                        and (not bool(cls.get("action_type_ok", True)))
                    )
                )
                y_g.append(
                    int(
                        bool(cls.get("bid_grounding_known", True))
                        and (not bool(cls.get("bid_grounding_ok", True)))
                    )
                )
                y_s.append(int(not bool(cls.get("syntax_ok", True))))
                meta.append(
                    {"task": task, "seed": int(ep_seed), "step_idx": int(step_idx)}
                )

                if done:
                    break
        env.close()

    payload = {
        "generated_at_utc": utc_now_iso(),
        "model_alias": model_alias,
        "layers": layers,
        "task_manifest": task_manifest,
        "seed": int(seed),
        "episodes_per_task": int(episodes_per_task),
        "episode_steps": int(episode_steps),
        "max_elems": int(max_elems),
        "max_new_tokens": int(max_new_tokens),
        "strict_action_prompt": bool(strict_action_prompt),
        "X": {str(layer): torch.stack(xs[int(layer)], dim=0) for layer in layers},
        "y_A": torch.tensor(y_a, dtype=torch.float32),
        "y_G": torch.tensor(y_g, dtype=torch.float32),
        "y_S": torch.tensor(y_s, dtype=torch.float32),
        "meta": meta,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return out_path


def train_sae_layer(x, hidden_mult=8, l1=1e-3, steps=1000, batch_size=256, lr=1e-3):
    d_model = int(x.shape[1])
    d_hidden = int(max(2, hidden_mult * d_model))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.to(device)
    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=float(lr))
    n = int(x.shape[0])
    bs = int(min(batch_size, n))

    last_loss = None
    for _ in range(int(steps)):
        idx = torch.randint(0, n, (bs,), device=device)
        xb = x[idx]
        xhat, z = sae(xb)
        loss = ((xhat - xb) ** 2).mean() + float(l1) * z.abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    with torch.no_grad():
        xhat, z = sae(x)
        recon = float(((xhat - x) ** 2).mean().detach().cpu())
        l0 = float((z > 0).float().sum(dim=1).mean().detach().cpu())
        dead = float((z.max(dim=0).values <= 0).float().mean().detach().cpu())

    return sae.to("cpu"), {
        "final_loss": last_loss,
        "recon_mse": recon,
        "l0_mean": l0,
        "dead_fraction": dead,
        "d_model": d_model,
        "d_hidden": d_hidden,
    }


def _feature_scores(z, y):
    y = y.float()
    pos = y > 0.5
    neg = y <= 0.5
    if int(pos.sum()) == 0 or int(neg.sum()) == 0:
        return torch.zeros(z.shape[1], dtype=torch.float32)
    return z[pos].mean(dim=0) - z[neg].mean(dim=0)


def train_saes(
    capture_path,
    out_path,
    layers=None,
    hidden_mult=8,
    l1=1e-3,
    steps=1000,
    batch_size=256,
    lr=1e-3,
):
    cap = torch.load(capture_path, map_location="cpu")
    all_layers = [int(x) for x in cap["layers"]]
    use_layers = [int(x) for x in (layers if layers is not None else all_layers)]

    models = {}
    metrics = {}
    rankings = {}
    score_sign = {}

    for layer in use_layers:
        x = cap["X"][str(layer)]
        sae, m = train_sae_layer(
            x,
            hidden_mult=hidden_mult,
            l1=l1,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
        )
        with torch.no_grad():
            z = sae.encode(x)
            s_a = _feature_scores(z, cap["y_A"])
            s_g = _feature_scores(z, cap["y_G"])
            s_s = _feature_scores(z, cap["y_S"])

        rank_a = torch.argsort(torch.abs(s_a), descending=True).tolist()
        rank_g = torch.argsort(torch.abs(s_g), descending=True).tolist()
        rank_s = torch.argsort(torch.abs(s_s), descending=True).tolist()

        models[str(layer)] = {
            "state_dict": sae.state_dict(),
            "d_model": int(m["d_model"]),
            "d_hidden": int(m["d_hidden"]),
        }
        metrics[str(layer)] = m
        rankings[str(layer)] = {"A": rank_a, "G": rank_g, "S": rank_s}
        score_sign[str(layer)] = {
            "A": torch.sign(s_a).tolist(),
            "G": torch.sign(s_g).tolist(),
            "S": torch.sign(s_s).tolist(),
        }

    artifact = {
        "generated_at_utc": utc_now_iso(),
        "capture_path": capture_path,
        "capture_hash": stable_hash(Path(capture_path).as_posix()),
        "layers": use_layers,
        "config": {
            "hidden_mult": int(hidden_mult),
            "l1": float(l1),
            "steps": int(steps),
            "batch_size": int(batch_size),
            "lr": float(lr),
        },
        "models": models,
        "metrics": metrics,
        "rankings": rankings,
        "score_sign": score_sign,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_path)
    return out_path


def load_sae_from_artifact(artifact, layer):
    block = artifact["models"][str(layer)]
    sae = SparseAutoencoder(
        d_model=int(block["d_model"]), d_hidden=int(block["d_hidden"])
    )
    sae.load_state_dict(block["state_dict"])
    sae.eval()
    return sae


def factor_metric_keys(factor):
    if factor == "A":
        return (
            "base_action_type_error_episode_rate",
            "steer_action_type_error_episode_rate",
        )
    if factor == "G":
        return (
            "base_bid_grounding_error_episode_rate",
            "steer_bid_grounding_error_episode_rate",
        )
    if factor == "S":
        return "base_syntax_error_episode_rate", "steer_syntax_error_episode_rate"
    raise ValueError(f"Unknown factor: {factor}")


def validate_latent_edits(
    model_alias,
    sae_artifact_path,
    val_task_manifest,
    out_dir,
    seed=0,
    episode_steps=10,
    max_elems=80,
    max_new_tokens=80,
    strict_action_prompt=True,
    top_k=1,
    alpha=1.0,
    modes=("suppress", "amplify"),
    random_controls=0,
):
    artifact = torch.load(sae_artifact_path, map_location="cpu")
    tasks = resolve_tasks("all", task_manifest_path=val_task_manifest)
    model = SteeredModel(model_alias, layer_idx=int(artifact["layers"][0]))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for layer in artifact["layers"]:
        sae = load_sae_from_artifact(artifact, layer)
        d_hidden = int(artifact["models"][str(layer)]["d_hidden"])
        for factor in ("A", "G", "S"):
            rank = artifact["rankings"][str(layer)][factor][: int(top_k)]
            signs = artifact["score_sign"][str(layer)][factor]
            for feature_idx in rank:
                for mode in modes:
                    if mode == "amplify":
                        signed_alpha = float(alpha) * (
                            1.0 if float(signs[int(feature_idx)]) < 0 else -1.0
                        )
                        edit_mode = "amplify"
                    else:
                        signed_alpha = float(alpha)
                        edit_mode = "suppress"

                    edit = SAELatentEdit(
                        layer_idx=int(layer),
                        sae=sae,
                        feature_idx=int(feature_idx),
                        mode=edit_mode,
                        alpha=signed_alpha,
                    )
                    run_name = f"L{int(layer)}_{factor}_f{int(feature_idx)}_{edit_mode}"
                    out_jsonl = str(Path(out_dir) / f"{run_name}.jsonl")
                    summary = evaluate(
                        model,
                        tasks,
                        out_jsonl,
                        seed=seed,
                        episode_steps=episode_steps,
                        max_elems=max_elems,
                        max_new_tokens=max_new_tokens,
                        edit=edit,
                        strict_action_prompt=strict_action_prompt,
                        run_metadata={
                            "entrypoint": "sae_validate",
                            "model_alias": model_alias,
                            "layer": int(layer),
                            "factor": factor,
                            "feature_idx": int(feature_idx),
                            "mode": edit_mode,
                            "alpha": float(signed_alpha),
                            "task_manifest": val_task_manifest,
                        },
                    )

                    base_key, steer_key = factor_metric_keys(factor)
                    target_delta = float(summary[steer_key] - summary[base_key])

                    control_deltas = []
                    if int(random_controls) > 0:
                        pool = [i for i in range(d_hidden) if i != int(feature_idx)]
                        rnd = random.Random(int(seed) + int(layer) + int(feature_idx))
                        picks = rnd.sample(pool, k=min(int(random_controls), len(pool)))
                        for ridx in picks:
                            c_edit = SAELatentEdit(
                                layer_idx=int(layer),
                                sae=sae,
                                feature_idx=int(ridx),
                                mode=edit_mode,
                                alpha=signed_alpha,
                            )
                            c_name = f"{run_name}_ctrl{int(ridx)}"
                            c_jsonl = str(Path(out_dir) / f"{c_name}.jsonl")
                            c_summary = evaluate(
                                model,
                                tasks,
                                c_jsonl,
                                seed=seed,
                                episode_steps=episode_steps,
                                max_elems=max_elems,
                                max_new_tokens=max_new_tokens,
                                edit=c_edit,
                                strict_action_prompt=strict_action_prompt,
                                run_metadata={
                                    "entrypoint": "sae_validate_control",
                                    "model_alias": model_alias,
                                    "layer": int(layer),
                                    "factor": factor,
                                    "feature_idx": int(ridx),
                                    "mode": edit_mode,
                                    "alpha": float(signed_alpha),
                                    "task_manifest": val_task_manifest,
                                },
                            )
                            control_deltas.append(
                                float(c_summary[steer_key] - c_summary[base_key])
                            )

                    row = {
                        "layer": int(layer),
                        "factor": factor,
                        "feature_idx": int(feature_idx),
                        "mode": edit_mode,
                        "alpha": float(signed_alpha),
                        "base_accuracy": float(summary["base_accuracy"]),
                        "steer_accuracy": float(summary["steer_accuracy"]),
                        "accuracy_delta": float(summary["improvement"]),
                        "base_parse_fail": float(summary["base_parse_fail"]),
                        "steer_parse_fail": float(summary["steer_parse_fail"]),
                        "target_metric_base": float(summary[base_key]),
                        "target_metric_steer": float(summary[steer_key]),
                        "target_metric_delta": target_delta,
                        "output_jsonl": out_jsonl,
                        "control_count": int(len(control_deltas)),
                        "control_mean": float(
                            sum(control_deltas) / max(1, len(control_deltas))
                        )
                        if control_deltas
                        else None,
                        "control_std": float(
                            torch.tensor(control_deltas).std(unbiased=False).item()
                        )
                        if len(control_deltas) > 1
                        else 0.0
                        if control_deltas
                        else None,
                        "paired_gap_vs_control": (
                            target_delta
                            - float(sum(control_deltas) / max(1, len(control_deltas)))
                        )
                        if control_deltas
                        else None,
                    }
                    results.append(row)

    out_json = Path(out_dir) / "sae_validation_summary.json"
    out_json.write_text(
        json.dumps({"generated_at_utc": utc_now_iso(), "rows": results}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return str(out_json)
