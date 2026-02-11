from agent_core import (
    build_prompt,
    classify_action_step,
    derive_episode_seed,
    make_miniwob_env,
)


def run_episode(
    env,
    model,
    seed,
    max_steps,
    max_elems,
    max_new_tokens,
    edit=None,
):
    obs, _ = env.reset(seed=seed)
    outputs, actions, errors = [], [], []
    step_analysis = []
    total_reward = 0.0
    success = False

    for step_idx in range(int(max_steps)):
        prompt = build_prompt(obs, max_elems=max_elems, strict_action_prompt=True)
        output = model.generate(
            prompt, max_new_tokens=max_new_tokens, edit=edit, deterministic=True
        )
        action = str(output or "").strip()
        outputs.append(output)
        actions.append(action)

        try:
            obs, reward, terminated, truncated, _info = env.step(action)
            reward = float(reward)
            done = bool(terminated or truncated)
            err = (
                str(obs.get("last_action_error", "") or "")
                if isinstance(obs, dict)
                else ""
            )
        except Exception as exc:
            reward = 0.0
            done = True
            err = f"step_exception:{type(exc).__name__}"

        total_reward += reward
        success = success or (reward > 0)
        errors.append(err)

        cls = classify_action_step(action, err)
        cls.update({"step_idx": step_idx, "error": err, "action": action})
        step_analysis.append(cls)
        if done:
            break

    last_error = ""
    for err in reversed(errors):
        if err:
            last_error = err
            break

    syntax_error_steps = sum(
        1 for x in step_analysis if not bool(x.get("syntax_ok", True))
    )
    action_type_error_steps = sum(
        1
        for x in step_analysis
        if bool(x.get("action_type_known", True))
        and (not bool(x.get("action_type_ok", True)))
    )
    bid_grounding_error_steps = sum(
        1
        for x in step_analysis
        if bool(x.get("bid_grounding_known", True))
        and (not bool(x.get("bid_grounding_ok", True)))
    )

    return {
        "steps": len(actions),
        "total_reward": total_reward,
        "success": bool(success),
        "error": last_error,
        "last_output": outputs[-1] if outputs else "",
        "last_action": actions[-1] if actions else "",
        "action_type_error_episode": bool(action_type_error_steps > 0),
        "bid_grounding_error_episode": bool(bid_grounding_error_steps > 0),
        "syntax_error_episode": bool(syntax_error_steps > 0),
    }


def _safe_rate(n, d):
    return n / max(1, d)


def evaluate(
    model,
    tasks,
    seed=0,
    episodes_per_task=3,
    episode_steps=8,
    max_elems=80,
    max_new_tokens=80,
    edit=None,
):
    # Paired comparison: baseline first, then steered variant on identical seeds.
    tasks = list(tasks)
    episodes_per_task = int(episodes_per_task)

    base_hits = 0
    base_total = 0
    base_parse_fail = 0
    base_a_ep = base_g_ep = base_s_ep = 0

    steer_hits = 0
    steer_total = 0
    steer_parse_fail = 0
    steer_a_ep = steer_g_ep = steer_s_ep = 0

    samples = []

    for task in tasks:
        env = make_miniwob_env(task)
        for episode_idx in range(episodes_per_task):
            ep_seed = derive_episode_seed(seed, "eval", task, episode_idx)
            base = run_episode(
                env,
                model,
                ep_seed,
                max_steps=episode_steps,
                max_elems=max_elems,
                max_new_tokens=max_new_tokens,
                edit=None,
            )
            base_hits += int(base["success"])
            base_total += 1
            base_parse_fail += int(bool(base["error"]))
            base_a_ep += int(base["action_type_error_episode"])
            base_g_ep += int(base["bid_grounding_error_episode"])
            base_s_ep += int(base["syntax_error_episode"])

            if edit is None and len(samples) < 3:
                samples.append(
                    {
                        "task": task,
                        "seed": ep_seed,
                        "success": base["success"],
                        "error": base["error"],
                        "last_action": base["last_action"],
                    }
                )

            if edit is not None:
                steer = run_episode(
                    env,
                    model,
                    ep_seed,
                    max_steps=episode_steps,
                    max_elems=max_elems,
                    max_new_tokens=max_new_tokens,
                    edit=edit,
                )
                steer_hits += int(steer["success"])
                steer_total += 1
                steer_parse_fail += int(bool(steer["error"]))
                steer_a_ep += int(steer["action_type_error_episode"])
                steer_g_ep += int(steer["bid_grounding_error_episode"])
                steer_s_ep += int(steer["syntax_error_episode"])

                if len(samples) < 3:
                    samples.append(
                        {
                            "task": task,
                            "seed": ep_seed,
                            "base_success": base["success"],
                            "steer_success": steer["success"],
                            "base_error": base["error"],
                            "steer_error": steer["error"],
                            "base_last_action": base["last_action"],
                            "steer_last_action": steer["last_action"],
                        }
                    )
        env.close()

    summary = {
        "model": model.model_alias,
        "tasks": tasks,
        "episodes_per_task": episodes_per_task,
        "total_episodes": len(tasks) * episodes_per_task,
        "base_accuracy": _safe_rate(base_hits, base_total),
        "base_parse_fail": _safe_rate(base_parse_fail, base_total),
        "base_action_type_error_episode_rate": _safe_rate(base_a_ep, base_total),
        "base_bid_grounding_error_episode_rate": _safe_rate(base_g_ep, base_total),
        "base_syntax_error_episode_rate": _safe_rate(base_s_ep, base_total),
        "samples": samples,
    }

    if edit is not None:
        steer_accuracy = _safe_rate(steer_hits, steer_total)
        summary.update(
            {
                "steer_accuracy": steer_accuracy,
                "improvement": steer_accuracy - summary["base_accuracy"],
                "steer_parse_fail": _safe_rate(steer_parse_fail, steer_total),
                "steer_action_type_error_episode_rate": _safe_rate(
                    steer_a_ep, steer_total
                ),
                "steer_bid_grounding_error_episode_rate": _safe_rate(
                    steer_g_ep, steer_total
                ),
                "steer_syntax_error_episode_rate": _safe_rate(steer_s_ep, steer_total),
            }
        )

    return summary
