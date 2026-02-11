import json

from agent_core import (
    build_prompt,
    classify_action_step,
    derive_episode_seed,
    make_miniwob_env,
    utc_now_iso,
)


def run_episode(
    env,
    model,
    seed,
    max_steps,
    max_elems,
    max_new_tokens,
    edit=None,
    strict_action_prompt=False,
):
    obs, _ = env.reset(seed=seed)
    outputs, actions, errors = [], [], []
    step_analysis = []
    total_reward = 0.0
    success = False

    for step_idx in range(int(max_steps)):
        prompt = build_prompt(
            obs, max_elems=max_elems, strict_action_prompt=strict_action_prompt
        )
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
    action_type_unknown_steps = sum(
        1 for x in step_analysis if not bool(x.get("action_type_known", True))
    )
    bid_grounding_unknown_steps = sum(
        1 for x in step_analysis if not bool(x.get("bid_grounding_known", True))
    )

    correction_events = 0
    for i in range(1, len(step_analysis)):
        if bool(step_analysis[i - 1].get("error")) and (
            not bool(step_analysis[i].get("error"))
        ):
            correction_events += 1

    return {
        "outputs": outputs,
        "actions": actions,
        "steps": len(actions),
        "total_reward": total_reward,
        "success": bool(success),
        "error": last_error,
        "step_analysis": step_analysis,
        "syntax_error_steps": syntax_error_steps,
        "action_type_error_steps": action_type_error_steps,
        "bid_grounding_error_steps": bid_grounding_error_steps,
        "action_type_unknown_steps": action_type_unknown_steps,
        "bid_grounding_unknown_steps": bid_grounding_unknown_steps,
        "syntax_error_episode": bool(syntax_error_steps > 0),
        "action_type_error_episode": bool(action_type_error_steps > 0),
        "bid_grounding_error_episode": bool(bid_grounding_error_steps > 0),
        "correction_events": correction_events,
    }


def evaluate(
    model,
    tasks,
    out_path,
    seed=0,
    episode_steps=10,
    max_elems=80,
    max_new_tokens=80,
    edit=None,
    strict_action_prompt=False,
    run_metadata=None,
):
    tasks = list(tasks)
    steps_per_task = 3

    base_hits = 0
    steer_hits = 0
    base_total = 0
    steer_total = 0
    base_parse_fail = 0
    steer_parse_fail = 0

    base_a_ep = base_g_ep = base_s_ep = 0
    steer_a_ep = steer_g_ep = steer_s_ep = 0
    base_a_steps = base_g_steps = base_s_steps = 0
    steer_a_steps = steer_g_steps = steer_s_steps = 0
    base_steps_total = steer_steps_total = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for task in tasks:
            env = make_miniwob_env(task)
            for episode_idx in range(steps_per_task):
                ep_seed = derive_episode_seed(seed, "eval", task, episode_idx)
                base = run_episode(
                    env,
                    model,
                    ep_seed,
                    max_steps=episode_steps,
                    max_elems=max_elems,
                    max_new_tokens=max_new_tokens,
                    edit=None,
                    strict_action_prompt=strict_action_prompt,
                )

                base_hits += int(base["success"])
                base_total += 1
                base_parse_fail += int(bool(base["error"]))
                base_a_ep += int(base["action_type_error_episode"])
                base_g_ep += int(base["bid_grounding_error_episode"])
                base_s_ep += int(base["syntax_error_episode"])
                base_a_steps += int(base["action_type_error_steps"])
                base_g_steps += int(base["bid_grounding_error_steps"])
                base_s_steps += int(base["syntax_error_steps"])
                base_steps_total += int(base["steps"])

                record = {
                    "task": task,
                    "seed": ep_seed,
                    "base_output": base["outputs"][-1] if base["outputs"] else "",
                    "base_outputs": base["outputs"],
                    "base_action": base["actions"][-1] if base["actions"] else "",
                    "base_actions": base["actions"],
                    "base_steps": base["steps"],
                    "base_total_reward": base["total_reward"],
                    "base_success": base["success"],
                    "base_error": base["error"],
                    "base_error_episode": bool(base["error"]),
                    "base_action_type_error_episode": base["action_type_error_episode"],
                    "base_bid_grounding_error_episode": base[
                        "bid_grounding_error_episode"
                    ],
                    "base_syntax_error_episode": base["syntax_error_episode"],
                    "base_action_type_error_steps": base["action_type_error_steps"],
                    "base_bid_grounding_error_steps": base["bid_grounding_error_steps"],
                    "base_syntax_error_steps": base["syntax_error_steps"],
                    "base_action_type_unknown_steps": base["action_type_unknown_steps"],
                    "base_bid_grounding_unknown_steps": base[
                        "bid_grounding_unknown_steps"
                    ],
                    "base_correction_events": base["correction_events"],
                    "base_step_analysis": base["step_analysis"],
                }

                if edit is not None:
                    steer = run_episode(
                        env,
                        model,
                        ep_seed,
                        max_steps=episode_steps,
                        max_elems=max_elems,
                        max_new_tokens=max_new_tokens,
                        edit=edit,
                        strict_action_prompt=strict_action_prompt,
                    )
                    steer_hits += int(steer["success"])
                    steer_total += 1
                    steer_parse_fail += int(bool(steer["error"]))
                    steer_a_ep += int(steer["action_type_error_episode"])
                    steer_g_ep += int(steer["bid_grounding_error_episode"])
                    steer_s_ep += int(steer["syntax_error_episode"])
                    steer_a_steps += int(steer["action_type_error_steps"])
                    steer_g_steps += int(steer["bid_grounding_error_steps"])
                    steer_s_steps += int(steer["syntax_error_steps"])
                    steer_steps_total += int(steer["steps"])

                    record.update(
                        {
                            "steer_output": steer["outputs"][-1]
                            if steer["outputs"]
                            else "",
                            "steer_outputs": steer["outputs"],
                            "steer_action": steer["actions"][-1]
                            if steer["actions"]
                            else "",
                            "steer_actions": steer["actions"],
                            "steer_steps": steer["steps"],
                            "steer_total_reward": steer["total_reward"],
                            "steer_success": steer["success"],
                            "steer_error": steer["error"],
                            "steer_error_episode": bool(steer["error"]),
                            "steer_action_type_error_episode": steer[
                                "action_type_error_episode"
                            ],
                            "steer_bid_grounding_error_episode": steer[
                                "bid_grounding_error_episode"
                            ],
                            "steer_syntax_error_episode": steer["syntax_error_episode"],
                            "steer_action_type_error_steps": steer[
                                "action_type_error_steps"
                            ],
                            "steer_bid_grounding_error_steps": steer[
                                "bid_grounding_error_steps"
                            ],
                            "steer_syntax_error_steps": steer["syntax_error_steps"],
                            "steer_action_type_unknown_steps": steer[
                                "action_type_unknown_steps"
                            ],
                            "steer_bid_grounding_unknown_steps": steer[
                                "bid_grounding_unknown_steps"
                            ],
                            "steer_correction_events": steer["correction_events"],
                            "steer_step_analysis": steer["step_analysis"],
                        }
                    )

                f.write(json.dumps(record) + "\n")
            env.close()

    base_acc = base_hits / max(1, base_total)
    steer_acc = steer_hits / max(1, steer_total) if edit is not None else 0.0
    summary = {
        "base_accuracy": base_acc,
        "steer_accuracy": steer_acc,
        "improvement": steer_acc - base_acc,
        "base_parse_fail": base_parse_fail / max(1, base_total),
        "steer_parse_fail": (steer_parse_fail / max(1, steer_total))
        if edit is not None
        else 0.0,
        "total_episodes": len(tasks) * steps_per_task,
        "base_action_type_error_episode_rate": base_a_ep / max(1, base_total),
        "base_bid_grounding_error_episode_rate": base_g_ep / max(1, base_total),
        "base_syntax_error_episode_rate": base_s_ep / max(1, base_total),
        "base_action_type_error_step_rate": base_a_steps / max(1, base_steps_total),
        "base_bid_grounding_error_step_rate": base_g_steps / max(1, base_steps_total),
        "base_syntax_error_step_rate": base_s_steps / max(1, base_steps_total),
        "steer_action_type_error_episode_rate": (steer_a_ep / max(1, steer_total))
        if edit is not None
        else 0.0,
        "steer_bid_grounding_error_episode_rate": (steer_g_ep / max(1, steer_total))
        if edit is not None
        else 0.0,
        "steer_syntax_error_episode_rate": (steer_s_ep / max(1, steer_total))
        if edit is not None
        else 0.0,
        "steer_action_type_error_step_rate": (steer_a_steps / max(1, steer_steps_total))
        if edit is not None
        else 0.0,
        "steer_bid_grounding_error_step_rate": (
            steer_g_steps / max(1, steer_steps_total)
        )
        if edit is not None
        else 0.0,
        "steer_syntax_error_step_rate": (steer_s_steps / max(1, steer_steps_total))
        if edit is not None
        else 0.0,
    }
    meta = {
        "generated_at_utc": utc_now_iso(),
        "output_jsonl": out_path,
        "summary": summary,
        "task_count": len(tasks),
        "tasks": list(tasks),
        "seed": int(seed),
        "run_metadata": run_metadata or {},
    }
    with open(f"{out_path}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return summary
