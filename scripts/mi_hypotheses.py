#!/usr/bin/env python3
"""Define MI hypotheses for WebAgent A/G/S analysis."""

import argparse
import json
from pathlib import Path


HYPOTHESES = [
    {
        "id": "H1-A",
        "target_factor": "A",
        "name": "Action-type localization",
        "statement": "A subset of layers causally mediates action-type selection errors.",
        "falsifier": "Layer ablation does not change action-type error metrics beyond baseline fluctuation.",
        "primary_metrics": [
            "steer_action_type_error_episode_rate",
            "steer_action_type_error_step_rate",
        ],
        "non_target_bounds": {
            "syntax_regression_max_delta": 0.10,
            "grounding_regression_max_delta": 0.10,
        },
    },
    {
        "id": "H1-G",
        "target_factor": "G",
        "name": "Grounding localization",
        "statement": "A subset of layers causally mediates BID grounding failures.",
        "falsifier": "Layer ablation does not change grounding error metrics beyond baseline fluctuation.",
        "primary_metrics": [
            "steer_bid_grounding_error_episode_rate",
            "steer_bid_grounding_error_step_rate",
        ],
        "non_target_bounds": {
            "syntax_regression_max_delta": 0.10,
            "action_type_regression_max_delta": 0.10,
        },
    },
    {
        "id": "H1-S",
        "target_factor": "S",
        "name": "Syntax localization",
        "statement": "A subset of layers causally mediates syntax/parse validity.",
        "falsifier": "Layer ablation does not change syntax error metrics beyond baseline fluctuation.",
        "primary_metrics": [
            "steer_syntax_error_episode_rate",
            "steer_syntax_error_step_rate",
        ],
        "non_target_bounds": {
            "grounding_regression_max_delta": 0.10,
            "action_type_regression_max_delta": 0.10,
        },
    },
    {
        "id": "H2",
        "target_factor": "A/G/S",
        "name": "Specificity over non-target factors",
        "statement": "Targeted perturbation effect should exceed non-target perturbation effects.",
        "falsifier": "Observed deltas are uniform across A/G/S or dominated by non-target regressions.",
        "primary_metrics": [
            "delta",
            "base_action_type_error_episode_rate",
            "steer_action_type_error_episode_rate",
            "base_bid_grounding_error_episode_rate",
            "steer_bid_grounding_error_episode_rate",
            "base_syntax_error_episode_rate",
            "steer_syntax_error_episode_rate",
        ],
        "non_target_bounds": {
            "max_non_target_delta": 0.10,
        },
    },
    {
        "id": "H3",
        "target_factor": "success/parse-fail",
        "name": "Non-collapse criterion",
        "statement": "Mechanism perturbation should not induce catastrophic parse-fail collapse.",
        "falsifier": "Parse-fail rises sharply with no interpretable factor-specific insight.",
        "primary_metrics": [
            "base_parse_fail",
            "steer_parse_fail",
            "base_accuracy",
            "steer_accuracy",
        ],
        "non_target_bounds": {
            "parse_fail_delta_max": 0.20,
        },
    },
]


def main():
    ap = argparse.ArgumentParser(description="Emit MI hypotheses registry for A/G/S")
    ap.add_argument("--out-json", default="runtime_state/mi_hypotheses_17b.json")
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_focus": "qwen3-1.7b",
        "decomposition": ["A", "G", "S"],
        "hypotheses": HYPOTHESES,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.print:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps({"out_json": str(out), "hypothesis_count": len(HYPOTHESES)}))


if __name__ == "__main__":
    main()
