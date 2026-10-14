from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_adapter


CONDITIONS = {
    "canonical_grpo": "grpo",
    "dr_grpo": "dr_grpo",
}


def _length_conditioned_statistics(adapter: Path):
    rows = read_jsonl(adapter / "rollouts.jsonl")
    included = [row for row in rows if row.get("training_included", True)]
    by_bin = {}
    for label in ("short", "medium", "long"):
        selected = [row for row in included if row["length_bin"] == label]
        if not selected:
            by_bin[label] = {
                "num_completions": 0,
                "mean_response_tokens": None,
                "mean_absolute_advantage": None,
                "mean_absolute_surrogate_coefficient_mass": None,
                "mean_reward": None,
            }
            continue
        by_bin[label] = {
            "num_completions": len(selected),
            "mean_response_tokens": float(
                np.mean([row["response_tokens"] for row in selected])
            ),
            "mean_absolute_advantage": float(
                np.mean([abs(row["group_advantage"]) for row in selected])
            ),
            "mean_absolute_surrogate_coefficient_mass": float(
                np.mean(
                    [row["absolute_surrogate_coefficient_mass"] for row in selected]
                )
            ),
            "mean_reward": float(
                np.mean([row["learned_reward"] for row in selected])
            ),
            "mean_sampled_kl_per_response": float(
                np.mean([row["sampled_kl_sequence"] for row in selected])
            ),
        }
    correlation = safe_corr(
        [row["response_tokens"] for row in included],
        [row["absolute_surrogate_coefficient_mass"] for row in included],
    )
    return {
        "length_bins": {
            "short": "response_tokens <= 0.25 * max_completion_length",
            "medium": "0.25 * max_completion_length < response_tokens <= 0.50 * max_completion_length",
            "long": "response_tokens > 0.50 * max_completion_length",
        },
        "statistic_definition": (
            "Absolute surrogate coefficient mass = |group advantage| times "
            "included response tokens divided by the normalization denominator. "
            "Canonical GRPO uses realized length; Dr. GRPO uses the fixed maximum."
        ),
        "overall_length_coefficient_correlation": (
            float(correlation) if np.isfinite(correlation) else None
        ),
        "by_length_bin": by_bin,
    }


def _cross_condition_candidates(records_by_condition: dict[str, list[dict]]):
    aligned = {}
    for condition, records in records_by_condition.items():
        for row in records:
            aligned.setdefault(str(row["prompt_id"]), {})[condition] = row

    candidates = []
    for prompt_id, condition_rows in aligned.items():
        if len(condition_rows) != len(CONDITIONS):
            continue
        rows = list(condition_rows.values())
        candidates.append({
            "prompt_id": prompt_id,
            "reward_spread": float(
                max(row["learned_reward"] for row in rows)
                - min(row["learned_reward"] for row in rows)
            ),
            "response_length_spread": int(
                max(row["response_tokens"] for row in rows)
                - min(row["response_tokens"] for row in rows)
            ),
            "conditions": condition_rows,
        })
    candidates.sort(
        key=lambda row: (row["response_length_spread"], row["reward_spread"]),
        reverse=True,
    )
    return {
        "selection_rule": (
            "Fixed held-out prompts ranked first by response-length spread and "
            "then reward spread across canonical and Dr. GRPO."
        ),
        "candidates": candidates[:10],
    }


def _load_completed_condition(cfg: dict, label: str):
    adapter = repo_path("outputs/task3_grpo") / label
    result_dir = repo_path(cfg["results_dir"]) / "normalization" / label
    required = [
        adapter / "training_summary.json",
        adapter / "rollouts.jsonl",
        result_dir / "summary.json",
        result_dir / "generations.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return None, missing
    training = load_json(adapter / "training_summary.json")
    evaluation = load_json(result_dir / "summary.json")
    records = read_jsonl(result_dir / "generations.jsonl")
    return {
        "condition": label,
        "loss_type": CONDITIONS[label],
        "budget": "matched_short_continuation",
        "training": training,
        "length_conditioned_training_statistics": _length_conditioned_statistics(
            adapter
        ),
        "evaluation": evaluation,
        "records": records,
    }, []


def _write_combined_comparison(cfg: dict):
    conditions = []
    records_by_condition = {}
    missing = []
    for label in CONDITIONS:
        condition, condition_missing = _load_completed_condition(cfg, label)
        missing.extend(condition_missing)
        if condition is not None:
            records_by_condition[label] = condition.pop("records")
            conditions.append(condition)
    if missing:
        print("Normalization comparison is not complete yet; missing:")
        for path in missing:
            print(f"  {path}")
        return None

    result_root = repo_path(cfg["results_dir"]) / "normalization"
    comparison = {
        "controlled_variables": [
            "supplied GRPO midpoint checkpoint",
            "prompt order and seed",
            "number of updates and K completions per prompt",
            "reward model, KL beta, and clipping epsilon",
            "generation settings and maximum completion length",
            "held-out prompts and evaluation decoding settings",
        ],
        "matched_generation_capacity_tokens": int(cfg["fork_updates"])
        * int(cfg["prompts_per_update"])
        * int(cfg["num_generations"])
        * int(cfg["max_completion_length"]),
        "realized_generated_tokens": {
            condition["condition"]: int(condition["training"]["generated_tokens"])
            for condition in conditions
        },
        "budget_note": (
            "Both forks have identical updates, prompts, K, and maximum response "
            "length. Realized token counts are also reported because EOS can make "
            "sampled responses end before the common capacity limit."
        ),
        "conditions": conditions,
    }
    save_json(result_root / "comparison.json", comparison)
    save_json(
        result_root / "qualitative_comparison_candidates.json",
        _cross_condition_candidates(records_by_condition),
    )
    print(f"Saved normalization comparison to {result_root / 'comparison.json'}")
    return comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument(
        "--condition",
        choices=["all", *CONDITIONS.keys()],
        default="all",
        help="Run both conditions or one condition for safer split Colab execution.",
    )
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--skip-evaluation", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    labels = list(CONDITIONS) if args.condition == "all" else [args.condition]
    result_root = repo_path(cfg["results_dir"]) / "normalization"
    result_root.mkdir(parents=True, exist_ok=True)

    for label in labels:
        loss_type = CONDITIONS[label]
        adapter = repo_path("outputs/task3_grpo") / label
        evaluation_dir = result_root / label
        print(f"Running matched normalization condition {label}")
        if args.skip_training:
            if not (adapter / "training_summary.json").exists():
                raise FileNotFoundError(
                    f"--skip-training requires a completed adapter at {adapter}"
                )
        else:
            run_grpo(
                args.config,
                output=str(adapter),
                updates=int(cfg["fork_updates"]),
                loss_type=loss_type,
                run_name=label,
            )

        if args.skip_evaluation:
            if not (evaluation_dir / "summary.json").exists() or not (
                evaluation_dir / "generations.jsonl"
            ).exists():
                raise FileNotFoundError(
                    f"--skip-evaluation requires completed results under {evaluation_dir}"
                )
        else:
            evaluate_adapter(
                args.config,
                str(adapter),
                label,
                output_dir=evaluation_dir,
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_combined_comparison(cfg)


if __name__ == "__main__":
    main()
