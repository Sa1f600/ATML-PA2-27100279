from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_adapter


def _kl_label(beta: float):
    return f"kl_{beta:.2f}".replace(".", "p")


def _trajectory_summary(adapter: Path):
    rows = read_jsonl(adapter / "train_metrics.jsonl")
    if not rows:
        return {}
    keys = (
        "learned_reward",
        "sampled_kl_per_token",
        "entropy",
        "response_length_tokens",
        "policy_loss",
        "value_loss",
        "gradient_norm",
    )
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
            "first": float(rows[0][key]),
            "last": float(rows[-1][key]),
            "change": float(rows[-1][key] - rows[0][key]),
        }
        for key in keys
    }


def _cross_condition_candidates(records_by_condition: dict[str, list[dict]]):
    """Select fixed-prompt examples for the required manual quality comparison."""
    aligned = {}
    for condition, records in records_by_condition.items():
        for row in records:
            aligned.setdefault(str(row["prompt_id"]), {})[condition] = row

    candidates = []
    for prompt_id, condition_rows in aligned.items():
        if len(condition_rows) < 2:
            continue
        rows = list(condition_rows.values())
        reward_spread = max(row["learned_reward"] for row in rows) - min(
            row["learned_reward"] for row in rows
        )
        kl_spread = max(row["sampled_kl_sequence"] for row in rows) - min(
            row["sampled_kl_sequence"] for row in rows
        )
        candidates.append({
            "prompt_id": prompt_id,
            "reward_spread": float(reward_spread),
            "kl_spread": float(kl_spread),
            "conditions": condition_rows,
        })
    candidates.sort(key=lambda row: (row["reward_spread"], row["kl_spread"]), reverse=True)
    return {
        "selection_rule": (
            "Fixed held-out prompts ranked by reward spread across KL-beta conditions. "
            "Manually inspect the compact responses to identify one reward/quality agreement "
            "and one disagreement; learned reward alone is not a quality label."
        ),
        "candidates": candidates[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--skip-evaluation", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = [float(value) for value in cfg["kl_values"]]
    result_root = repo_path(cfg["results_dir"]) / "kl_ablation"
    result_root.mkdir(parents=True, exist_ok=True)

    conditions = []
    records_by_condition = {}
    for beta in betas:
        label = _kl_label(beta)
        adapter = repo_path("outputs/task2_ppo") / label
        print(f"Running matched PPO KL-pressure condition {label}")

        if args.skip_training:
            if not adapter.exists():
                raise FileNotFoundError(
                    f"--skip-training was set but adapter does not exist: {adapter}"
                )
            training_path = adapter / "training_summary.json"
            training = load_json(training_path) if training_path.exists() else None
        else:
            training = run_ppo(
                args.config,
                output=str(adapter),
                updates=int(cfg["fork_updates"]),
                clip_epsilon=float(cfg["clip_epsilon"]),
                kl_beta=beta,
                run_name=label,
            )

        evaluation_dir = result_root / label
        evaluation_path = evaluation_dir / "summary.json"
        generations_path = evaluation_dir / "generations.jsonl"
        if args.skip_evaluation:
            if not evaluation_path.exists() or not generations_path.exists():
                raise FileNotFoundError(
                    "--skip-evaluation requires existing summary.json and generations.jsonl "
                    f"under {evaluation_dir}"
                )
            evaluation = load_json(evaluation_path)
            records = read_jsonl(generations_path)
        else:
            evaluated = evaluate_adapter(
                args.config,
                str(adapter),
                label,
                output_dir=evaluation_dir,
            )
            evaluation = evaluated["summary"]
            records = evaluated["records"]
        records_by_condition[label] = records

        conditions.append({
            "condition": label,
            "kl_beta": beta,
            "budget": "matched_short_continuation",
            "training": training,
            "training_trajectory": _trajectory_summary(adapter),
            "evaluation": evaluation,
        })
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = {
        "controlled_variables": [
            "supplied policy and value midpoint checkpoints",
            "prompt order and seed",
            "number of updates and PPO epochs",
            "clip epsilon and reward model",
            "generation settings",
            "held-out prompts and decoding settings",
        ],
        "conditions": conditions,
    }
    save_json(result_root / "comparison.json", comparison)
    save_json(
        result_root / "qualitative_comparison_candidates.json",
        _cross_condition_candidates(records_by_condition),
    )
    print(f"Saved KL-pressure comparison to {result_root / 'comparison.json'}")


if __name__ == "__main__":
    main()
