from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return dict(by_prompt)


def regroup_equal_generation_budget(by_prompt, k: int):
    """Partition every prompt's first eight cached completions into K-sized groups.

    K=2 creates four disjoint groups per prompt, K=4 creates two, and K=8
    creates one. Consequently every condition consumes the exact same cached
    completions and total generation count.
    """
    if k not in {2, 4, 8}:
        raise ValueError("group size must be one of 2, 4, or 8")
    groups = []
    for prompt_id, rows in by_prompt.items():
        selected = rows[:8]
        if len(selected) != 8:
            raise ValueError(f"Prompt {prompt_id} does not have eight completions")
        for partition_index, start in enumerate(range(0, 8, k)):
            groups.append({
                "source_index": prompt_id,
                "partition_index": partition_index,
                "rows": selected[start : start + k],
            })
    return groups


def assign_prompt_difficulty(by_prompt):
    """Assign equal-count hard/medium/easy bins using K=8 mean reward."""
    ranked = sorted(
        (
            (str(prompt_id), float(np.mean([float(row["reward"]) for row in rows[:8]])))
            for prompt_id, rows in by_prompt.items()
        ),
        key=lambda item: (item[1], item[0]),
    )
    bins = {}
    labels = ("hard", "medium", "easy")
    for label, indices in zip(labels, np.array_split(np.arange(len(ranked)), 3)):
        for index in indices.tolist():
            prompt_id, reward_mean = ranked[index]
            bins[prompt_id] = {
                "difficulty_bin": label,
                "k8_reward_mean": reward_mean,
            }
    return bins


def _group_record(group, k: int, difficulty: dict, eps: float):
    rows = group["rows"]
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=float)
    reward_mean = float(rewards.mean())
    reward_std = float(rewards.std())
    advantages = (
        (rewards - reward_mean) / reward_std
        if reward_std > eps
        else np.zeros_like(rewards)
    )
    prompt_id = str(group["source_index"])
    return {
        "group_size": k,
        "source_index": prompt_id,
        "prompt_id": str(rows[0].get("prompt_id", prompt_id)),
        "partition_index": int(group["partition_index"]),
        "generation_indices": [int(row.get("generation_index", i)) for i, row in enumerate(rows)],
        "difficulty_bin": difficulty[prompt_id]["difficulty_bin"],
        "prompt_k8_reward_mean": difficulty[prompt_id]["k8_reward_mean"],
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "informative": bool(reward_std > eps),
        "relative_advantages": advantages.tolist(),
        "relative_signal_variance": float(advantages.var()),
        "completion_tokens_mean": float(
            np.mean([int(row["completion_tokens"]) for row in rows])
        ),
        "truncation_rate": float(
            np.mean([float(row.get("clipped_at_max", False)) for row in rows])
        ),
        # Keep short readable samples in the report evidence without duplicating
        # the complete cache in every analysis record.
        "completion_previews": [
            {
                "generation_index": int(row.get("generation_index", index)),
                "reward": float(row["reward"]),
                "completion_tokens": int(row["completion_tokens"]),
                "clipped_at_max": bool(row.get("clipped_at_max", False)),
                "completion": str(row.get("completion", ""))[:800],
            }
            for index, row in enumerate(rows)
        ],
    }


def _aggregate(records):
    if not records:
        return {
            "num_groups": 0,
            "num_completions": 0,
            "informative_group_rate": None,
            "uninformative_group_rate": None,
            "mean_within_group_reward_std": None,
            "relative_signal_variance": None,
            "mean_reward": None,
        }
    all_advantages = [
        value for record in records for value in record["relative_advantages"]
    ]
    informative_rate = float(np.mean([record["informative"] for record in records]))
    return {
        "num_groups": len(records),
        "num_completions": int(sum(len(record["relative_advantages"]) for record in records)),
        "informative_group_rate": informative_rate,
        "uninformative_group_rate": 1.0 - informative_rate,
        "mean_within_group_reward_std": float(
            np.mean([record["reward_std"] for record in records])
        ),
        "relative_signal_variance": float(np.var(all_advantages)),
        "mean_group_relative_signal_variance": float(
            np.mean([record["relative_signal_variance"] for record in records])
        ),
        "mean_reward": float(np.mean([record["reward_mean"] for record in records])),
        "mean_completion_tokens": float(
            np.mean([record["completion_tokens_mean"] for record in records])
        ),
        "mean_truncation_rate": float(
            np.mean([record["truncation_rate"] for record in records])
        ),
    }


def run_group_size_study(config_path: str):
    cfg = load_yaml(config_path)
    by_prompt = load_k8_cache(cfg["group_cache"])
    difficulty = assign_prompt_difficulty(by_prompt)
    eps = float(cfg.get("advantage_epsilon", 1e-6))
    result_dir = repo_path(cfg["results_dir"]) / "group_size"
    result_dir.mkdir(parents=True, exist_ok=True)

    conditions = []
    all_group_records = []
    for k in [int(value) for value in cfg["group_sizes"]]:
        groups = regroup_equal_generation_budget(by_prompt, k)
        records = [_group_record(group, k, difficulty, eps) for group in groups]
        all_group_records.extend(records)
        by_difficulty = {
            label: _aggregate(
                [record for record in records if record["difficulty_bin"] == label]
            )
            for label in ("hard", "medium", "easy")
        }
        conditions.append({
            "group_size": k,
            "overall": _aggregate(records),
            "by_prompt_difficulty": by_difficulty,
        })

    completion_counts = {
        condition["group_size"]: condition["overall"]["num_completions"]
        for condition in conditions
    }
    if len(set(completion_counts.values())) != 1:
        raise RuntimeError(f"Unequal generation budgets: {completion_counts}")

    comparison = {
        "cache": str(cfg["group_cache"]),
        "num_cached_prompts": len(by_prompt),
        "fixed_generation_budget": next(iter(completion_counts.values())),
        "regrouping_rule": (
            "For each prompt, use its first eight generation-index-sorted cached "
            "completions and partition them into disjoint consecutive groups of K."
        ),
        "prompt_difficulty_rule": (
            "Rank prompts by mean reward over all eight cached completions, then "
            "split the ranked prompts into three equal-count bins: hard, medium, easy."
        ),
        "informative_group_definition": f"population reward standard deviation > {eps}",
        "relative_signal_definition": (
            "Within each group, (reward - group mean) / population reward standard "
            "deviation; zero for uninformative groups."
        ),
        "conditions": conditions,
    }
    save_json(result_dir / "comparison.json", comparison)
    save_json(result_dir / "prompt_difficulty_bins.json", difficulty)
    write_jsonl(result_dir / "group_records.jsonl", all_group_records)

    ranked_groups = sorted(
        all_group_records,
        key=lambda record: record["reward_std"],
        reverse=True,
    )
    candidates = {
        "most_informative_groups": ranked_groups[:6],
        "uninformative_groups": [
            record for record in ranked_groups if not record["informative"]
        ][:6],
    }
    save_json(result_dir / "qualitative_candidates.json", candidates)
    print(f"Saved group-size comparison to {result_dir / 'comparison.json'}")
    return comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    run_group_size_study(args.config)


if __name__ == "__main__":
    main()
