from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import set_seed

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward


VARIANT_ORDER = (
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
)
EXPECTED_VARIANTS = set(VARIANT_ORDER)

# A is the manually validated, diagnostically better response in every pair.
PAIR_SPECS = (
    {
        "perturbation": "reasoning_corruption",
        "better": "clean_correct",
        "worse": "corrupt_reasoning_correct_final",
        "role": "reasoning_sensitivity",
    },
    {
        "perturbation": "wrong_final",
        "better": "clean_correct",
        "worse": "good_reasoning_wrong_final",
        "role": "outcome_sensitivity",
    },
    {
        "perturbation": "persuasive_filler",
        "better": "clean_correct",
        "worse": "persuasive_filler_correct",
        "role": "filler_susceptibility",
    },
    {
        "perturbation": "gold_distractor",
        "better": "clean_correct",
        "worse": "gold_distractor_wrong_final",
        "role": "distractor_robustness",
    },
)


def load_diagnostic_groups(path) -> dict[str, dict[str, dict]]:
    rows = read_jsonl(path)
    by_problem: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        pid = str(row["problem_id"])
        variant = str(row["variant_type"])
        if variant not in EXPECTED_VARIANTS:
            raise ValueError(f"Problem {pid} has unexpected variant: {variant}")
        if variant in by_problem[pid]:
            raise ValueError(f"Problem {pid} has duplicate variant: {variant}")
        by_problem[pid][variant] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    if not by_problem:
        raise ValueError("Task 5 diagnostic set is empty")
    return dict(by_problem)


def _problem_text(row: dict) -> str:
    for key in ("problem", "question", "prompt"):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key]
    for message in reversed(prompt_messages(row)):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError("Diagnostic row has no problem text")


def _response_text(row: dict) -> str:
    for key in ("response", "completion", "candidate"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError("Diagnostic row has no response text")


def _gold_final(row: dict) -> str:
    for key in ("gold_final", "gold_answer", "answer"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError("Diagnostic row has no gold final answer")


def preference_from_scores(better_score: float, worse_score: float, tol: float = 1e-12) -> str:
    if better_score > worse_score + tol:
        return "A"
    if worse_score > better_score + tol:
        return "B"
    return "TIE"


def preference_flags(preference: str) -> dict:
    if preference not in {"A", "B", "TIE"}:
        raise ValueError(f"Invalid preference: {preference}")
    return {
        "better": preference == "A",
        "tie": preference == "TIE",
        "wrong": preference == "B",
    }


def aggregate_preferences(rows: list[dict], field: str) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate an empty controlled-pair set")
    counts = Counter(str(row[field]) for row in rows)
    invalid = set(counts).difference({"A", "B", "TIE"})
    if invalid:
        raise ValueError(f"Invalid preferences in {field}: {sorted(invalid)}")
    total = len(rows)
    return {
        "num_pairs": total,
        "better_response_rate": float(counts["A"] / total),
        "tie_rate": float(counts["TIE"] / total),
        "wrong_preference_rate": float(counts["B"] / total),
        "better_count": int(counts["A"]),
        "tie_count": int(counts["TIE"]),
        "wrong_count": int(counts["B"]),
    }


def summarize_diagnostics(score_rows: list[dict], pair_rows: list[dict]) -> dict:
    variant_summary = {}
    for variant in VARIANT_ORDER:
        subset = [row for row in score_rows if row["variant_type"] == variant]
        if not subset:
            raise ValueError(f"No diagnostic scores for variant: {variant}")
        group_rewards = np.asarray([row["rlaif_group_reward"] for row in subset], dtype=float)
        variant_summary[variant] = {
            "num_responses": len(subset),
            "rlvr_exact_reward_mean": float(
                np.mean([row["rlvr_exact_reward"] for row in subset])
            ),
            "rlaif_group_reward_mean": float(group_rewards.mean()),
            "rlaif_group_reward_std": float(group_rewards.std()),
        }

    controlled = {}
    for mechanism, field in (
        ("rlvr", "rlvr_preference"),
        ("rlaif", "rlaif_preference"),
    ):
        by_perturbation = {}
        for spec in PAIR_SPECS:
            subset = [
                row for row in pair_rows if row["perturbation"] == spec["perturbation"]
            ]
            by_perturbation[spec["perturbation"]] = aggregate_preferences(subset, field)
        controlled[mechanism] = {
            "overall": aggregate_preferences(pair_rows, field),
            "by_perturbation": by_perturbation,
        }

    return {
        "num_problems": len({row["problem_id"] for row in score_rows}),
        "num_responses": len(score_rows),
        "num_controlled_pairs": len(pair_rows),
        "variant_scores": variant_summary,
        "controlled_pair_results": controlled,
        "sensitivity": {
            "S_reason": {
                mechanism: controlled[mechanism]["by_perturbation"]["reasoning_corruption"][
                    "better_response_rate"
                ]
                for mechanism in ("rlvr", "rlaif")
            },
            "S_outcome": {
                mechanism: controlled[mechanism]["by_perturbation"]["wrong_final"][
                    "better_response_rate"
                ]
                for mechanism in ("rlvr", "rlaif")
            },
            "filler_susceptibility_wrong_preference_rate": {
                mechanism: controlled[mechanism]["by_perturbation"]["persuasive_filler"][
                    "wrong_preference_rate"
                ]
                for mechanism in ("rlvr", "rlaif")
            },
            "gold_distractor_robustness": {
                mechanism: controlled[mechanism]["by_perturbation"]["gold_distractor"][
                    "better_response_rate"
                ]
                for mechanism in ("rlvr", "rlaif")
            },
        },
    }


def qualitative_candidates(pair_rows: list[dict]) -> list[dict]:
    selected = []
    for spec in PAIR_SPECS:
        pool = [row for row in pair_rows if row["perturbation"] == spec["perturbation"]]
        pool.sort(
            key=lambda row: (
                row["rlvr_preference"] == row["rlaif_preference"],
                row["rlaif_preference"] != "TIE",
                str(row["problem_id"]),
            )
        )
        if pool:
            selected.append(pool[0])
    return selected


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 5 diagnostic AI judging requires a CUDA GPU runtime")
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["seed"]))
    judge = PairwiseAIJudge(
        cfg,
        cfg.get("task5_judge_cache", "results/task5_feedback/pairwise_judge_cache.json"),
    )
    score_rows = []
    pair_rows = []
    try:
        for number, (problem_id, variants) in enumerate(groups.items(), start=1):
            problem_texts = {_problem_text(row) for row in variants.values()}
            gold_values = {_gold_final(row) for row in variants.values()}
            if len(problem_texts) != 1 or len(gold_values) != 1:
                raise ValueError(f"Problem {problem_id} has inconsistent problem/gold fields")
            problem = next(iter(problem_texts))
            gold = next(iter(gold_values))
            responses = [_response_text(variants[name]) for name in VARIANT_ORDER]
            rlvr_rewards = [exact_reward(response, gold) for response in responses]
            rlaif_rewards = judge.group_rewards(problem, responses)
            for variant, response, rlvr_score, rlaif_score in zip(
                VARIANT_ORDER, responses, rlvr_rewards, rlaif_rewards
            ):
                score_rows.append(
                    {
                        "problem_id": problem_id,
                        "problem": problem,
                        "gold_final": gold,
                        "variant_type": variant,
                        "response": response,
                        "rlvr_exact_reward": float(rlvr_score),
                        "rlaif_group_reward": float(rlaif_score),
                    }
                )

            for spec in PAIR_SPECS:
                better = _response_text(variants[spec["better"]])
                worse = _response_text(variants[spec["worse"]])
                better_rlvr = exact_reward(better, gold)
                worse_rlvr = exact_reward(worse, gold)
                pair_rows.append(
                    {
                        "problem_id": problem_id,
                        "problem": problem,
                        "gold_final": gold,
                        "perturbation": spec["perturbation"],
                        "diagnostic_role": spec["role"],
                        "better_variant": spec["better"],
                        "worse_variant": spec["worse"],
                        "better_response": better,
                        "worse_response": worse,
                        "better_rlvr_reward": float(better_rlvr),
                        "worse_rlvr_reward": float(worse_rlvr),
                        "rlvr_preference": preference_from_scores(better_rlvr, worse_rlvr),
                        "rlaif_preference": judge.compare(problem, better, worse),
                    }
                )
            write_jsonl(outdir / "diagnostic_scores.jsonl", score_rows)
            write_jsonl(outdir / "controlled_pairs.jsonl", pair_rows)
            print(f"Scored diagnostic problem {number}/{len(groups)}")
    finally:
        del judge
        gc.collect()
        torch.cuda.empty_cache()

    summary = summarize_diagnostics(score_rows, pair_rows)
    _write_json(outdir / "summary.json", summary)
    pd.DataFrame(
        [
            {
                "mechanism": mechanism,
                "perturbation": perturbation,
                **metrics,
            }
            for mechanism, mechanism_data in summary["controlled_pair_results"].items()
            for perturbation, metrics in mechanism_data["by_perturbation"].items()
        ]
    ).to_csv(outdir / "controlled_pair_summary.csv", index=False)
    _write_json(outdir / "qualitative_candidates.json", qualitative_candidates(pair_rows))
    print(pd.read_csv(outdir / "controlled_pair_summary.csv").to_string(index=False))
    print("Saved Task 5 controlled diagnostic analysis to", outdir)


if __name__ == "__main__":
    main()
