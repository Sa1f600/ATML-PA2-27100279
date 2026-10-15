from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import set_seed

from common.data import (
    load_yaml,
    prompt_messages,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import batch_generate
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


POLICY_ORDER = ("sft", "rlvr", "rlaif")
PAIRWISE_POLICIES = ("rlvr", "rlaif")


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    from common.models import load_tokenizer

    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    from common.models import load_policy

    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def _row_id(row: dict, index: int) -> str:
    for key in ("prompt_id", "source_index", "gsm8k_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    return f"row:{index}"


def _question(row: dict) -> str:
    if row.get("question") is not None:
        return str(row["question"])
    if isinstance(row.get("prompt"), str):
        return str(row["prompt"])
    for message in reversed(prompt_messages(row)):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError("Math evaluation row has no user question")


def _gold_final(row: dict) -> str:
    for key in ("gold_final", "answer", "final_answer"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError("Math evaluation row has no gold final answer")


def prepare_math_rows(rows: list[dict], dataset: str) -> list[dict]:
    """Normalize course GSM8K/SVAMP schemas without altering prompt contents."""
    prepared = []
    seen = set()
    for index, row in enumerate(rows):
        eval_id = _row_id(row, index)
        if eval_id in seen:
            raise ValueError(f"Duplicate evaluation ID: {eval_id}")
        seen.add(eval_id)
        prepared.append(
            {
                "eval_id": eval_id,
                "dataset": dataset,
                "source_index": str(row.get("source_index", eval_id)),
                "prompt_id": str(row.get("prompt_id", eval_id)),
                "question": _question(row),
                "messages": prompt_messages(row),
                "gold_final": _gold_final(row),
            }
        )
    if not prepared:
        raise ValueError("Math evaluation dataset is empty")
    return prepared


def _length_summary(values: list[int | float]) -> dict:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"mean": None, "std": None, "iqr": None, "min": None, "max": None}
    q25, q75 = np.percentile(array, [25, 75])
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "iqr": float(q75 - q25),
        "min": int(array.min()),
        "max": int(array.max()),
    }


def aggregate_policy(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate an empty policy result")
    return {
        "num_examples": len(rows),
        "exact_answer_accuracy": float(np.mean([bool(row["exact_correct"]) for row in rows])),
        "format_compliance_rate": float(
            np.mean([bool(row["format_compliant"]) for row in rows])
        ),
        "response_length_tokens": _length_summary(
            [int(row["response_tokens"]) for row in rows]
        ),
        "termination_rate": float(
            np.mean([bool(row.get("terminated_with_eos", False)) for row in rows])
        ),
        "truncation_rate": float(
            np.mean([bool(row.get("truncated", False)) for row in rows])
        ),
    }


def verifier_preference(candidate_reward: float, reference_reward: float) -> str:
    if candidate_reward > reference_reward:
        return "A"
    if candidate_reward < reference_reward:
        return "B"
    return "TIE"


def pairwise_win_value(preference: str) -> float:
    if preference == "A":
        return 1.0
    if preference == "B":
        return 0.0
    if preference == "TIE":
        return 0.5
    raise ValueError(f"Invalid pairwise preference: {preference}")


def aggregate_pairwise(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate an empty pairwise comparison")
    counts = Counter(str(row["ai_preference"]) for row in rows)
    verifier_counts = Counter(str(row["verifier_preference"]) for row in rows)
    agreements = [row["ai_preference"] == row["verifier_preference"] for row in rows]
    decisive = [
        row
        for row in rows
        if row["ai_preference"] != "TIE" and row["verifier_preference"] != "TIE"
    ]
    return {
        "num_comparisons": len(rows),
        "ai_win_rate_vs_sft": float(
            np.mean([pairwise_win_value(str(row["ai_preference"])) for row in rows])
        ),
        "ai_wins": int(counts["A"]),
        "ai_ties": int(counts["TIE"]),
        "ai_losses": int(counts["B"]),
        "ai_tie_rate": float(counts["TIE"] / len(rows)),
        "verifier_candidate_better": int(verifier_counts["A"]),
        "verifier_ties": int(verifier_counts["TIE"]),
        "verifier_candidate_worse": int(verifier_counts["B"]),
        "verifier_judge_exact_agreement_rate": float(np.mean(agreements)),
        "verifier_judge_decisive_agreement_rate": (
            float(np.mean([row["ai_preference"] == row["verifier_preference"] for row in decisive]))
            if decisive
            else None
        ),
        "num_jointly_decisive": len(decisive),
    }


def _existing_generations(path: Path, policy: str, overwrite: bool) -> dict[str, dict]:
    if overwrite or not path.exists():
        return {}
    completed = {}
    for row in read_jsonl(path):
        if str(row.get("policy")) != policy:
            raise ValueError(f"{path} contains a different policy")
        key = str(row["eval_id"])
        if key in completed:
            raise ValueError(f"Duplicate eval_id={key} in {path}")
        completed[key] = row
    return completed


def generate_for_policy(
    cfg: dict,
    dataset: str,
    prepared_rows: list[dict],
    policy: str,
    output_path: Path,
    batch_size: int,
    overwrite: bool = False,
) -> list[dict]:
    if policy not in POLICY_ORDER:
        raise KeyError(policy)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    completed = _existing_generations(output_path, policy, overwrite)
    expected_by_id = {row["eval_id"]: row for row in prepared_rows}
    unexpected = set(completed).difference(expected_by_id)
    if unexpected:
        raise ValueError(f"Existing output has unknown evaluation IDs: {sorted(unexpected)[:5]}")
    for eval_id, saved in completed.items():
        expected = expected_by_id[eval_id]
        for field in ("dataset", "question", "gold_final"):
            if str(saved.get(field)) != str(expected[field]):
                raise ValueError(
                    f"Existing output does not match the current {field} for "
                    f"eval_id={eval_id}; rerun with --overwrite-generation"
                )
    if len(completed) == len(prepared_rows):
        print(f"[{dataset}:{policy}] already generated {len(completed)}/{len(prepared_rows)}")
        return [completed[row["eval_id"]] for row in prepared_rows]

    from common.models import load_tokenizer

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_frozen_policy(cfg, policy)
    set_seed(int(cfg["seed"]))
    remaining = [row for row in prepared_rows if row["eval_id"] not in completed]
    try:
        for start in range(0, len(remaining), batch_size):
            chunk = remaining[start : start + batch_size]
            generation = batch_generate(
                model,
                tokenizer,
                [row["messages"] for row in chunk],
                max_prompt_length=int(cfg.get("math_max_prompt_length", 512)),
                max_new_tokens=int(cfg["math_max_new_tokens"]),
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            for row, response, tokens, terminated, truncated in zip(
                chunk,
                generation["responses"],
                generation["response_lengths"],
                generation["terminated_with_eos"],
                generation["truncated"],
            ):
                designated = extract_designated_final(response)
                completed[row["eval_id"]] = {
                    key: row[key]
                    for key in (
                        "eval_id",
                        "dataset",
                        "source_index",
                        "prompt_id",
                        "question",
                        "gold_final",
                    )
                } | {
                    "policy": policy,
                    "response": response,
                    "response_tokens": int(tokens),
                    "terminated_with_eos": bool(terminated),
                    "truncated": bool(truncated),
                    "designated_final": designated,
                    "format_compliant": designated is not None,
                    "exact_correct": bool(exact_reward(response, row["gold_final"])),
                }
            ordered = [completed[row["eval_id"]] for row in prepared_rows if row["eval_id"] in completed]
            write_jsonl(output_path, ordered)
            print(f"[{dataset}:{policy}] generated {len(completed)}/{len(prepared_rows)}")
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return [completed[row["eval_id"]] for row in prepared_rows]


def _existing_comparisons(path: Path, candidate_policy: str, overwrite: bool) -> dict[str, dict]:
    if overwrite or not path.exists():
        return {}
    completed = {}
    for row in read_jsonl(path):
        if str(row.get("candidate_policy")) != candidate_policy:
            raise ValueError(f"{path} contains a different candidate policy")
        key = str(row["eval_id"])
        if key in completed:
            raise ValueError(f"Duplicate eval_id={key} in {path}")
        completed[key] = row
    return completed


def judge_against_sft(
    judge: PairwiseAIJudge,
    dataset: str,
    candidate_policy: str,
    candidate_rows: list[dict],
    sft_rows: list[dict],
    output_path: Path,
    checkpoint_every: int,
    overwrite: bool = False,
) -> list[dict]:
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    sft_by_id = {str(row["eval_id"]): row for row in sft_rows}
    candidate_ids = [str(row["eval_id"]) for row in candidate_rows]
    if set(candidate_ids) != set(sft_by_id):
        raise ValueError("Candidate and SFT generations do not cover identical evaluation IDs")
    completed = _existing_comparisons(output_path, candidate_policy, overwrite)
    if set(completed).difference(candidate_ids):
        raise ValueError("Existing comparison file contains unknown evaluation IDs")
    candidate_by_id = {str(row["eval_id"]): row for row in candidate_rows}
    for eval_id, saved in completed.items():
        candidate = candidate_by_id[eval_id]
        reference = sft_by_id[eval_id]
        expected = {
            "dataset": dataset,
            "question": candidate["question"],
            "candidate_response": candidate["response"],
            "reference_response": reference["response"],
        }
        for field, value in expected.items():
            if str(saved.get(field)) != str(value):
                raise ValueError(
                    f"Existing comparison does not match the current {field} for "
                    f"eval_id={eval_id}; rerun with --overwrite-judging"
                )

    for candidate in candidate_rows:
        eval_id = str(candidate["eval_id"])
        if eval_id in completed:
            continue
        reference = sft_by_id[eval_id]
        if candidate["question"] != reference["question"]:
            raise ValueError(f"Question mismatch for eval_id={eval_id}")
        ai_preference = judge.compare(
            str(candidate["question"]),
            str(candidate["response"]),
            str(reference["response"]),
        )
        exact_preference = verifier_preference(
            float(candidate["exact_correct"]), float(reference["exact_correct"])
        )
        completed[eval_id] = {
            "eval_id": eval_id,
            "dataset": dataset,
            "question": candidate["question"],
            "candidate_policy": candidate_policy,
            "reference_policy": "sft",
            "candidate_response": candidate["response"],
            "reference_response": reference["response"],
            "candidate_exact_correct": bool(candidate["exact_correct"]),
            "reference_exact_correct": bool(reference["exact_correct"]),
            "verifier_preference": exact_preference,
            "ai_preference": ai_preference,
            "ai_win_value": pairwise_win_value(ai_preference),
            "verifier_judge_agree": ai_preference == exact_preference,
        }
        if len(completed) % checkpoint_every == 0 or len(completed) == len(candidate_rows):
            write_jsonl(
                output_path,
                [completed[x] for x in candidate_ids if x in completed],
            )
        print(f"[{dataset}:{candidate_policy}] judged {len(completed)}/{len(candidate_rows)} vs SFT")
    ordered = [completed[x] for x in candidate_ids]
    write_jsonl(output_path, ordered)
    return ordered


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _summary_rows(summary: dict) -> list[dict]:
    rows = []
    for policy in POLICY_ORDER:
        metrics = summary["policies"][policy]
        pair = metrics.get("pairwise_vs_sft") or {}
        rows.append(
            {
                "dataset": summary["dataset"],
                "policy": policy,
                "num_examples": metrics["num_examples"],
                "exact_answer_accuracy": metrics["exact_answer_accuracy"],
                "format_compliance_rate": metrics["format_compliance_rate"],
                "mean_response_tokens": metrics["response_length_tokens"]["mean"],
                "ai_pairwise_win_rate_vs_sft": 0.5 if policy == "sft" else pair["ai_win_rate_vs_sft"],
                "ai_tie_rate_vs_sft": None if policy == "sft" else pair["ai_tie_rate"],
                "verifier_judge_agreement_rate": (
                    None if policy == "sft" else pair["verifier_judge_exact_agreement_rate"]
                ),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--overwrite-generation", action="store_true")
    ap.add_argument("--overwrite-judging", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 5 generation and pairwise judging require a CUDA GPU runtime")
    raw_rows = read_jsonl(dataset_path(cfg, args.dataset))
    rows = prepare_math_rows(raw_rows, args.dataset)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size or int(cfg.get("math_batch_size", 4))
    started = time.perf_counter()

    generated = {}
    for policy in POLICY_ORDER:
        generated[policy] = generate_for_policy(
            cfg,
            args.dataset,
            rows,
            policy,
            outdir / f"generated_{policy}.jsonl",
            batch_size=batch_size,
            overwrite=args.overwrite_generation,
        )
    write_jsonl(outdir / "generated_all.jsonl", [row for policy in POLICY_ORDER for row in generated[policy]])

    set_seed(int(cfg["seed"]))
    judge = PairwiseAIJudge(cfg, cfg.get("task5_judge_cache", "results/task5_feedback/pairwise_judge_cache.json"))
    comparisons = {}
    try:
        for policy in PAIRWISE_POLICIES:
            comparisons[policy] = judge_against_sft(
                judge,
                args.dataset,
                policy,
                generated[policy],
                generated["sft"],
                outdir / f"pairwise_{policy}_vs_sft.jsonl",
                checkpoint_every=int(cfg.get("math_judge_checkpoint_every", 5)),
                overwrite=args.overwrite_judging,
            )
    finally:
        del judge
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "dataset": args.dataset,
        "dataset_path": dataset_path(cfg, args.dataset),
        "num_examples": len(rows),
        "policy_order": list(POLICY_ORDER),
        "policies": {},
        "decoding": {
            "max_prompt_length": int(cfg.get("math_max_prompt_length", 512)),
            "max_new_tokens": int(cfg["math_max_new_tokens"]),
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        },
        "seed": int(cfg["seed"]),
    }
    for policy in POLICY_ORDER:
        metrics = aggregate_policy(generated[policy])
        if policy in comparisons:
            metrics["pairwise_vs_sft"] = aggregate_pairwise(comparisons[policy])
        else:
            metrics["pairwise_vs_sft"] = {
                "reference_baseline": True,
                "ai_win_rate_vs_sft": 0.5,
            }
        summary["policies"][policy] = metrics
    summary["elapsed_seconds"] = float(time.perf_counter() - started)
    _write_json(outdir / "summary.json", summary)
    pd.DataFrame(_summary_rows(summary)).to_csv(outdir / "summary.csv", index=False)
    _write_json(
        outdir / "manifest.json",
        {
            "dataset": args.dataset,
            "eval_ids": [row["eval_id"] for row in rows],
            "generated_rows": {policy: len(generated[policy]) for policy in POLICY_ORDER},
            "pairwise_rows": {policy: len(comparisons[policy]) for policy in PAIRWISE_POLICIES},
            "judge_model": cfg["ai_judge_model"],
            "judge_cache": cfg.get("task5_judge_cache"),
        },
    )
    print(pd.DataFrame(_summary_rows(summary)).to_string(index=False))
    print("Saved Task 5", args.dataset, "evaluation to", outdir)


if __name__ == "__main__":
    main()
