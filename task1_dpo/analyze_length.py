from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.metrics import parse_word_limit, word_count, word_limit_compliance
from common.models import load_policy, load_tokenizer
from task1_dpo.evaluate import evaluate_adapter
from task1_dpo.train import run_training


def _length_stats(values):
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return {"mean": None, "std": None, "iqr": None}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


def _canonical_stratum(value: str | None):
    if value is None:
        return None
    compact = str(value).lower().replace("_", "-").replace(" ", "-")
    if "chosen" in compact or "preferred" in compact:
        return "preferred-longer"
    if "rejected" in compact:
        return "rejected-longer"
    if "match" in compact or "equal" in compact or "similar" in compact:
        return "matched-length"
    return str(value)


def _infer_stratum(tokenizer, pair_record):
    chosen = len(tokenizer(pair_record["chosen"], add_special_tokens=False)["input_ids"])
    rejected = len(tokenizer(pair_record["rejected"], add_special_tokens=False)["input_ids"])
    tolerance = max(5, round(0.1 * max(chosen, rejected)))
    if chosen - rejected > tolerance:
        return "preferred-longer"
    if rejected - chosen > tolerance:
        return "rejected-longer"
    return "matched-length"


def _per_stratum(pair_records, tokenizer):
    grouped = defaultdict(list)
    for record in pair_records:
        stratum = _canonical_stratum(record.get("stratum"))
        if stratum is None:
            stratum = _infer_stratum(tokenizer, record)
        grouped[stratum].append(record)
    return {
        stratum: {
            "count": len(records),
            "held_out_dpo_loss": float(mean(row["dpo_loss"] for row in records)),
            "preference_accuracy": float(mean(
                float(row["adjusted_preference_margin"] > 0) for row in records
            )),
            "adjusted_preference_margin_mean": float(mean(
                row["adjusted_preference_margin"] for row in records
            )),
        }
        for stratum, records in sorted(grouped.items())
    }


def evaluate_word_limits(config_path: str, adapter: str, name: str, output_dir: Path):
    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    generation_cfg = cfg.get("generation", {})
    batch_size = int(cfg.get("eval_batch_size", cfg["batch_size"]))
    set_seed(int(cfg["seed"]))
    records = []

    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompts = [prompt_messages(row) for row in chunk]
        generated = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg.get("max_prompt_length", cfg["max_sequence_length"])),
            max_new_tokens=int(cfg["max_generation_tokens"]),
            temperature=float(generation_cfg.get("temperature", 0.7)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
            do_sample=bool(generation_cfg.get("do_sample", True)),
        )
        for local_index, (row, messages, response, token_count) in enumerate(zip(
            chunk,
            prompts,
            generated["responses"],
            generated["response_lengths"],
        )):
            prompt_text = "\n".join(str(message.get("content", "")) for message in messages)
            compliance = word_limit_compliance(prompt_text, response)
            if compliance is None:
                raise ValueError(
                    f"Could not parse a word limit from prompt {row.get('prompt_id', start + local_index)}"
                )
            records.append({
                "prompt_id": row.get("prompt_id", start + local_index),
                "prompt": prompt_text,
                "response": response,
                "response_tokens": int(token_count),
                "response_words": word_count(response),
                "word_limit": parse_word_limit(prompt_text),
                "compliant": bool(compliance),
            })

    summary = {
        "name": name,
        "adapter": str(repo_path(adapter)),
        "num_prompts": len(records),
        "word_limit_compliance": float(mean(float(row["compliant"]) for row in records)),
        "response_length_tokens": _length_stats(row["response_tokens"] for row in records),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "word_limit_generations.jsonl", records)
    save_json(output_dir / "word_limit_summary.json", summary)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument(
        "--skip-training",
        action="store_true",
        help="Use an existing length-balanced adapter instead of retraining it.",
    )
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    standard_adapter = repo_path(cfg["standard_output"])
    balanced_adapter = repo_path(cfg["length_output"])
    if not standard_adapter.exists():
        raise FileNotFoundError(
            "The standard DPO adapter is required for this comparison. Run "
            "`python -m task1_dpo.train --config configs/dpo.yaml --run-name standard` first."
        )

    train_summary = None
    if not args.skip_training:
        train_summary = run_training(
            args.config,
            run_name="length_balanced",
            dataset_path=cfg["paths"]["dpo_length_train"],
            output_path=str(balanced_adapter),
        )
    elif not balanced_adapter.exists():
        raise FileNotFoundError(
            f"--skip-training was set but adapter does not exist: {balanced_adapter}"
        )

    result_root = repo_path(cfg["results_dir"]) / "length_analysis"
    conditions = {}
    tokenizer = load_tokenizer(cfg["base_model"])
    for name, adapter in (
        ("standard", standard_adapter),
        ("length_balanced", balanced_adapter),
    ):
        print(f"Evaluating {name} on the fixed length-stratified pairs")
        evaluation = evaluate_adapter(
            args.config,
            str(adapter),
            f"{name}_length_stratified",
            dataset_path=cfg["paths"]["dpo_length_eval"],
            output_dir=result_root / name,
            generate=False,
            score_reward=False,
        )
        strata = _per_stratum(evaluation["pair_records"], tokenizer)
        word_limits = evaluate_word_limits(
            args.config,
            str(adapter),
            name,
            result_root / name,
        )
        conditions[name] = {
            "overall_preference": evaluation["summary"],
            "preference_by_length_stratum": strata,
            "word_limit_evaluation": word_limits,
        }
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(result_root / "comparison.json", {
        "length_balanced_training": train_summary,
        "conditions": conditions,
        "note": (
            "When the released dataset has no explicit stratum field, strata are inferred "
            "from response-token lengths using a max(5 tokens, 10%) matching tolerance."
        ),
    })
    print(f"Saved length-confounding comparison to {result_root / 'comparison.json'}")


if __name__ == "__main__":
    main()
