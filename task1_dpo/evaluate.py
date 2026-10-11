from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data import (
    load_yaml,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json, set_seed
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import filter_overlong_prompts, make_collate


def load_evaluation_bundle(config_path: str, adapter: str, dataset_path: str | None = None):
    cfg = load_yaml(config_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    source_rows = read_jsonl(dataset_path or cfg["paths"]["dpo_standard_eval"])
    rows, skipped_rows = filter_overlong_prompts(
        tokenizer,
        source_rows,
        int(cfg["max_sequence_length"]),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "source_row_count": len(source_rows),
        "skipped_rows": skipped_rows,
        "tokenizer": tokenizer,
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
    }


def _to_device(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def _row_id(row: dict, fallback: int):
    for key in ("prompt_id", "pair_id", "source_index", "id"):
        if key in row:
            return row[key]
    return fallback


def _row_stratum(row: dict):
    for key in (
        "length_stratum",
        "stratum",
        "length_group",
        "length_bucket",
        "length_category",
        "category",
    ):
        if key in row:
            return str(row[key])
    return None


def _length_stats(values: list[int]):
    if not values:
        return {"mean": None, "std": None, "iqr": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "min": int(array.min()),
        "max": int(array.max()),
    }


def _pairwise_metrics(model, tokenizer, rows, cfg, beta: float):
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    device = next(model.parameters()).device
    records = []
    offset = 0
    with torch.inference_mode():
        for chosen, rejected in loader:
            chosen = _to_device(chosen, device)
            rejected = _to_device(rejected, device)
            policy_chosen = response_sequence_logprobs(model, chosen)[0]
            policy_rejected = response_sequence_logprobs(model, rejected)[0]
            with reference_mode(model):
                ref_chosen = response_sequence_logprobs(model, chosen)[0]
                ref_rejected = response_sequence_logprobs(model, rejected)[0]

            margins = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
            losses = -torch.nn.functional.logsigmoid(beta * margins)
            for local_index in range(len(margins)):
                row = rows[offset + local_index]
                chosen_text, rejected_text = preference_responses(row)
                records.append({
                    "pair_id": _row_id(row, offset + local_index),
                    "stratum": _row_stratum(row),
                    "prompt": prompt_messages_from_preference(row),
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "policy_chosen_logp": float(policy_chosen[local_index].item()),
                    "policy_rejected_logp": float(policy_rejected[local_index].item()),
                    "reference_chosen_logp": float(ref_chosen[local_index].item()),
                    "reference_rejected_logp": float(ref_rejected[local_index].item()),
                    "adjusted_preference_margin": float(margins[local_index].item()),
                    "dpo_loss": float(losses[local_index].item()),
                    "preference_correct": bool(margins[local_index].item() > 0),
                })
            offset += len(margins)

    if offset != len(rows):
        raise RuntimeError(f"Evaluated {offset} preference pairs but expected {len(rows)}")
    return records


def _score_pair_rewards(reward_bundle, rows, records, batch_size: int, max_length: int):
    reward_model, reward_tokenizer = reward_bundle
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompts = [prompt_messages_from_preference(row) for row in chunk]
        chosen = [preference_responses(row)[0] for row in chunk]
        rejected = [preference_responses(row)[1] for row in chunk]
        chosen_scores = score_reward_pairs(
            reward_model, reward_tokenizer, prompts, chosen, max_length=max_length
        )
        rejected_scores = score_reward_pairs(
            reward_model, reward_tokenizer, prompts, rejected, max_length=max_length
        )
        for index, (chosen_score, rejected_score) in enumerate(
            zip(chosen_scores.tolist(), rejected_scores.tolist()), start=start
        ):
            records[index]["chosen_reward_score"] = float(chosen_score)
            records[index]["rejected_reward_score"] = float(rejected_score)
            records[index]["reward_margin"] = float(chosen_score - rejected_score)


def _generation_metrics(model, tokenizer, reward_bundle, rows, cfg):
    generation_cfg = cfg.get("generation", {})
    batch_size = int(cfg.get("eval_batch_size", cfg["batch_size"]))
    max_prompt_length = int(cfg.get("max_prompt_length", cfg["max_sequence_length"]))
    max_new_tokens = int(cfg["max_generation_tokens"])
    records = []
    device = next(model.parameters()).device

    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompts = [prompt_messages_from_preference(row) for row in chunk]
        generated = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=float(generation_cfg.get("temperature", 0.7)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
            do_sample=bool(generation_cfg.get("do_sample", True)),
        )

        with torch.inference_mode():
            policy_logp, _ = response_token_logprobs(
                model,
                generated["sequences"],
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )
            with reference_mode(model):
                reference_logp, _ = response_token_logprobs(
                    model,
                    generated["sequences"],
                    generated["attention_mask"],
                    generated["prompt_width"],
                    generated["response_ids"],
                )
        mask = generated["response_mask"].to(device)
        sequence_kls = ((policy_logp - reference_logp) * mask).sum(-1)
        entropies = -(policy_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)

        reward_scores = [None] * len(chunk)
        if reward_bundle is not None:
            reward_scores = score_reward_pairs(
                reward_bundle[0],
                reward_bundle[1],
                prompts,
                generated["responses"],
                max_length=int(cfg.get("reward_max_length", 1024)),
            ).tolist()

        for local_index, row in enumerate(chunk):
            records.append({
                "prompt_id": _row_id(row, start + local_index),
                "prompt": prompts[local_index],
                "response": generated["responses"][local_index],
                "response_tokens": int(generated["response_lengths"][local_index]),
                "terminated_with_eos": bool(generated["terminated_with_eos"][local_index]),
                "truncated": bool(generated["truncated"][local_index]),
                "sampled_kl_sequence": float(sequence_kls[local_index].item()),
                "sampled_entropy": float(entropies[local_index].item()),
                "reward_score": (
                    None if reward_scores[local_index] is None
                    else float(reward_scores[local_index])
                ),
            })

    if not records:
        return records, {}
    total_kl = sum(record["sampled_kl_sequence"] for record in records)
    total_tokens = sum(record["response_tokens"] for record in records)
    scores = [record["reward_score"] for record in records if record["reward_score"] is not None]
    return records, {
        "sampled_kl_per_response_mean": float(mean(record["sampled_kl_sequence"] for record in records)),
        "sampled_kl_per_token": float(total_kl / max(total_tokens, 1)),
        "sampled_entropy": float(mean(record["sampled_entropy"] for record in records)),
        "reward_model_score_mean": float(mean(scores)) if scores else None,
        "response_length_tokens": _length_stats([record["response_tokens"] for record in records]),
        "truncation_rate": float(mean(float(record["truncated"]) for record in records)),
    }


def _qualitative_candidates(pair_records, generation_records):
    candidates = {}
    if pair_records:
        candidates["largest_adjusted_margin"] = max(
            pair_records, key=lambda row: row["adjusted_preference_margin"]
        )
        candidates["smallest_adjusted_margin"] = min(
            pair_records, key=lambda row: row["adjusted_preference_margin"]
        )
        disagreements = [
            row for row in pair_records
            if row.get("reward_margin") is not None
            and row["adjusted_preference_margin"] * row["reward_margin"] < 0
        ]
        if disagreements:
            candidates["policy_reward_disagreement"] = max(
                disagreements,
                key=lambda row: abs(row["adjusted_preference_margin"]),
            )
    if generation_records:
        candidates["longest_generation"] = max(
            generation_records, key=lambda row: row["response_tokens"]
        )
        rewarded = [row for row in generation_records if row["reward_score"] is not None]
        if rewarded:
            candidates["highest_reward_generation"] = max(
                rewarded, key=lambda row: row["reward_score"]
            )
    return candidates


def evaluate_adapter(
    config_path: str,
    adapter: str,
    name: str,
    dataset_path: str | None = None,
    output_dir: str | Path | None = None,
    generate: bool = True,
    score_reward: bool = True,
    beta: float | None = None,
):
    bundle = load_evaluation_bundle(config_path, adapter, dataset_path)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    if not rows:
        raise ValueError(
            f"DPO evaluation dataset is empty: "
            f"{dataset_path or cfg['paths']['dpo_standard_eval']}"
        )
    tokenizer = bundle["tokenizer"]
    model = bundle["policy"]
    set_seed(int(cfg["seed"]))
    beta_value = float(cfg["beta"] if beta is None else beta)

    pair_records = _pairwise_metrics(model, tokenizer, rows, cfg, beta_value)
    reward_bundle = load_reward_model(cfg) if score_reward else None
    if reward_bundle is not None:
        _score_pair_rewards(
            reward_bundle,
            rows,
            pair_records,
            int(cfg.get("eval_batch_size", cfg["batch_size"])),
            int(cfg.get("reward_max_length", 1024)),
        )

    generation_records, generation_summary = ([], {})
    if generate:
        set_seed(int(cfg["seed"]))
        generation_records, generation_summary = _generation_metrics(
            model, tokenizer, reward_bundle, rows, cfg
        )

    margins = [record["adjusted_preference_margin"] for record in pair_records]
    losses = [record["dpo_loss"] for record in pair_records]
    summary = {
        "name": name,
        "adapter": str(repo_path(adapter)),
        "dataset": str(dataset_path or cfg["paths"]["dpo_standard_eval"]),
        "seed": int(cfg["seed"]),
        "beta_used_for_eval_loss": beta_value,
        "num_preference_pairs": len(pair_records),
        "source_preference_pairs": bundle["source_row_count"],
        "skipped_overlong_prompts": len(bundle["skipped_rows"]),
        "held_out_dpo_loss": float(mean(losses)) if losses else None,
        "held_out_preference_accuracy": (
            float(mean(float(value > 0) for value in margins)) if margins else None
        ),
        "adjusted_preference_margin_mean": float(mean(margins)) if margins else None,
        **generation_summary,
    }

    destination = repo_path(output_dir or Path(cfg["results_dir"]) / name)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "preference_pairs.jsonl", pair_records)
    write_jsonl(destination / "skipped_overlong_prompts.jsonl", bundle["skipped_rows"])
    if generation_records:
        write_jsonl(destination / "generations.jsonl", generation_records)
    save_json(
        destination / "qualitative_candidates.json",
        _qualitative_candidates(pair_records, generation_records),
    )
    save_json(destination / "summary.json", summary)

    # Drop the heavyweight objects before emptying the CUDA cache.  The return
    # value contains only Python records and summaries.
    bundle.clear()
    del model
    del reward_bundle
    clear_gpu()
    return {
        "summary": summary,
        "pair_records": pair_records,
        "generation_records": generation_records,
        "output_dir": str(destination),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output-dir")
    ap.add_argument("--beta", type=float)
    args = ap.parse_args()
    result = evaluate_adapter(
        args.config,
        args.adapter,
        args.name,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        beta=args.beta,
    )
    print(result["summary"])


if __name__ == "__main__":
    main()
