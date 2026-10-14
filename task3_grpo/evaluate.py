from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)


def _row_id(row: dict, fallback: int):
    for key in ("prompt_id", "source_index", "id"):
        if key in row:
            return row[key]
    return fallback


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


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    if not rows:
        raise ValueError("The GRPO held-out prompt set is empty")
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def _qualitative_candidates(records: list[dict]):
    if not records:
        return {}
    candidates = {
        "highest_reward": max(records, key=lambda row: row["learned_reward"]),
        "lowest_reward": min(records, key=lambda row: row["learned_reward"]),
        "largest_policy_drift": max(
            records, key=lambda row: row["sampled_kl_sequence"]
        ),
        "longest_response": max(records, key=lambda row: row["response_tokens"]),
    }
    reward_median = float(np.median([row["learned_reward"] for row in records]))
    high_reward = [row for row in records if row["learned_reward"] >= reward_median]
    if high_reward:
        candidates["high_reward_high_drift"] = max(
            high_reward, key=lambda row: row["sampled_kl_sequence"]
        )
    return candidates


def evaluate_adapter(
    config_path: str,
    adapter: str,
    name: str,
    output_dir: str | Path | None = None,
    max_prompts: int | None = None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("Task 3 evaluation requires a CUDA GPU")
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    if max_prompts is not None:
        rows = rows[: int(max_prompts)]
    if not rows:
        raise ValueError("No held-out prompts were selected for evaluation")

    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model, reward_tokenizer = bundle["reward"]
    device = next(policy.parameters()).device
    generation_cfg = cfg.get("generation", {})
    batch_size = int(cfg.get("eval_batch_size", 1))
    max_new_tokens = int(
        cfg.get("eval_max_completion_length", cfg["max_completion_length"])
    )
    set_seed(int(cfg["seed"]))
    destination = repo_path(output_dir or Path(cfg["results_dir"]) / name)
    destination.mkdir(parents=True, exist_ok=True)

    records = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [prompt_messages(row) for row in chunk]
        generated = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=max_new_tokens,
            temperature=float(generation_cfg.get("temperature", 0.7)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
            do_sample=bool(generation_cfg.get("do_sample", True)),
        )
        mask = generated["response_mask"].to(device)

        with torch.inference_mode():
            policy_logp, policy_logits = response_token_logprobs(
                policy,
                generated["sequences"],
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )
            del policy_logits
            with reference_mode(policy):
                reference_logp, reference_logits = response_token_logprobs(
                    policy,
                    generated["sequences"],
                    generated["attention_mask"],
                    generated["prompt_width"],
                    generated["response_ids"],
                )
            del reference_logits
            rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                prompts,
                generated["responses"],
                max_length=int(
                    cfg.get(
                        "reward_max_length",
                        int(cfg["max_prompt_length"]) + max_new_tokens,
                    )
                ),
            )

        sequence_kls = ((policy_logp - reference_logp) * mask).sum(-1)
        sequence_entropies = -(policy_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
        for local_index, row in enumerate(chunk):
            records.append({
                "prompt_id": _row_id(row, start + local_index),
                "source_index": start + local_index,
                "prompt": prompts[local_index],
                "response": generated["responses"][local_index],
                "learned_reward": float(rewards[local_index].item()),
                "sampled_kl_sequence": float(sequence_kls[local_index].item()),
                "sampled_entropy": float(sequence_entropies[local_index].item()),
                "response_tokens": int(generated["response_lengths"][local_index]),
                "terminated_with_eos": bool(
                    generated["terminated_with_eos"][local_index]
                ),
                "truncated": bool(generated["truncated"][local_index]),
            })
        print(
            f"[{name}] evaluated {min(start + len(chunk), len(rows))}/{len(rows)} prompts",
            flush=True,
        )
        completed = min(start + len(chunk), len(rows))
        if completed % 10 == 0 or completed == len(rows):
            write_jsonl(destination / "generations.jsonl", records)
            save_json(
                destination / "evaluation_progress.json",
                {
                    "name": name,
                    "evaluated_prompts": completed,
                    "target_prompts": len(rows),
                    "complete": completed == len(rows),
                },
            )
        # Release the previous prompt's large full-vocabulary tensors before
        # allocating the next generation.
        del (
            generated,
            mask,
            policy_logp,
            reference_logp,
            rewards,
            sequence_kls,
            sequence_entropies,
        )

    total_kl = sum(row["sampled_kl_sequence"] for row in records)
    total_tokens = sum(row["response_tokens"] for row in records)
    summary = {
        "name": name,
        "adapter": str(repo_path(adapter)),
        "dataset": str(cfg["paths"]["rl_prompt_eval"]),
        "seed": int(cfg["seed"]),
        "num_prompts": len(records),
        "source_prompts": len(bundle["rows"]),
        "decoding": {
            "max_prompt_length": int(cfg["max_prompt_length"]),
            "max_completion_length": max_new_tokens,
            "temperature": float(generation_cfg.get("temperature", 0.7)),
            "top_p": float(generation_cfg.get("top_p", 0.9)),
            "do_sample": bool(generation_cfg.get("do_sample", True)),
        },
        "learned_reward_mean": float(mean(row["learned_reward"] for row in records)),
        "learned_reward_std": float(
            np.std([row["learned_reward"] for row in records])
        ),
        "sampled_kl_per_response_mean": float(
            mean(row["sampled_kl_sequence"] for row in records)
        ),
        "sampled_kl_per_token": float(total_kl / max(total_tokens, 1)),
        "sampled_entropy": float(mean(row["sampled_entropy"] for row in records)),
        "response_length_tokens": _length_stats(
            [row["response_tokens"] for row in records]
        ),
        "termination_rate": float(
            mean(float(row["terminated_with_eos"]) for row in records)
        ),
        "truncation_rate": float(mean(float(row["truncated"]) for row in records)),
    }

    write_jsonl(destination / "generations.jsonl", records)
    save_json(destination / "qualitative_candidates.json", _qualitative_candidates(records))
    save_json(destination / "summary.json", summary)

    bundle.clear()
    del policy, reward_model, reward_tokenizer
    clear_gpu()
    return {"summary": summary, "records": records, "output_dir": str(destination)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--output-dir")
    ap.add_argument("--max-prompts", type=int)
    args = ap.parse_args()
    result = evaluate_adapter(
        args.config,
        args.adapter,
        args.name,
        output_dir=args.output_dir,
        max_prompts=args.max_prompts,
    )
    print(result["summary"])


if __name__ == "__main__":
    main()
