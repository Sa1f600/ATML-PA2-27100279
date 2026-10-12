from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from common.data import load_yaml, prompt_messages, read_jsonl, render_prompt, repo_path
from common.generation import response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json
from common.metrics import masked_mean
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    token_values,
)
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_adapter
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields.
    normalized = []
    for source in rows:
        row = dict(source)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def _as_1d(value, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().reshape(-1)
    if tensor.ndim != 1:
        raise ValueError(f"Cached {name} must be one-dimensional after squeezing")
    return tensor


def _source_row(prompt_rows: list[dict], source_index):
    try:
        index = int(source_index)
    except (TypeError, ValueError):
        index = -1
    if 0 <= index < len(prompt_rows):
        return index, prompt_rows[index]
    for index, row in enumerate(prompt_rows):
        if any(str(row.get(key)) == str(source_index) for key in ("prompt_id", "source_index", "id")):
            return index, row
    raise KeyError(f"Could not resolve cached source_index={source_index!r}")


def _cached_response_ids(row: dict, tokenizer, expected_steps: int):
    ids = None
    for key in ("response_ids", "response_token_ids", "completion_ids"):
        if key in row:
            ids = torch.as_tensor(row[key], dtype=torch.long).reshape(-1).tolist()
            break
    if ids is None:
        ids = tokenizer(str(row["response"]), add_special_tokens=False)["input_ids"]

    if len(ids) < expected_steps and tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]
    if len(ids) < expected_steps:
        raise ValueError(
            f"Could only reconstruct {len(ids)} response tokens for a cache row "
            f"containing {expected_steps} log-probabilities"
        )
    return ids[:expected_steps]


def _cached_task_reward(row: dict):
    for key in ("task_reward", "learned_reward", "reward_score", "reward"):
        if key in row:
            value = torch.as_tensor(row[key]).squeeze()
            if value.numel() == 1:
                return float(value.item())
    return None


def run_cached_clipping_study(config_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("The Task 2 cached-rollout study requires a CUDA GPU")
    cfg = load_yaml(config_path)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    prompt_rows = read_jsonl(cfg["paths"]["rl_prompt_train"])
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=False,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode="frozen",
    )
    device = next(policy.parameters()).device

    new_logps = []
    old_logps = []
    ref_logps = []
    masks = []
    values = []
    task_rewards = []
    terminated_with_eos = []
    prompt_batches = []
    response_texts = []
    missing_reward_indices = []

    for cache_index, row in enumerate(rows):
        old_logp = _as_1d(row["old_logprobs"], "old_logprobs")
        ref_logp = _as_1d(row["ref_logprobs"], "ref_logprobs")
        if len(old_logp) != len(ref_logp):
            raise ValueError(f"Cached policy/reference lengths differ in row {cache_index}")
        steps = len(old_logp)

        _, source = _source_row(prompt_rows, row["source_index"])
        prompt = prompt_messages(source)
        rendered = render_prompt(tokenizer, prompt)
        prompt_ids = tokenizer(
            rendered,
            truncation=True,
            max_length=int(cfg["max_prompt_length"]),
        )["input_ids"]
        response_ids = _cached_response_ids(row, tokenizer, steps)
        sequence = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
        attention = torch.ones_like(sequence)
        response_tensor = torch.tensor([response_ids], dtype=torch.long, device=device)

        with torch.inference_mode():
            new_logp, logits = response_token_logprobs(
                policy,
                sequence,
                attention,
                len(prompt_ids),
                response_tensor,
            )
            del logits
            all_values = token_values(value_model, sequence, attention)
            row_values = all_values[:, len(prompt_ids) - 1 : len(prompt_ids) - 1 + steps]

        if "response_mask" in row:
            mask = _as_1d(row["response_mask"], "response_mask")[:steps]
        else:
            mask = torch.ones(steps, dtype=torch.float32)
        if len(mask) != steps:
            raise ValueError(f"Cached response mask has the wrong length in row {cache_index}")

        new_logps.append(new_logp[0].float().cpu())
        old_logps.append(old_logp)
        ref_logps.append(ref_logp)
        masks.append(mask)
        values.append(row_values[0].float().cpu())
        cached_reward = _cached_task_reward(row)
        task_rewards.append(cached_reward)
        valid_steps = int(mask.sum().item())
        terminated_with_eos.append(
            bool(
                tokenizer.eos_token_id is not None
                and tokenizer.eos_token_id in response_ids[:valid_steps]
            )
        )
        if cached_reward is None:
            missing_reward_indices.append(cache_index)
        prompt_batches.append(prompt)
        response_texts.append(str(row["response"]))

    if missing_reward_indices:
        reward_model, reward_tokenizer = load_reward_model(cfg)
        for start in range(0, len(missing_reward_indices), int(cfg.get("eval_batch_size", 1))):
            indices = missing_reward_indices[start : start + int(cfg.get("eval_batch_size", 1))]
            scores = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                [prompt_batches[index] for index in indices],
                [response_texts[index] for index in indices],
                max_length=int(cfg["reward_max_length"]),
            ).tolist()
            for index, score in zip(indices, scores):
                task_rewards[index] = float(score)
        del reward_model, reward_tokenizer

    padded_new = pad_sequence(new_logps, batch_first=True)
    padded_old = pad_sequence(old_logps, batch_first=True)
    padded_ref = pad_sequence(ref_logps, batch_first=True)
    padded_mask = pad_sequence(masks, batch_first=True)
    padded_values = pad_sequence(values, batch_first=True)
    raw_reward_tensor = torch.tensor(task_rewards, dtype=torch.float32)
    reward_tensor = raw_reward_tensor.clone()
    missing_eos_penalty = float(cfg.get("missing_eos_penalty", 0.0))
    for index, terminated in enumerate(terminated_with_eos):
        if not terminated:
            reward_tensor[index] -= missing_eos_penalty

    shaped = shaped_rewards(
        reward_tensor,
        padded_old,
        padded_ref,
        padded_mask,
        float(cfg["kl_beta"]),
    )
    advantages, returns = compute_gae(
        shaped,
        padded_values,
        padded_mask,
        gamma=float(cfg["gamma"]),
        lam=float(cfg["gae_lambda"]),
    )
    advantages = normalize_advantages(advantages, padded_mask)

    conditions = []
    for epsilon in [float(value) for value in cfg["clip_values"]]:
        loss, ratio, affected = ppo_policy_loss(
            padded_new,
            padded_old,
            advantages,
            padded_mask,
            eps=epsilon,
        )
        outside = ((ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)).float()
        conditions.append({
            "clip_epsilon": epsilon,
            "clipped_surrogate": float(-loss.item()),
            "affected_token_fraction": float(affected.item()),
            "ratio_outside_clip_fraction": float(masked_mean(outside, padded_mask).item()),
            "ratio_mean": float(masked_mean(ratio, padded_mask).item()),
            "ratio_std": float(ratio[padded_mask.bool()].std(unbiased=False).item()),
        })

    result = {
        "cache": str(cfg["cached_rollouts"]),
        "num_rollouts": len(rows),
        "num_response_tokens": int(padded_mask.sum().item()),
        "kl_beta_used_for_advantages": float(cfg["kl_beta"]),
        "gamma": float(cfg["gamma"]),
        "gae_lambda": float(cfg["gae_lambda"]),
        "learned_reward_mean": float(raw_reward_tensor.mean().item()),
        "penalized_task_reward_mean": float(reward_tensor.mean().item()),
        "termination_rate": float(np.mean(terminated_with_eos)),
        "return_mean": float(masked_mean(returns, padded_mask).item()),
        "conditions": conditions,
    }
    destination = repo_path(cfg["results_dir"]) / "clipping"
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / "cached_rollout_analysis.json", result)

    del sequence, attention, response_tensor, new_logp, all_values, row_values
    del policy, value_model
    clear_gpu()
    return result


def _epsilon_label(value: float):
    return f"epsilon_{value:.2f}".replace(".", "p")


def _stability_statistics(output: Path):
    rows = read_jsonl(output / "train_metrics.jsonl")
    if not rows:
        return {}
    return {
        "definition": "population standard deviation across matched continuation updates",
        "policy_loss_std": float(np.std([row["policy_loss"] for row in rows])),
        "value_loss_std": float(np.std([row["value_loss"] for row in rows])),
        "learned_reward_std": float(np.std([row["learned_reward"] for row in rows])),
        "kl_per_token_std": float(np.std([row["sampled_kl_per_token"] for row in rows])),
        "policy_gradient_norm_max": float(max(row["policy_gradient_norm"] for row in rows)),
        "mean_clip_fraction": float(np.mean([row["clip_fraction"] for row in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--skip-evaluation", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    result_root = repo_path(cfg["results_dir"]) / "clipping"
    result_root.mkdir(parents=True, exist_ok=True)

    cached_result = None
    cached_path = result_root / "cached_rollout_analysis.json"
    if args.skip_cache:
        if not cached_path.exists():
            raise FileNotFoundError(f"No existing cached analysis at {cached_path}")
        cached_result = load_json(cached_path)
    else:
        print("Analyzing the fixed cached rollout batch")
        cached_result = run_cached_clipping_study(args.config)
    if args.cache_only:
        print(f"Saved cached clipping analysis to {cached_path}")
        return

    conditions = []
    for epsilon in [float(value) for value in cfg["clip_values"]]:
        label = _epsilon_label(epsilon)
        adapter = repo_path("outputs/task2_ppo") / label
        print(f"Running matched PPO clipping condition {label}")
        if args.skip_training:
            if not adapter.exists():
                raise FileNotFoundError(
                    f"--skip-training was set but adapter does not exist: {adapter}"
                )
            summary_path = adapter / "training_summary.json"
            training = load_json(summary_path) if summary_path.exists() else None
        else:
            training = run_ppo(
                args.config,
                output=str(adapter),
                updates=int(cfg["fork_updates"]),
                clip_epsilon=epsilon,
                kl_beta=float(cfg["kl_beta"]),
                run_name=label,
            )

        evaluation_path = result_root / label / "summary.json"
        if args.skip_evaluation:
            if not evaluation_path.exists():
                raise FileNotFoundError(
                    f"--skip-evaluation was set but result does not exist: {evaluation_path}"
                )
            evaluation = load_json(evaluation_path)
        else:
            evaluation = evaluate_adapter(
                args.config,
                str(adapter),
                label,
                output_dir=result_root / label,
            )["summary"]

        conditions.append({
            "condition": label,
            "clip_epsilon": epsilon,
            "budget": "matched_short_continuation",
            "training": training,
            "stability": _stability_statistics(adapter),
            "evaluation": evaluation,
        })
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = {
        "cached_rollout_analysis": cached_result,
        "controlled_variables": [
            "supplied policy and value midpoint checkpoints",
            "prompt order and seed",
            "number of updates and PPO epochs",
            "reward model and KL beta",
            "generation settings",
            "held-out prompts and decoding settings",
        ],
        "stability_statistic": (
            "Population standard deviation of policy loss across short-fork updates; "
            "value-loss/reward/KL standard deviations and maximum policy gradient norm "
            "are also reported."
        ),
        "conditions": conditions,
    }
    save_json(result_root / "comparison.json", comparison)
    print(f"Saved clipping comparison to {result_root / 'comparison.json'}")


if __name__ == "__main__":
    main()
