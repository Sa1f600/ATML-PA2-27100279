from __future__ import annotations

import argparse
import random
from statistics import mean

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.metrics import masked_mean, mean_response_length, sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def _row_id(row: dict, fallback: int):
    for key in ("prompt_id", "source_index", "id"):
        if key in row:
            return row[key]
    return fallback


def _disable_dropout(model) -> None:
    """Keep likelihood ratios deterministic while retaining train-mode checkpointing."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def _response_values(value_model, generated: dict) -> torch.Tensor:
    """Return V(s_t) aligned with every generated response token a_t."""
    all_values = token_values(
        value_model,
        generated["sequences"],
        generated["attention_mask"],
    )
    start = int(generated["prompt_width"]) - 1
    steps = generated["response_ids"].shape[1]
    values = all_values[:, start : start + steps]
    if values.shape != generated["response_mask"].shape:
        raise RuntimeError(
            f"Critic/response alignment failed: values={tuple(values.shape)}, "
            f"mask={tuple(generated['response_mask'].shape)}"
        )
    return values.float()


def _mean_or_none(records: list[dict], key: str):
    values = [float(row[key]) for row in records if row.get(key) is not None]
    return float(mean(values)) if values else None


def _save_compact_logs(cfg: dict, run_name: str, history: list[dict], rollouts: list[dict], summary: dict):
    """Mirror small, Git-friendly evidence outside the ignored adapter directory."""
    destination = repo_path(cfg["results_dir"]) / "training_logs" / run_name
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "train_metrics.jsonl", history)
    write_jsonl(destination / "rollouts.jsonl", rollouts)
    save_json(destination / "training_summary.json", summary)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    if not prompts:
        raise ValueError("The PPO training prompt pool is empty")

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    run_name: str = "standard",
):
    """Continue PPO from the supplied policy and critic midpoint checkpoints."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timer = wall_timer()
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)

    num_updates = int(cfg["updates"])
    prompts_per_update = int(cfg["prompts_per_update"])
    ppo_epochs = int(cfg["ppo_epochs"])
    epsilon = float(cfg["clip_epsilon"])
    beta = float(cfg["kl_beta"])
    if num_updates < 1 or prompts_per_update < 1 or ppo_epochs < 1:
        raise ValueError("updates, prompts_per_update, and ppo_epochs must all be positive")

    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    policy_optimizer = bundle["policy_optimizer"]
    value_optimizer = bundle["value_optimizer"]
    policy_params = trainable_parameters(policy)
    value_params = trainable_parameters(value_model)
    device = next(policy.parameters()).device
    # PPO likelihood ratios assume that old/new log-probability differences are
    # caused by parameter updates, not independent dropout masks. Keep the
    # parent model in train mode (for gradient checkpointing) but disable its
    # dropout submodules, as is standard in language-model PPO implementations.
    _disable_dropout(policy)
    _disable_dropout(value_model)

    generation_cfg = cfg.get("generation", {})
    max_grad_norm = float(cfg["max_grad_norm"])
    value_coef = float(cfg["value_coef"])
    missing_eos_penalty = float(cfg.get("missing_eos_penalty", 0.0))

    # Every run/fork receives the same deterministic prompt order. When the
    # pool is exhausted, a new deterministic cycle is shuffled.
    rng = random.Random(int(cfg["seed"]))
    prompt_order = list(range(len(prompt_rows)))
    rng.shuffle(prompt_order)
    prompt_cursor = 0

    history: list[dict] = []
    rollout_records: list[dict] = []
    generated_tokens = 0

    for update_index in range(num_updates):
        selected_indices = []
        for _ in range(prompts_per_update):
            if prompt_cursor >= len(prompt_order):
                rng.shuffle(prompt_order)
                prompt_cursor = 0
            selected_indices.append(prompt_order[prompt_cursor])
            prompt_cursor += 1
        selected_rows = [prompt_rows[index] for index in selected_indices]
        prompts = [prompt_messages(row) for row in selected_rows]

        generated = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=int(cfg["max_response_length"]),
            temperature=float(generation_cfg.get("temperature", 0.7)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
            do_sample=bool(generation_cfg.get("do_sample", True)),
        )
        # batch_generate restores parent train mode recursively.
        _disable_dropout(policy)
        response_mask = generated["response_mask"].to(device)
        generated_tokens += int(response_mask.sum().item())

        # Freeze behavior-policy quantities for all PPO epochs over this
        # rollout. The disabled LoRA adapter is the supplied frozen reference.
        with torch.inference_mode():
            old_logp, old_logits = response_token_logprobs(
                policy,
                generated["sequences"],
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )
            del old_logits
            with reference_mode(policy):
                ref_logp, ref_logits = response_token_logprobs(
                    policy,
                    generated["sequences"],
                    generated["attention_mask"],
                    generated["prompt_width"],
                    generated["response_ids"],
                )
            _disable_dropout(policy)
            del ref_logits
            rollout_values = _response_values(value_model, generated)
            raw_task_reward = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                prompts,
                generated["responses"],
                max_length=int(cfg["reward_max_length"]),
            ).to(device)

        task_reward = raw_task_reward.clone()
        for row_index, terminated in enumerate(generated["terminated_with_eos"]):
            if not terminated:
                task_reward[row_index] -= missing_eos_penalty

        rewards = shaped_rewards(
            task_reward,
            old_logp.detach(),
            ref_logp.detach(),
            response_mask,
            beta,
        )
        advantages, returns = compute_gae(
            rewards,
            rollout_values.detach(),
            response_mask,
            gamma=float(cfg["gamma"]),
            lam=float(cfg["gae_lambda"]),
        )
        advantages = normalize_advantages(advantages, response_mask).detach()
        returns = returns.detach()

        rollout_kl_token = float(sampled_kl(old_logp, ref_logp, response_mask).item())
        rollout_kl_sequence = float(
            (((old_logp - ref_logp) * response_mask).sum(-1)).mean().item()
        )
        rollout_entropy = float(sample_entropy(old_logp, response_mask).item())
        rollout_length = mean_response_length(response_mask)

        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_clip_fractions = []
        epoch_outside_fractions = []
        epoch_policy_grad_norms = []
        epoch_value_grad_norms = []
        epoch_entropies = []

        for _epoch in range(ppo_epochs):
            policy_optimizer.zero_grad(set_to_none=True)
            new_logp, new_logits = response_token_logprobs(
                policy,
                generated["sequences"],
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )
            del new_logits
            policy_loss, ratio, clip_fraction = ppo_policy_loss(
                new_logp,
                old_logp.detach(),
                advantages,
                response_mask,
                eps=epsilon,
            )
            if not torch.isfinite(policy_loss):
                raise FloatingPointError(
                    f"Non-finite PPO policy loss at update {update_index + 1}"
                )
            policy_loss.backward()
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(policy_params, max_grad_norm)
            policy_optimizer.step()

            outside = ((ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)).float()
            epoch_policy_losses.append(float(policy_loss.detach().item()))
            epoch_clip_fractions.append(float(clip_fraction.item()))
            epoch_outside_fractions.append(float(masked_mean(outside, response_mask).item()))
            epoch_policy_grad_norms.append(float(torch.as_tensor(policy_grad_norm).item()))
            epoch_entropies.append(float(sample_entropy(new_logp.detach(), response_mask).item()))

            value_optimizer.zero_grad(set_to_none=True)
            predicted_values = _response_values(value_model, generated)
            value_loss = value_mse_loss(predicted_values, returns, response_mask)
            scaled_value_loss = value_coef * value_loss
            if not torch.isfinite(scaled_value_loss):
                raise FloatingPointError(
                    f"Non-finite PPO value loss at update {update_index + 1}"
                )
            scaled_value_loss.backward()
            value_grad_norm = torch.nn.utils.clip_grad_norm_(value_params, max_grad_norm)
            value_optimizer.step()
            epoch_value_losses.append(float(value_loss.detach().item()))
            epoch_value_grad_norms.append(float(torch.as_tensor(value_grad_norm).item()))

        record = {
            "run_name": run_name,
            "update": update_index + 1,
            "policy_optimizer_steps": (update_index + 1) * ppo_epochs,
            "value_optimizer_steps": (update_index + 1) * ppo_epochs,
            "prompt_indices": selected_indices,
            "generated_tokens_this_update": int(response_mask.sum().item()),
            "generated_tokens_total": generated_tokens,
            "learned_reward": float(raw_task_reward.mean().item()),
            "penalized_task_reward": float(task_reward.mean().item()),
            "sampled_kl_per_token": rollout_kl_token,
            "sampled_kl_per_response": rollout_kl_sequence,
            "policy_loss": float(mean(epoch_policy_losses)),
            "value_loss": float(mean(epoch_value_losses)),
            "entropy": float(mean(epoch_entropies)),
            "rollout_entropy": rollout_entropy,
            "gradient_norm": float(mean(epoch_policy_grad_norms)),
            "policy_gradient_norm": float(mean(epoch_policy_grad_norms)),
            "value_gradient_norm": float(mean(epoch_value_grad_norms)),
            "clip_fraction": float(mean(epoch_clip_fractions)),
            "ratio_outside_clip_fraction": float(mean(epoch_outside_fractions)),
            "response_length_tokens": rollout_length,
            "termination_rate": float(
                mean(float(value) for value in generated["terminated_with_eos"])
            ),
            "truncation_rate": float(mean(float(value) for value in generated["truncated"])),
            "value_mean": float(masked_mean(rollout_values, response_mask).item()),
            "return_mean": float(masked_mean(returns, response_mask).item()),
            "advantage_std": float(
                advantages[response_mask.bool()].std(unbiased=False).item()
            ),
            "clip_epsilon": epsilon,
            "kl_beta": beta,
            "elapsed_seconds": timer(),
        }
        history.append(record)

        per_response_kl = ((old_logp - ref_logp) * response_mask).sum(-1)
        for local_index, row in enumerate(selected_rows):
            rollout_records.append({
                "run_name": run_name,
                "update": update_index + 1,
                "source_index": selected_indices[local_index],
                "prompt_id": _row_id(row, selected_indices[local_index]),
                "prompt": prompts[local_index],
                "response": generated["responses"][local_index],
                "learned_reward": float(raw_task_reward[local_index].item()),
                "penalized_task_reward": float(task_reward[local_index].item()),
                "sampled_kl_sequence": float(per_response_kl[local_index].item()),
                "response_tokens": int(generated["response_lengths"][local_index]),
                "terminated_with_eos": bool(generated["terminated_with_eos"][local_index]),
                "truncated": bool(generated["truncated"][local_index]),
            })

        print(
            f"[{run_name}] update {update_index + 1}/{num_updates} "
            f"reward={record['learned_reward']:.4f} "
            f"kl/token={record['sampled_kl_per_token']:.5f} "
            f"policy={record['policy_loss']:.4f} "
            f"value={record['value_loss']:.4f} "
            f"clip={record['clip_fraction']:.3f} "
            f"length={record['response_length_tokens']:.1f}"
        )

    policy.save_pretrained(out)
    tokenizer.save_pretrained(out)
    value_model.save_pretrained(out / "value_model")
    write_jsonl(out / "train_metrics.jsonl", history)
    write_jsonl(out / "rollouts.jsonl", rollout_records)

    elapsed = timer()
    peak_allocated_gib = None
    peak_reserved_gib = None
    gpu_name = None
    if torch.cuda.is_available():
        peak_allocated_gib = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
        peak_reserved_gib = float(torch.cuda.max_memory_reserved() / (1024 ** 3))
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    summary = {
        "run_name": run_name,
        "output": str(out),
        "policy_start_checkpoint": str(cfg["paths"]["ppo_midpoint_policy"]),
        "value_start_checkpoint": str(cfg["paths"]["ppo_midpoint_value"]),
        "prompt_dataset": str(cfg["paths"]["rl_prompt_train"]),
        "seed": int(cfg["seed"]),
        "updates": num_updates,
        "prompts_per_update": prompts_per_update,
        "ppo_epochs": ppo_epochs,
        "policy_optimizer_steps": num_updates * ppo_epochs,
        "value_optimizer_steps": num_updates * ppo_epochs,
        "generated_tokens": generated_tokens,
        "clip_epsilon": epsilon,
        "kl_beta": beta,
        "elapsed_seconds": elapsed,
        "gpu_name": gpu_name,
        "peak_vram_allocated_gib": peak_allocated_gib,
        "peak_vram_reserved_gib": peak_reserved_gib,
        "trajectory_means": {
            key: _mean_or_none(history, key)
            for key in (
                "learned_reward",
                "sampled_kl_per_token",
                "policy_loss",
                "value_loss",
                "entropy",
                "gradient_norm",
                "clip_fraction",
                "response_length_tokens",
            )
        },
        "final_update": history[-1] if history else None,
    }
    save_json(out / "training_summary.json", summary)
    _save_compact_logs(cfg, run_name, history, rollout_records, summary)

    # Release heavyweight models before an ablation driver loads the next fork
    # or evaluator in the same Python process.
    bundle.clear()
    del policy_optimizer, value_optimizer, policy_params, value_params
    del generated, old_logp, ref_logp, new_logp, rollout_values, predicted_values
    del advantages, returns, ratio, response_mask
    del policy, value_model, reward_model, reward_tokenizer
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(
        args.config,
        args.output,
        args.updates,
        args.clip_epsilon,
        args.kl_beta,
        args.run_name,
    )


if __name__ == "__main__":
    main()
