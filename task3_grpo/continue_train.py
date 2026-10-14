from __future__ import annotations

import argparse
import random
from statistics import mean

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.metrics import mean_response_length, sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import (
    grpo_policy_loss,
    group_relative_advantages,
    mask_truncated_sequences,
)


def _row_id(row: dict, fallback: int):
    for key in ("prompt_id", "source_index", "id"):
        if key in row:
            return row[key]
    return fallback


def _disable_dropout(model) -> None:
    """Keep old/new likelihood ratios deterministic during policy training."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def _require_finite(name: str, tensor: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    values = tensor[mask.bool()] if mask is not None else tensor.reshape(-1)
    if values.numel() == 0:
        raise FloatingPointError(f"{name} has no valid values")
    if not bool(torch.isfinite(values).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")


@torch.no_grad()
def _require_finite_parameters(parameters: list[torch.nn.Parameter]) -> None:
    for index, parameter in enumerate(parameters):
        if not bool(torch.isfinite(parameter).all().item()):
            raise FloatingPointError(
                f"policy parameter {index} became non-finite after optimizer.step()"
            )


def _mean_or_none(records: list[dict], key: str):
    values = [float(row[key]) for row in records if row.get(key) is not None]
    return float(mean(values)) if values else None


def _save_progress(cfg: dict, run_name: str, output, history, rollouts) -> None:
    """Checkpoint small evidence after every completed update."""
    write_jsonl(output / "train_metrics.jsonl", history)
    write_jsonl(output / "rollouts.jsonl", rollouts)
    mirror = repo_path(cfg["results_dir"]) / "training_logs" / run_name
    mirror.mkdir(parents=True, exist_ok=True)
    write_jsonl(mirror / "train_metrics.jsonl", history)
    write_jsonl(mirror / "rollouts.jsonl", rollouts)


def _group_statistics(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float):
    standard_deviations = []
    means = []
    for group_id in torch.unique(group_ids, sorted=True):
        group_rewards = rewards[group_ids == group_id].float()
        means.append(float(group_rewards.mean().item()))
        standard_deviations.append(float(group_rewards.std(unbiased=False).item()))
    uninformative = [value <= eps for value in standard_deviations]
    return {
        "group_reward_means": means,
        "group_reward_stds": standard_deviations,
        "mean_group_reward_std": float(mean(standard_deviations)),
        "uninformative_group_fraction": float(mean(float(x) for x in uninformative)),
    }


def _length_bin(length: int, maximum: int) -> str:
    if length <= maximum / 4:
        return "short"
    if length <= maximum / 2:
        return "medium"
    return "long"


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    if not prompts:
        raise ValueError("The GRPO training prompt pool is empty")

    print("Loading GRPO tokenizer and midpoint policy...", flush=True)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    print("Loading frozen reward model...", flush=True)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    # Model construction consumes RNG state. Reset so every matched fork uses
    # the same prompt order and rollout sampling stream.
    set_seed(int(cfg["seed"]))
    optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["learning_rate"]),
        weight_decay=0.0,
    )
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
):
    """Continue GRPO from the supplied midpoint adapter."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Task 3 GRPO requires a CUDA GPU. In Colab select Runtime > "
            "Change runtime type > T4 GPU, then rerun this command."
        )
    if loss_type not in {"grpo", "dr_grpo"}:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    torch.cuda.reset_peak_memory_stats()
    timer = wall_timer()
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    num_updates = int(updates if updates is not None else cfg["updates"])
    prompts_per_update = int(cfg["prompts_per_update"])
    policy_epochs = int(cfg["policy_epochs"])
    train_micro_batch_size = int(cfg.get("train_micro_batch_size", 1))
    num_generations = int(cfg["num_generations"])
    max_completion_length = int(cfg["max_completion_length"])
    epsilon = float(cfg["clip_epsilon"])
    beta = float(cfg["kl_beta"])
    max_grad_norm = float(cfg["max_grad_norm"])
    advantage_eps = float(cfg.get("advantage_epsilon", 1e-6))
    if min(
        num_updates,
        prompts_per_update,
        policy_epochs,
        num_generations,
        train_micro_batch_size,
    ) < 1:
        raise ValueError(
            "updates, prompts_per_update, policy_epochs, num_generations, and "
            "train_micro_batch_size must be positive"
        )

    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]
    policy_params = trainable_parameters(policy)
    device = next(policy.parameters()).device
    print(
        f"[{run_name}] device={device} gpu={torch.cuda.get_device_name(0)} "
        f"loss_type={loss_type} K={num_generations} "
        f"trainable_dtype={policy_params[0].dtype}",
        flush=True,
    )
    _disable_dropout(policy)

    generation_cfg = cfg.get("generation", {})
    rng = random.Random(int(cfg["seed"]))
    prompt_order = list(range(len(prompt_rows)))
    rng.shuffle(prompt_order)
    prompt_cursor = 0
    history: list[dict] = []
    rollout_records: list[dict] = []
    generated_tokens = 0
    training_tokens = 0
    optimizer_steps = 0

    for update_index in range(num_updates):
        selected_indices = []
        for _ in range(prompts_per_update):
            if prompt_cursor >= len(prompt_order):
                rng.shuffle(prompt_order)
                prompt_cursor = 0
            selected_indices.append(prompt_order[prompt_cursor])
            prompt_cursor += 1
        selected_rows = [prompt_rows[index] for index in selected_indices]

        prompts = []
        group_ids_list = []
        row_indices = []
        generation_indices = []
        for group_index, (row_index, row) in enumerate(zip(selected_indices, selected_rows)):
            messages = prompt_messages(row)
            for generation_index in range(num_generations):
                prompts.append(messages)
                group_ids_list.append(group_index)
                row_indices.append(row_index)
                generation_indices.append(generation_index)

        generated = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=max_completion_length,
            temperature=float(generation_cfg.get("temperature", 0.7)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
            do_sample=bool(generation_cfg.get("do_sample", True)),
        )
        _disable_dropout(policy)
        raw_mask = generated["response_mask"].to(device)
        train_mask = raw_mask
        if bool(cfg.get("mask_truncated_completions", True)):
            train_mask = mask_truncated_sequences(train_mask, generated["truncated"])
        group_ids = torch.tensor(group_ids_list, device=device, dtype=torch.long)
        generated_tokens += int(raw_mask.sum().item())
        training_tokens += int(train_mask.sum().item())

        with torch.no_grad():
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
            rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                prompts,
                generated["responses"],
                max_length=int(
                    cfg.get(
                        "reward_max_length",
                        int(cfg["max_prompt_length"]) + max_completion_length,
                    )
                ),
            ).to(device).float()

        _require_finite("old policy log-probabilities", old_logp, raw_mask)
        _require_finite("reference log-probabilities", ref_logp, raw_mask)
        _require_finite("learned rewards", rewards)
        advantages = group_relative_advantages(rewards, group_ids, eps=advantage_eps).detach()
        _require_finite("group-relative advantages", advantages)
        group_stats = _group_statistics(rewards, group_ids, advantage_eps)

        rollout_kl = float(sampled_kl(old_logp, ref_logp, raw_mask).item())
        rollout_entropy = float(sample_entropy(old_logp, raw_mask).item())
        rollout_length = mean_response_length(raw_mask)
        valid_completion = train_mask.sum(-1) > 0

        epoch_losses = []
        epoch_policy_terms = []
        epoch_kl_penalties = []
        epoch_clip_fractions = []
        epoch_grad_norms = []
        epoch_entropies = []
        if bool(valid_completion.any().item()):
            valid_indices = valid_completion.nonzero(as_tuple=False).flatten()
            total_valid_sequences = int(valid_indices.numel())
            total_valid_tokens = float(train_mask.sum().item())
            for epoch_index in range(policy_epochs):
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                accumulated_policy = 0.0
                accumulated_kl = 0.0
                accumulated_clip = 0.0
                accumulated_entropy = 0.0
                for start in range(0, total_valid_sequences, train_micro_batch_size):
                    indices = valid_indices[start : start + train_micro_batch_size]
                    chunk_mask = train_mask.index_select(0, indices)
                    chunk_tokens = float(chunk_mask.sum().item())
                    policy_weight = float(indices.numel()) / total_valid_sequences
                    kl_weight = chunk_tokens / total_valid_tokens
                    new_logp, new_logits = response_token_logprobs(
                        policy,
                        generated["sequences"].index_select(0, indices),
                        generated["attention_mask"].index_select(0, indices),
                        generated["prompt_width"],
                        generated["response_ids"].index_select(0, indices),
                    )
                    del new_logits
                    _require_finite(
                        f"new policy log-probabilities at update {update_index + 1}",
                        new_logp,
                        chunk_mask,
                    )
                    loss, diagnostics = grpo_policy_loss(
                        new_logp,
                        old_logp.index_select(0, indices).detach(),
                        advantages.index_select(0, indices),
                        chunk_mask,
                        ref_logp.index_select(0, indices).detach(),
                        eps=epsilon,
                        beta=beta,
                        loss_type=loss_type,
                        max_completion_length=max_completion_length,
                        policy_weight=policy_weight,
                        kl_weight=kl_weight,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite GRPO loss at update {update_index + 1}, "
                            f"epoch {epoch_index + 1}"
                        )
                    loss.backward()
                    accumulated_loss += float(loss.detach().item())
                    accumulated_policy += (
                        policy_weight * float(diagnostics["policy_term"].item())
                    )
                    accumulated_kl += (
                        kl_weight * float(diagnostics["sampled_kl"].item())
                    )
                    accumulated_clip += (
                        kl_weight * float(diagnostics["clip_fraction"].item())
                    )
                    accumulated_entropy += (
                        kl_weight * float(diagnostics["sample_entropy"].item())
                    )
                    del new_logp, loss, diagnostics
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy_params,
                        max_grad_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    raise FloatingPointError(
                        f"Non-finite GRPO gradient at update {update_index + 1}"
                    ) from exc
                optimizer.step()
                optimizer_steps += 1
                _require_finite_parameters(policy_params)
                epoch_losses.append(accumulated_loss)
                epoch_policy_terms.append(accumulated_policy)
                epoch_kl_penalties.append(accumulated_kl)
                epoch_clip_fractions.append(accumulated_clip)
                epoch_grad_norms.append(float(torch.as_tensor(grad_norm).item()))
                epoch_entropies.append(accumulated_entropy)
        else:
            # The PA requires max-length completions to be masked. If every
            # completion hits the cap, this update contains no policy signal.
            epoch_losses = [0.0]
            epoch_policy_terms = [0.0]
            epoch_kl_penalties = [0.0]
            epoch_clip_fractions = [0.0]
            epoch_grad_norms = [0.0]
            epoch_entropies = [rollout_entropy]

        record = {
            "run_name": run_name,
            "loss_type": loss_type,
            "update": update_index + 1,
            "optimizer_steps": optimizer_steps,
            "prompt_indices": selected_indices,
            "groups_this_update": prompts_per_update,
            "completions_this_update": len(prompts),
            "generated_tokens_this_update": int(raw_mask.sum().item()),
            "generated_tokens_total": generated_tokens,
            "training_tokens_this_update": int(train_mask.sum().item()),
            "training_tokens_total": training_tokens,
            "learned_reward": float(rewards.mean().item()),
            "learned_reward_std": float(rewards.std(unbiased=False).item()),
            "mean_group_reward_std": group_stats["mean_group_reward_std"],
            "uninformative_group_fraction": group_stats[
                "uninformative_group_fraction"
            ],
            "sampled_kl_per_token": rollout_kl,
            "policy_loss": float(mean(epoch_losses)),
            "policy_term": float(mean(epoch_policy_terms)),
            "kl_penalty_estimate": float(mean(epoch_kl_penalties)),
            "entropy": float(mean(epoch_entropies)),
            "rollout_entropy": rollout_entropy,
            "gradient_norm": float(mean(epoch_grad_norms)),
            "clip_fraction": float(mean(epoch_clip_fractions)),
            "response_length_tokens": rollout_length,
            "valid_completion_fraction": float(valid_completion.float().mean().item()),
            "termination_rate": float(
                mean(float(value) for value in generated["terminated_with_eos"])
            ),
            "truncation_rate": float(mean(float(value) for value in generated["truncated"])),
            "clip_epsilon": epsilon,
            "kl_beta": beta,
            "elapsed_seconds": timer(),
        }
        history.append(record)

        per_response_kl = ((old_logp - ref_logp) * raw_mask).sum(-1)
        for local_index, response in enumerate(generated["responses"]):
            length = int(generated["response_lengths"][local_index])
            included = bool(valid_completion[local_index].item())
            denominator = (
                max(length, 1) if loss_type == "grpo" else max_completion_length
            )
            coefficient_mass = (
                abs(float(advantages[local_index].item()))
                * (length if included else 0)
                / denominator
            )
            source_row = prompt_rows[row_indices[local_index]]
            rollout_records.append({
                "run_name": run_name,
                "loss_type": loss_type,
                "update": update_index + 1,
                "source_index": row_indices[local_index],
                "prompt_id": _row_id(source_row, row_indices[local_index]),
                "generation_index": generation_indices[local_index],
                "prompt": prompts[local_index],
                "response": response,
                "learned_reward": float(rewards[local_index].item()),
                "group_advantage": float(advantages[local_index].item()),
                "sampled_kl_sequence": float(per_response_kl[local_index].item()),
                "response_tokens": length,
                "length_bin": _length_bin(length, max_completion_length),
                "normalization_denominator": int(denominator),
                "absolute_surrogate_coefficient_mass": float(coefficient_mass),
                "training_included": included,
                "terminated_with_eos": bool(
                    generated["terminated_with_eos"][local_index]
                ),
                "truncated": bool(generated["truncated"][local_index]),
            })

        _save_progress(cfg, run_name, out, history, rollout_records)
        print(
            f"[{run_name}] update {update_index + 1}/{num_updates} "
            f"reward={record['learned_reward']:.4f} "
            f"group_std={record['mean_group_reward_std']:.4f} "
            f"uninformative={record['uninformative_group_fraction']:.3f} "
            f"kl/token={record['sampled_kl_per_token']:.5f} "
            f"policy={record['policy_loss']:.4f} "
            f"clip={record['clip_fraction']:.3f} "
            f"length={record['response_length_tokens']:.1f}",
            flush=True,
        )
        # Do not keep the previous rollout's full token/log-probability tensors
        # alive while the next K-way generation is being allocated.
        del (
            generated,
            raw_mask,
            train_mask,
            group_ids,
            old_logp,
            ref_logp,
            rewards,
            advantages,
            valid_completion,
            per_response_kl,
        )

    policy.save_pretrained(out)
    tokenizer.save_pretrained(out)
    _save_progress(cfg, run_name, out, history, rollout_records)

    elapsed = timer()
    summary = {
        "run_name": run_name,
        "loss_type": loss_type,
        "output": str(out),
        "policy_start_checkpoint": str(cfg["paths"]["grpo_midpoint_policy"]),
        "prompt_dataset": str(cfg["paths"]["rl_prompt_train"]),
        "seed": int(cfg["seed"]),
        "updates": num_updates,
        "prompts_per_update": prompts_per_update,
        "num_generations": num_generations,
        "policy_epochs": policy_epochs,
        "train_micro_batch_size": train_micro_batch_size,
        "optimizer_steps": optimizer_steps,
        "generated_tokens": generated_tokens,
        "training_tokens_after_truncation_mask": training_tokens,
        "clip_epsilon": epsilon,
        "kl_beta": beta,
        "max_completion_length": max_completion_length,
        "mask_truncated_completions": bool(
            cfg.get("mask_truncated_completions", True)
        ),
        "elapsed_seconds": elapsed,
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "peak_vram_allocated_gib": float(
            torch.cuda.max_memory_allocated() / (1024**3)
        ),
        "peak_vram_reserved_gib": float(
            torch.cuda.max_memory_reserved() / (1024**3)
        ),
        "trajectory_means": {
            key: _mean_or_none(history, key)
            for key in (
                "learned_reward",
                "sampled_kl_per_token",
                "mean_group_reward_std",
                "uninformative_group_fraction",
                "policy_loss",
                "gradient_norm",
                "entropy",
                "clip_fraction",
                "response_length_tokens",
                "valid_completion_fraction",
            )
        },
        "final_update": history[-1],
    }
    save_json(out / "training_summary.json", summary)
    mirror = repo_path(cfg["results_dir"]) / "training_logs" / run_name
    save_json(mirror / "training_summary.json", summary)

    bundle.clear()
    del optimizer, policy, reward_model, reward_tokenizer
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
