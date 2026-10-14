from __future__ import annotations

import torch

from common.metrics import masked_mean, sample_entropy


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    """Return one scalar advantage per sampled completion.

    `group_ids[i]` identifies which prompt produced reward `rewards[i]`.
    Validate this implementation against the group-relative definition in the assignment manual.
    """
    if rewards.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("rewards and group_ids must both be one-dimensional")
    if rewards.shape != group_ids.shape:
        raise ValueError("rewards and group_ids must have identical shapes")
    if rewards.numel() == 0:
        raise ValueError("cannot compute GRPO advantages for an empty batch")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards contain non-finite values")

    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    rewards_float = rewards.float()
    for group_id in torch.unique(group_ids, sorted=True):
        group_mask = group_ids == group_id
        group_rewards = rewards_float[group_mask]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std(unbiased=False)
        if float(group_std.item()) > float(eps):
            advantages[group_mask] = (group_rewards - group_mean) / group_std
        # A zero-variance group carries no relative preference information.
        # Keeping its advantages at exactly zero avoids amplifying roundoff by
        # dividing by epsilon.
    return advantages


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
    policy_weight: float = 1.0,
    kl_weight: float = 1.0,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    if new_logp.shape != old_logp.shape or new_logp.shape != ref_logp.shape:
        raise ValueError("new, old, and reference log-probabilities must have identical shapes")
    if token_mask.shape != new_logp.shape:
        raise ValueError("token_mask must match the token log-probability shape")
    if seq_adv.ndim != 1 or seq_adv.shape[0] != new_logp.shape[0]:
        raise ValueError("seq_adv must provide one scalar per completion")
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    valid_sequence = token_mask.sum(-1) > 0
    if not bool(valid_sequence.any().item()):
        raise ValueError("GRPO loss has no unmasked completion tokens")

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence[valid_sequence].mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        if int(max_completion_length) < 1:
            raise ValueError("max_completion_length must be positive")
        # Constant normalization rather than dividing by each response's realized length.
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence[valid_sequence].mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = float(policy_weight) * policy_term + float(beta) * float(kl_weight) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    if token_mask.ndim != 2:
        raise ValueError("token_mask must have shape [completions, response_tokens]")
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    if truncated.ndim != 1 or truncated.shape[0] != token_mask.shape[0]:
        raise ValueError("truncated must provide one flag per completion")
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
