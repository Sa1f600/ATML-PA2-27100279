from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE over response positions.

    rewards, values, mask: [batch, response_steps]. Padding positions must have mask=0.
    The final valid response position bootstraps with zero.
    """
    if rewards.shape != values.shape or rewards.shape != mask.shape:
        raise ValueError("rewards, values, and mask must have identical shapes")
    if rewards.ndim != 2:
        raise ValueError("PPO token tensors must have shape [batch, response_steps]")

    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            next_value = values[:, t + 1] * next_valid
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid
        advantages[:, t] = last_adv

    returns = advantages + values
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Sampled-action KL shaping plus terminal learned reward."""
    if policy_logp.shape != ref_logp.shape or policy_logp.shape != response_mask.shape:
        raise ValueError("policy_logp, ref_logp, and response_mask must have identical shapes")
    if task_reward.ndim != 1 or task_reward.shape[0] != policy_logp.shape[0]:
        raise ValueError("task_reward must contain one scalar per response")

    rewards = -float(beta_kl) * (policy_logp - ref_logp) * response_mask
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b]
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """Return PPO clipped policy loss and diagnostics.

    ``clip_fraction`` follows the standard PPO diagnostic: the fraction of
    valid sampled tokens whose probability ratio lies outside the clip range.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not (new_logp.shape == old_logp.shape == advantage.shape == mask.shape):
        raise ValueError("all PPO policy-loss tensors must have identical shapes")

    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage

    # PPO is pessimistic: once the ratio moves too far in a direction that
    # improves the surrogate, use the clipped (smaller) objective.  The starter
    # intentionally used maximum here, which reverses PPO's trust-region logic.
    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).to(mask.dtype)
    clip_fraction = masked_mean(affected, mask)
    return loss, ratio.detach(), clip_fraction.detach()


def value_mse_loss(predicted_values, returns, mask):
    return masked_mean((predicted_values - returns) ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask
