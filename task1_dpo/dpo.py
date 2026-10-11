from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return the DPO loss and diagnostics for a batch of preference pairs.

    All four inputs are *summed response-token* log-probabilities.  The prompt
    tokens must not contribute to them.  For each pair the DPO logit is

        beta * ((log pi(y+|x) - log pi(y-|x))
                - (log pi_ref(y+|x) - log pi_ref(y-|x))).

    Keeping this small function separate makes the sign in the objective easy
    to unit test independently of model and data loading.
    """

    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    tensors = (
        policy_chosen_logp,
        policy_rejected_logp,
        ref_chosen_logp,
        ref_rejected_logp,
    )
    if any(t.shape != policy_chosen_logp.shape for t in tensors[1:]):
        raise ValueError("All DPO log-probability tensors must have the same shape")

    policy_margin = (
        policy_chosen_logp
        - policy_rejected_logp
    )

    ref_margin = (
        ref_chosen_logp
        - ref_rejected_logp
    )

    logits = beta * (
        policy_margin - ref_margin
    )

    loss = -F.logsigmoid(
        logits
    ).mean()

    return loss, {
        "logit_mean":
            logits.detach().mean(),

        "policy_margin_mean":
            policy_margin.detach().mean(),

        "reference_margin_mean":
            ref_margin.detach().mean(),

        "adjusted_margin_mean":
            (policy_margin - ref_margin).detach().mean(),

        "preference_accuracy": (
            (policy_margin - ref_margin) > 0
        ).float().mean().detach(),
    }
