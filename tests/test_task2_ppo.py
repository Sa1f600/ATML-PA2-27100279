import unittest
from types import SimpleNamespace

import torch

from common.generation import batch_generate
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


class PPOObjectiveTests(unittest.TestCase):
    def test_generated_token_ids_can_be_reused_by_autograd(self):
        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def apply_chat_template(self, *_args, **_kwargs):
                return "prompt"

            def __call__(self, *_args, **_kwargs):
                return {
                    "input_ids": torch.tensor([[1]]),
                    "attention_mask": torch.tensor([[1]]),
                }

            def decode(self, *_args, **_kwargs):
                return "response"

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = SimpleNamespace(use_cache=False)

            def generate(self, input_ids, **_kwargs):
                response = torch.tensor([[1, 2]], device=input_ids.device)
                return torch.cat([input_ids, response], dim=1)

        generated = batch_generate(
            DummyModel(),
            DummyTokenizer(),
            [[{"role": "user", "content": "test"}]],
            max_prompt_length=8,
            max_new_tokens=2,
        )
        logits = torch.randn(1, 2, 4, requires_grad=True)
        chosen = torch.gather(
            logits.log_softmax(-1),
            -1,
            generated["response_ids"].unsqueeze(-1),
        ).sum()
        chosen.backward()
        self.assertIsNotNone(logits.grad)

    def test_clipped_surrogate_uses_pessimistic_minimum(self):
        ratios = torch.tensor([[1.5, 1.5, 0.5, 0.5]])
        new_logp = ratios.log()
        old_logp = torch.zeros_like(new_logp)
        advantages = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        mask = torch.ones_like(advantages)

        loss, returned_ratios, affected = ppo_policy_loss(
            new_logp, old_logp, advantages, mask, eps=0.2
        )

        expected_objective = torch.tensor([1.2, -1.5, 0.5, -0.8]).mean()
        self.assertTrue(torch.allclose(loss, -expected_objective))
        self.assertTrue(torch.allclose(returned_ratios, ratios))
        self.assertAlmostEqual(float(affected), 1.0)

    def test_gae_stops_at_padding(self):
        rewards = torch.tensor([[0.0, 1.0, 99.0]])
        values = torch.zeros_like(rewards)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        advantages, returns = compute_gae(
            rewards, values, mask, gamma=1.0, lam=1.0
        )
        expected = torch.tensor([[1.0, 1.0, 0.0]])
        self.assertTrue(torch.allclose(advantages, expected))
        self.assertTrue(torch.allclose(returns, expected))

    def test_reward_shaping_adds_task_reward_only_at_terminal_token(self):
        task_reward = torch.tensor([2.0])
        policy_logp = torch.tensor([[-1.0, -0.5, -9.0]])
        ref_logp = torch.tensor([[-1.5, -1.0, -9.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        rewards = shaped_rewards(
            task_reward, policy_logp, ref_logp, mask, beta_kl=0.1
        )
        self.assertTrue(torch.allclose(rewards, torch.tensor([[-0.05, 1.95, 0.0]])))

    def test_advantage_normalization_respects_mask(self):
        advantages = torch.tensor([[1.0, 3.0, 100.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        normalized = normalize_advantages(advantages, mask)
        self.assertTrue(torch.allclose(normalized, torch.tensor([[-1.0, 1.0, 0.0]])))

    def test_value_loss_ignores_padding(self):
        predicted = torch.tensor([[1.0, 2.0, 100.0]])
        returns = torch.tensor([[0.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        self.assertAlmostEqual(float(value_mse_loss(predicted, returns, mask)), 2.5)


if __name__ == "__main__":
    unittest.main()
