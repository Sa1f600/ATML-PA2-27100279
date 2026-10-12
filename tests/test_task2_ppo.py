import unittest

import torch

from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


class PPOObjectiveTests(unittest.TestCase):
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
