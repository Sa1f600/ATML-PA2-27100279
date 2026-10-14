import unittest

import torch

from task3_grpo.analyze_group_size import (
    assign_prompt_difficulty,
    regroup_equal_generation_budget,
)
from task3_grpo.grpo import (
    grpo_policy_loss,
    group_relative_advantages,
    mask_truncated_sequences,
)


class GRPOObjectiveTests(unittest.TestCase):
    def test_advantages_are_normalized_within_each_prompt_group(self):
        rewards = torch.tensor([1.0, 3.0, 10.0, 14.0])
        group_ids = torch.tensor([0, 0, 1, 1])
        advantages = group_relative_advantages(rewards, group_ids)
        self.assertTrue(
            torch.allclose(advantages, torch.tensor([-1.0, 1.0, -1.0, 1.0]))
        )

    def test_uninformative_group_has_zero_advantages(self):
        rewards = torch.tensor([2.0, 2.0, 1.0, 3.0])
        group_ids = torch.tensor([0, 0, 1, 1])
        advantages = group_relative_advantages(rewards, group_ids)
        self.assertTrue(torch.equal(advantages[:2], torch.zeros(2)))
        self.assertTrue(torch.allclose(advantages[2:], torch.tensor([-1.0, 1.0])))

    def test_clipped_surrogate_uses_pessimistic_minimum(self):
        ratios = torch.tensor([[1.5], [0.5]])
        new_logp = ratios.log()
        old_logp = torch.zeros_like(new_logp)
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones_like(new_logp)
        loss, diagnostics = grpo_policy_loss(
            new_logp,
            old_logp,
            advantages,
            mask,
            ref_logp=new_logp.detach(),
            eps=0.2,
            beta=0.0,
        )
        self.assertAlmostEqual(float(loss), -0.2, places=6)
        self.assertAlmostEqual(float(diagnostics["clip_fraction"]), 1.0)

    def test_masked_completion_does_not_dilute_sequence_average(self):
        logp = torch.zeros(2, 2)
        mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        loss, _ = grpo_policy_loss(
            logp,
            logp,
            torch.tensor([1.0, 100.0]),
            mask,
            ref_logp=logp,
            eps=0.2,
            beta=0.0,
        )
        self.assertAlmostEqual(float(loss), -1.0)

    def test_dr_grpo_uses_fixed_maximum_length_normalization(self):
        logp = torch.zeros(2, 2)
        mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        advantages = torch.tensor([2.0, 2.0])
        canonical, _ = grpo_policy_loss(
            logp,
            logp,
            advantages,
            mask,
            ref_logp=logp,
            eps=0.2,
            beta=0.0,
            loss_type="grpo",
            max_completion_length=2,
        )
        dr_grpo, _ = grpo_policy_loss(
            logp,
            logp,
            advantages,
            mask,
            ref_logp=logp,
            eps=0.2,
            beta=0.0,
            loss_type="dr_grpo",
            max_completion_length=2,
        )
        self.assertAlmostEqual(float(canonical), -2.0)
        self.assertAlmostEqual(float(dr_grpo), -1.5)

    def test_microbatch_weighting_matches_full_batch_objective(self):
        new_logp = torch.tensor([[0.2, -0.1], [0.0, 0.3]])
        old_logp = torch.zeros_like(new_logp)
        ref_logp = torch.tensor([[0.1, -0.2], [-0.1, 0.1]])
        mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        advantages = torch.tensor([1.0, -1.0])
        full, _ = grpo_policy_loss(
            new_logp,
            old_logp,
            advantages,
            mask,
            ref_logp,
            eps=0.2,
            beta=0.1,
        )
        pieces = []
        total_tokens = float(mask.sum())
        for index in range(2):
            chunk = slice(index, index + 1)
            value, _ = grpo_policy_loss(
                new_logp[chunk],
                old_logp[chunk],
                advantages[chunk],
                mask[chunk],
                ref_logp[chunk],
                eps=0.2,
                beta=0.1,
                policy_weight=0.5,
                kl_weight=float(mask[chunk].sum()) / total_tokens,
            )
            pieces.append(value)
        self.assertTrue(torch.allclose(full, sum(pieces), atol=1e-7))

    def test_truncated_completions_are_fully_masked(self):
        mask = torch.ones(3, 4)
        returned = mask_truncated_sequences(mask, [False, True, False])
        self.assertEqual(float(returned[0].sum()), 4.0)
        self.assertEqual(float(returned[1].sum()), 0.0)
        self.assertEqual(float(returned[2].sum()), 4.0)


class GroupSizeStudyTests(unittest.TestCase):
    @staticmethod
    def _cache():
        return {
            str(prompt): [
                {
                    "source_index": prompt,
                    "generation_index": generation,
                    "reward": float(prompt + generation),
                    "completion_tokens": 10 + generation,
                    "clipped_at_max": False,
                    "prompt_id": f"p{prompt}",
                }
                for generation in range(8)
            ]
            for prompt in range(6)
        }

    def test_regrouping_keeps_equal_generation_budget(self):
        cache = self._cache()
        budgets = {}
        for k in (2, 4, 8):
            groups = regroup_equal_generation_budget(cache, k)
            budgets[k] = sum(len(group["rows"]) for group in groups)
            self.assertTrue(all(len(group["rows"]) == k for group in groups))
        self.assertEqual(set(budgets.values()), {48})

    def test_difficulty_bins_are_equal_count(self):
        bins = assign_prompt_difficulty(self._cache())
        counts = {
            label: sum(row["difficulty_bin"] == label for row in bins.values())
            for label in ("hard", "medium", "easy")
        }
        self.assertEqual(counts, {"hard": 2, "medium": 2, "easy": 2})


if __name__ == "__main__":
    unittest.main()
