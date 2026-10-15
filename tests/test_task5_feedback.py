import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from common.data import write_jsonl
from task5_feedback.compare_feedback import build_comparison
from task5_feedback.evaluate_math import (
    aggregate_pairwise,
    aggregate_policy,
    pairwise_win_value,
    prepare_math_rows,
    verifier_preference,
)
from task5_feedback.rlvr import exact_reward, extract_designated_final
from task5_feedback.score_perturbations import (
    PAIR_SPECS,
    VARIANT_ORDER,
    aggregate_preferences,
    load_diagnostic_groups,
    preference_from_scores,
    summarize_diagnostics,
)


class Task5FeedbackTests(unittest.TestCase):
    def test_exact_verifier_uses_last_designated_final_only(self):
        text = "A discarded candidate is 99. #### 12\nCorrection: #### 13"
        self.assertEqual(extract_designated_final(text), "13")
        self.assertEqual(exact_reward(text, "13"), 1.0)
        self.assertEqual(exact_reward("The gold number 13 is a distractor. #### 12", "13"), 0.0)
        self.assertEqual(exact_reward("Final answer is 13", "13"), 0.0)

    def test_prepare_math_rows_supports_course_schema_and_rejects_duplicate_ids(self):
        rows = [
            {
                "prompt_id": "gsm:1",
                "question": "What is 1+1?",
                "messages": [{"role": "user", "content": "Compute 1+1. #### format"}],
                "gold_final": "2",
            }
        ]
        prepared = prepare_math_rows(rows, "gsm")
        self.assertEqual(prepared[0]["eval_id"], "gsm:1")
        self.assertEqual(prepared[0]["gold_final"], "2")
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation ID"):
            prepare_math_rows(rows + rows, "gsm")

    def test_policy_aggregation_reports_required_metrics(self):
        rows = [
            {
                "exact_correct": True,
                "format_compliant": True,
                "response_tokens": 10,
                "terminated_with_eos": True,
                "truncated": False,
            },
            {
                "exact_correct": False,
                "format_compliant": False,
                "response_tokens": 30,
                "terminated_with_eos": False,
                "truncated": True,
            },
        ]
        summary = aggregate_policy(rows)
        self.assertEqual(summary["exact_answer_accuracy"], 0.5)
        self.assertEqual(summary["format_compliance_rate"], 0.5)
        self.assertEqual(summary["response_length_tokens"]["mean"], 20.0)
        self.assertEqual(summary["termination_rate"], 0.5)

    def test_pairwise_scoring_and_verifier_judge_agreement(self):
        self.assertEqual(verifier_preference(1, 0), "A")
        self.assertEqual(verifier_preference(0, 1), "B")
        self.assertEqual(verifier_preference(1, 1), "TIE")
        self.assertEqual(pairwise_win_value("TIE"), 0.5)
        rows = [
            {"ai_preference": "A", "verifier_preference": "A"},
            {"ai_preference": "TIE", "verifier_preference": "TIE"},
            {"ai_preference": "B", "verifier_preference": "A"},
        ]
        result = aggregate_pairwise(rows)
        self.assertEqual(result["ai_win_rate_vs_sft"], 0.5)
        self.assertAlmostEqual(result["verifier_judge_exact_agreement_rate"], 2 / 3)
        self.assertEqual(result["num_jointly_decisive"], 2)

    def test_diagnostic_group_validation_catches_missing_and_duplicate_variants(self):
        base_rows = [
            {"problem_id": "p1", "variant_type": variant}
            for variant in VARIANT_ORDER
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.jsonl"
            write_jsonl(path, base_rows)
            groups = load_diagnostic_groups(path)
            self.assertEqual(set(groups["p1"]), set(VARIANT_ORDER))
            write_jsonl(path, base_rows + [base_rows[0]])
            with self.assertRaisesRegex(ValueError, "duplicate variant"):
                load_diagnostic_groups(path)
            write_jsonl(path, base_rows[:-1])
            with self.assertRaisesRegex(ValueError, "missing variants"):
                load_diagnostic_groups(path)

    def test_controlled_diagnostic_summaries_use_better_tie_wrong_convention(self):
        self.assertEqual(preference_from_scores(1, 1), "TIE")
        self.assertEqual(preference_from_scores(1, 0), "A")
        score_rows = []
        pair_rows = []
        for problem in ("p1", "p2"):
            for variant in VARIANT_ORDER:
                score_rows.append(
                    {
                        "problem_id": problem,
                        "variant_type": variant,
                        "rlvr_exact_reward": 0.0 if "wrong_final" in variant else 1.0,
                        "rlaif_group_reward": 0.5,
                    }
                )
            for spec in PAIR_SPECS:
                pair_rows.append(
                    {
                        "problem_id": problem,
                        "perturbation": spec["perturbation"],
                        "rlvr_preference": "TIE" if spec["role"] in {
                            "reasoning_sensitivity", "filler_susceptibility"
                        } else "A",
                        "rlaif_preference": "A",
                    }
                )
        summary = summarize_diagnostics(score_rows, pair_rows)
        self.assertEqual(summary["num_responses"], 10)
        self.assertEqual(summary["sensitivity"]["S_reason"]["rlvr"], 0.0)
        self.assertEqual(summary["sensitivity"]["S_reason"]["rlaif"], 1.0)
        self.assertEqual(summary["sensitivity"]["S_outcome"]["rlvr"], 1.0)
        reasoning = [r for r in pair_rows if r["perturbation"] == "reasoning_corruption"]
        self.assertEqual(aggregate_preferences(reasoning, "rlvr_preference")["tie_rate"], 1.0)

    def test_final_comparison_calculates_transfer_drop(self):
        def policy(exact, fmt, length, win=0.5):
            return {
                "num_examples": 10,
                "exact_answer_accuracy": exact,
                "format_compliance_rate": fmt,
                "response_length_tokens": {"mean": length},
                "pairwise_vs_sft": {"ai_win_rate_vs_sft": win},
            }

        gsm = {
            "dataset": "gsm",
            "policies": {
                "sft": policy(0.5, 0.8, 100),
                "rlvr": policy(0.8, 0.9, 120, 0.7),
                "rlaif": policy(0.7, 0.85, 110, 0.75),
            },
        }
        transfer = {
            "dataset": "transfer",
            "policies": {
                "sft": policy(0.4, 0.7, 105),
                "rlvr": policy(0.6, 0.8, 130, 0.55),
                "rlaif": policy(0.65, 0.8, 115, 0.70),
            },
        }
        result = build_comparison(gsm, transfer, {"num_responses": 100})
        self.assertAlmostEqual(
            result["policies"]["rlvr"]["transfer_change"]["exact_accuracy_drop"],
            0.2,
        )
        self.assertAlmostEqual(
            result["policies"]["rlvr"]["transfer_change"][
                "ai_pairwise_win_rate_drop_vs_sft"
            ],
            0.15,
        )


if __name__ == "__main__":
    unittest.main()
