import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from task4_safety.evaluate_safety import (
    _manual_outputs,
    aggregate_policy,
    category_breakdown,
    manual_agreement,
)
from task4_safety.make_audit_sheet import build_blind_audit_rows, fixed_audit_ids


class Task4SafetyTests(unittest.TestCase):
    def test_fixed_audit_ids_are_balanced_and_reproducible(self):
        rows = [
            {"xstest_id": i, "benchmark_class": "SAFE" if i < 10 else "UNSAFE"}
            for i in range(20)
        ]
        first = fixed_audit_ids(rows, per_class=4, seed=6304)
        second = fixed_audit_ids(rows, per_class=4, seed=6304)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(sum(i < 10 for i in first), 4)
        self.assertEqual(sum(i >= 10 for i in first), 4)

    def test_blind_audit_joins_every_policy_without_ai_labels(self):
        policy_rows = {}
        for policy in ("sft", "dpo", "ppo", "grpo"):
            policy_rows[policy] = [
                {
                    "xstest_id": 7,
                    "policy": policy,
                    "prompt": "benign prompt",
                    "benchmark_class": "SAFE",
                    "type": "homonyms",
                    "response": f"{policy} response",
                    "response_tokens": 3,
                }
            ]
        frame = build_blind_audit_rows(policy_rows, [7])
        self.assertEqual(len(frame), 4)
        self.assertNotIn("ai_label", frame.columns)
        self.assertTrue((frame["manual_label"] == "").all())

    def test_required_policy_rates_use_correct_denominators(self):
        rows = [
            self._row(1, "SAFE", "SAFE_ANSWER", 10),
            self._row(2, "SAFE", "OVER_REFUSAL", 20),
            self._row(3, "UNSAFE", "UNSAFE_COMPLIANCE", 30),
            self._row(4, "UNSAFE", "JUSTIFIED_REFUSAL", 40),
        ]
        summary = aggregate_policy(rows)
        self.assertEqual(summary["safe_prompt_answer_rate"], 0.5)
        self.assertEqual(summary["safe_prompt_over_refusal_rate"], 0.5)
        self.assertEqual(summary["unsafe_prompt_unsafe_compliance_rate"], 0.5)
        self.assertEqual(summary["unsafe_prompt_justified_refusal_rate"], 0.5)
        self.assertEqual(summary["ambiguous_judge_rate"], 0.0)
        self.assertEqual(summary["response_length_tokens"]["mean"], 25.0)

    def test_category_breakdown_contains_all_five_label_rates(self):
        rows = [
            self._row(1, "SAFE", "SAFE_ANSWER", 10),
            self._row(2, "SAFE", "AMBIGUOUS", 20),
        ]
        result = category_breakdown(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["safe_answer_rate"], 0.5)
        self.assertEqual(result[0]["ambiguous_rate"], 0.5)
        self.assertEqual(result[0]["over_refusal_count"], 0)

    def test_manual_agreement_and_confusion_matrix(self):
        rows = [
            {"policy": "sft", "ai_label": "SAFE_ANSWER", "manual_label": "SAFE_ANSWER"},
            {"policy": "dpo", "ai_label": "OVER_REFUSAL", "manual_label": "SAFE_ANSWER"},
            {"policy": "ppo", "ai_label": "AMBIGUOUS", "manual_label": "AMBIGUOUS"},
            {"policy": "grpo", "ai_label": "JUSTIFIED_REFUSAL", "manual_label": "JUSTIFIED_REFUSAL"},
        ]
        result = manual_agreement(rows)
        self.assertEqual(result["exact_agreement_rate"], 0.75)
        self.assertEqual(result["ai_ambiguous_rate"], 0.25)
        self.assertEqual(
            result["confusion_matrix_ai_rows_manual_columns"]["OVER_REFUSAL"]["SAFE_ANSWER"],
            1,
        )

    def test_completed_manual_audit_writes_disagreement_outputs(self):
        judged = []
        manual_labels = {
            "sft": "OVER_REFUSAL",
            "dpo": "SAFE_ANSWER",
            "ppo": "SAFE_ANSWER",
            "grpo": "SAFE_ANSWER",
        }
        for policy in ("sft", "dpo", "ppo", "grpo"):
            judged.append(
                {
                    "xstest_id": 1,
                    "policy": policy,
                    "benchmark_class": "SAFE",
                    "type": "category",
                    "prompt": "prompt",
                    "response": "response",
                    "response_tokens": 2,
                    "ai_label": "SAFE_ANSWER",
                    "ai_confidence": 0.9,
                }
            )
        with TemporaryDirectory() as directory:
            outdir = Path(directory)
            audit_path = outdir / "audit.csv"
            pd.DataFrame(
                [
                    {
                        "xstest_id": 1,
                        "policy": policy,
                        "manual_label": manual_labels[policy],
                        "manual_notes": "",
                    }
                    for policy in ("sft", "dpo", "ppo", "grpo")
                ]
            ).to_csv(audit_path, index=False)
            status = _manual_outputs(outdir, judged, audit_path)
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["disagreement_rows"], 1)
            self.assertTrue((outdir / "manual_confusion_matrix.csv").exists())
            self.assertTrue((outdir / "qualitative_disagreement_evidence.json").exists())

    @staticmethod
    def _row(xstest_id, benchmark_class, label, tokens):
        return {
            "xstest_id": xstest_id,
            "policy": "sft",
            "benchmark_class": benchmark_class,
            "type": "category",
            "ai_label": label,
            "response_tokens": tokens,
        }


if __name__ == "__main__":
    unittest.main()
