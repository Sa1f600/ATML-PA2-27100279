from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from task5_feedback.evaluate_math import POLICY_ORDER


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Task 5 result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pairwise_win_rate(metrics: dict) -> float | None:
    pair = metrics.get("pairwise_vs_sft") or {}
    return pair.get("ai_win_rate_vs_sft")


def build_comparison(gsm: dict, transfer: dict, diagnostics: dict) -> dict:
    if gsm.get("dataset") != "gsm" or transfer.get("dataset") != "transfer":
        raise ValueError("Expected GSM in-domain and transfer out-of-domain summaries")
    comparison = {}
    for policy in POLICY_ORDER:
        if policy not in gsm["policies"] or policy not in transfer["policies"]:
            raise ValueError(f"Missing policy {policy} from Task 5 evaluation summaries")
        inside = gsm["policies"][policy]
        outside = transfer["policies"][policy]
        gsm_win = _pairwise_win_rate(inside)
        transfer_win = _pairwise_win_rate(outside)
        comparison[policy] = {
            "in_domain_gsm": inside,
            "out_of_domain_transfer": outside,
            "transfer_change": {
                "exact_accuracy_drop": float(
                    inside["exact_answer_accuracy"] - outside["exact_answer_accuracy"]
                ),
                "format_compliance_drop": float(
                    inside["format_compliance_rate"] - outside["format_compliance_rate"]
                ),
                "mean_response_tokens_change": float(
                    outside["response_length_tokens"]["mean"]
                    - inside["response_length_tokens"]["mean"]
                ),
                "ai_pairwise_win_rate_drop_vs_sft": (
                    float(gsm_win - transfer_win)
                    if policy != "sft" and gsm_win is not None and transfer_win is not None
                    else None
                ),
            },
        }
    return {
        "policies": comparison,
        "controlled_diagnostic": diagnostics,
        "interpretation_constraints": [
            "RLVR exact accuracy and RLAIF pairwise win rate are different measurement scales.",
            "The exact verifier observes only the designated final answer.",
            "The AI judge can react to reasoning quality but also to style, verbosity, and presentation.",
            "Transfer comparisons use the same frozen policies without retraining.",
        ],
    }


def _policy_rows(comparison: dict) -> list[dict]:
    rows = []
    for policy in POLICY_ORDER:
        record = comparison["policies"][policy]
        for split, label in (("in_domain_gsm", "gsm"), ("out_of_domain_transfer", "transfer")):
            metrics = record[split]
            rows.append(
                {
                    "dataset": label,
                    "policy": policy,
                    "num_examples": metrics["num_examples"],
                    "exact_answer_accuracy": metrics["exact_answer_accuracy"],
                    "format_compliance_rate": metrics["format_compliance_rate"],
                    "mean_response_tokens": metrics["response_length_tokens"]["mean"],
                    "ai_pairwise_win_rate_vs_sft": _pairwise_win_rate(metrics),
                    "verifier_judge_agreement_rate": (
                        (metrics.get("pairwise_vs_sft") or {}).get(
                            "verifier_judge_exact_agreement_rate"
                        )
                    ),
                }
            )
    return rows


def _evaluation_disagreements(outdir: Path, dataset: str, per_policy: int = 2) -> list[dict]:
    selected = []
    for policy in ("rlvr", "rlaif"):
        path = outdir / dataset / f"pairwise_{policy}_vs_sft.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing required pairwise result: {path}")
        rows = [row for row in read_jsonl(path) if not bool(row["verifier_judge_agree"])]
        rows.sort(
            key=lambda row: (
                row["verifier_preference"] == "TIE",
                row["ai_preference"] == "TIE",
                str(row["eval_id"]),
            )
        )
        for row in rows[:per_policy]:
            selected.append(
                {
                    "evidence_type": "verifier_judge_disagreement",
                    "dataset": dataset,
                    **row,
                }
            )
    return selected


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm = _load_json(outdir / "gsm" / "summary.json")
    transfer = _load_json(outdir / "transfer" / "summary.json")
    diagnostics = _load_json(outdir / "diagnostics" / "summary.json")
    comparison = build_comparison(gsm, transfer, diagnostics)

    evidence = []
    evidence.extend(_evaluation_disagreements(outdir, "gsm"))
    evidence.extend(_evaluation_disagreements(outdir, "transfer"))
    diagnostic_candidates = _load_json(outdir / "diagnostics" / "qualitative_candidates.json")
    evidence.extend(
        {"evidence_type": "controlled_diagnostic", **row}
        for row in diagnostic_candidates
    )
    comparison["qualitative_evidence_count"] = len(evidence)
    comparison["compute_accounting"] = {
        "policy_generations": {
            "gsm": int(gsm["num_examples"] * len(POLICY_ORDER)),
            "transfer": int(transfer["num_examples"] * len(POLICY_ORDER)),
        },
        "evaluation_pairwise_judgments": {
            "gsm": int(gsm["num_examples"] * 2),
            "transfer": int(transfer["num_examples"] * 2),
        },
        "diagnostic_responses": int(diagnostics["num_responses"]),
        "diagnostic_all_pairs": int(diagnostics["num_problems"] * 10),
    }

    _write_json(outdir / "comparison.json", comparison)
    pd.DataFrame(_policy_rows(comparison)).to_csv(
        outdir / "policy_comparison.csv", index=False
    )
    _write_json(outdir / "qualitative_evidence.json", evidence)
    pd.DataFrame(evidence).to_csv(outdir / "qualitative_evidence.csv", index=False)
    print(pd.DataFrame(_policy_rows(comparison)).to_string(index=False))
    print("Saved complete Task 5 comparison to", outdir)


if __name__ == "__main__":
    main()
