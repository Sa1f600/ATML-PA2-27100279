from __future__ import annotations

import argparse
import gc

import torch

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task1_dpo.evaluate import evaluate_adapter
from task1_dpo.train import run_training


def _beta_label(beta: float):
    return f"beta_{beta:.2f}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument(
        "--skip-training",
        action="store_true",
        help="Evaluate existing beta adapters without retraining them.",
    )
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = [float(value) for value in cfg["betas"]]
    max_examples = int(cfg["short_ablation_examples"])
    output_root = repo_path("outputs/task1_dpo")
    result_root = repo_path(cfg["results_dir"]) / "beta_ablation"
    result_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for beta in betas:
        label = _beta_label(beta)
        adapter = output_root / label
        print(f"Running matched short DPO condition {label}")
        train_summary = None
        if not args.skip_training:
            train_summary = run_training(
                args.config,
                run_name=label,
                output_path=str(adapter),
                beta=beta,
                max_examples=max_examples,
            )
        elif not adapter.exists():
            raise FileNotFoundError(
                f"--skip-training was set but adapter does not exist: {adapter}"
            )

        evaluation = evaluate_adapter(
            args.config,
            str(adapter),
            label,
            output_dir=result_root / label,
            beta=beta,
        )
        summaries.append({
            "condition": label,
            "beta": beta,
            "budget": "short",
            "max_training_examples": max_examples,
            "training": train_summary,
            "evaluation": evaluation["summary"],
        })
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(result_root / "comparison.json", {
        "controlled_variables": [
            "initial policy",
            "training subset and order seed",
            "optimizer",
            "LoRA configuration",
            "held-out examples",
            "decoding configuration",
        ],
        "conditions": summaries,
    })
    print(f"Saved beta comparison to {result_root / 'comparison.json'}")


if __name__ == "__main__":
    main()
