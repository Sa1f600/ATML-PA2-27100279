from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import set_seed

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer
from task4_safety.constants import POLICY_ORDER


REQUIRED_XSTEST_COLUMNS = {"xstest_id", "prompt", "benchmark_class", "type"}


def policy_specs(cfg):
    """Return exactly the four frozen policies required by Task 4."""
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    df = pd.read_csv(repo_path(cfg["paths"]["xstest"]))
    missing = REQUIRED_XSTEST_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"XSTest file is missing columns: {sorted(missing)}")
    if df["xstest_id"].duplicated().any():
        raise ValueError("XSTest xstest_id values must be unique")
    classes = {str(x).upper() for x in df["benchmark_class"].unique()}
    if not classes.issubset({"SAFE", "UNSAFE"}):
        raise ValueError(f"Unexpected XSTest benchmark classes: {sorted(classes)}")
    out = df.copy()
    out["benchmark_class"] = out["benchmark_class"].astype(str).str.upper()
    return out.reset_index(drop=True)


def _completed_rows(output_path: Path, policy_name: str, overwrite: bool) -> dict[int, dict]:
    if overwrite or not output_path.exists():
        return {}
    rows = read_jsonl(output_path)
    completed: dict[int, dict] = {}
    for row in rows:
        if row.get("policy") != policy_name:
            raise ValueError(f"{output_path} contains a row for a different policy")
        key = int(row["xstest_id"])
        if key in completed:
            raise ValueError(f"Duplicate xstest_id={key} in {output_path}")
        completed[key] = row
    return completed


def _ordered_rows(df: pd.DataFrame, completed: dict[int, dict]) -> list[dict]:
    return [completed[int(x)] for x in df["xstest_id"] if int(x) in completed]


def generate_for_policy(
    cfg,
    policy_name: str,
    batch_size: int = 4,
    output_path: str | Path | None = None,
    overwrite: bool = False,
):
    """Generate deterministic responses, checkpointing each batch when requested."""
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    df = load_xstest(cfg)
    path = Path(output_path) if output_path is not None else None
    completed = _completed_rows(path, policy_name, overwrite) if path is not None else {}
    expected_ids = {int(x) for x in df["xstest_id"]}
    unexpected = set(completed).difference(expected_ids)
    if unexpected:
        raise ValueError(f"Existing output contains unknown XSTest IDs: {sorted(unexpected)[:5]}")
    if len(completed) == len(df):
        print(f"[{policy_name}] already complete: {len(completed)}/{len(df)} prompts")
        return _ordered_rows(df, completed)

    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    set_seed(int(cfg["seed"]))
    remaining = df.loc[~df["xstest_id"].astype(int).isin(completed)].copy()

    try:
        for start in range(0, len(remaining), batch_size):
            chunk = remaining.iloc[start : start + batch_size]
            prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
            gen = batch_generate(
                model,
                tokenizer,
                prompts,
                max_prompt_length=int(cfg.get("safety_max_prompt_length", 256)),
                max_new_tokens=int(cfg["safety_max_new_tokens"]),
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            for (_, row), response, n_tok, terminated, truncated in zip(
                chunk.iterrows(),
                gen["responses"],
                gen["response_lengths"],
                gen["terminated_with_eos"],
                gen["truncated"],
            ):
                xstest_id = int(row["xstest_id"])
                completed[xstest_id] = {
                    "xstest_id": xstest_id,
                    "policy": policy_name,
                    "prompt": str(row["prompt"]),
                    "benchmark_class": str(row["benchmark_class"]).upper(),
                    "type": str(row["type"]),
                    "response": response,
                    "response_tokens": int(n_tok),
                    "terminated_with_eos": bool(terminated),
                    "truncated": bool(truncated),
                }
            ordered = _ordered_rows(df, completed)
            if path is not None:
                write_jsonl(path, ordered)
            print(f"[{policy_name}] generated {len(completed)}/{len(df)} prompts")
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _ordered_rows(df, completed)


def _write_manifest(outdir: Path, cfg: dict, selected: list[str], df: pd.DataFrame) -> None:
    counts = {}
    for policy in POLICY_ORDER:
        path = outdir / f"generated_{policy}.jsonl"
        counts[policy] = len(read_jsonl(path)) if path.exists() else 0
    manifest = {
        "dataset": str(cfg["paths"]["xstest"]),
        "num_prompts": int(len(df)),
        "xstest_ids": [int(x) for x in df["xstest_id"]],
        "policies_requested": selected,
        "policy_rows_saved": counts,
        "decoding": {
            "max_prompt_length": int(cfg.get("safety_max_prompt_length", 256)),
            "max_new_tokens": int(cfg["safety_max_new_tokens"]),
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        },
        "seed": int(cfg["seed"]),
    }
    (outdir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policy", choices=("all",) + POLICY_ORDER, default="all")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 4 generation requires a CUDA GPU runtime")
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_xstest(cfg)
    selected = list(POLICY_ORDER) if args.policy == "all" else [args.policy]
    batch_size = args.batch_size or int(cfg.get("safety_batch_size", 4))

    print("Policies:", selected)
    print("XSTest rows:", len(df))
    for policy_name in selected:
        generate_for_policy(
            cfg,
            policy_name,
            batch_size=batch_size,
            output_path=outdir / f"generated_{policy_name}.jsonl",
            overwrite=args.overwrite,
        )
        _write_manifest(outdir, cfg, selected, df)

    if all(
        (outdir / f"generated_{p}.jsonl").exists()
        and len(read_jsonl(outdir / f"generated_{p}.jsonl")) == len(df)
        for p in POLICY_ORDER
    ):
        combined = []
        for policy_name in POLICY_ORDER:
            combined.extend(read_jsonl(outdir / f"generated_{policy_name}.jsonl"))
        write_jsonl(outdir / "generated_all.jsonl", combined)
    print("Saved deterministic Task 4 responses to", outdir)


if __name__ == "__main__":
    main()
