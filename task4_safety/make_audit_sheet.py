from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from task4_safety.constants import LABELS, POLICY_ORDER


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    """Choose a reproducible, class-balanced set of XSTest prompt IDs."""
    if per_class < 1:
        raise ValueError("per_class must be positive")
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[
            meta["benchmark_class"].astype(str).str.upper() == label, "xstest_id"
        ].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def build_blind_audit_rows(policy_rows: dict[str, list[dict]], audit_ids: list[int]) -> pd.DataFrame:
    """Join the same audited prompts to every policy without exposing AI labels."""
    expected_ids = set(audit_ids)
    rows = []
    audit_order = 1
    for xstest_id in audit_ids:
        metadata = None
        for policy in POLICY_ORDER:
            indexed = {int(row["xstest_id"]): row for row in policy_rows[policy]}
            if set(indexed).intersection(expected_ids) != expected_ids:
                missing = expected_ids.difference(indexed)
                raise ValueError(f"{policy} is missing audit IDs: {sorted(missing)[:5]}")
            row = indexed[xstest_id]
            current = (
                str(row["prompt"]),
                str(row["benchmark_class"]).upper(),
                str(row["type"]),
            )
            if metadata is None:
                metadata = current
            elif current != metadata:
                raise ValueError(f"Policy metadata differs for xstest_id={xstest_id}")
            rows.append(
                {
                    "audit_order": audit_order,
                    "xstest_id": xstest_id,
                    "policy": policy,
                    "benchmark_class": current[1],
                    "type": current[2],
                    "prompt": current[0],
                    "response": str(row["response"]),
                    "response_tokens": int(row["response_tokens"]),
                    "manual_label": "",
                    "manual_notes": "",
                }
            )
            audit_order += 1
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)
    audit_path = outdir / "manual_audit_blind.csv"
    if audit_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {audit_path}; it may contain manual labels. "
            "Pass --overwrite only if you intentionally want a fresh sheet."
        )

    policy_rows = {}
    for policy in POLICY_ORDER:
        src = outdir / f"generated_{policy}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"Generate/save {policy} responses first: {src}")
        rows = read_jsonl(src)
        if any(str(row.get("policy")) != policy for row in rows):
            raise ValueError(f"Inconsistent policy field in {src}")
        policy_rows[policy] = rows

    base_rows = policy_rows["sft"]
    ids = fixed_audit_ids(
        base_rows,
        int(cfg["manual_audit_per_class"]),
        int(cfg["seed"]),
    )
    id_meta = pd.DataFrame(base_rows).set_index("xstest_id")
    id_sheet = pd.DataFrame(
        {
            "xstest_id": ids,
            "benchmark_class": [str(id_meta.loc[x, "benchmark_class"]).upper() for x in ids],
            "manual_label": [""] * len(ids),
        }
    )
    id_sheet.to_csv(outdir / "manual_audit_ids.csv", index=False)

    audit = build_blind_audit_rows(policy_rows, ids)
    audit.to_csv(audit_path, index=False)
    instructions = {
        "prompt_subset_size": len(ids),
        "policy_response_rows": len(audit),
        "safe_prompts": int((id_sheet["benchmark_class"] == "SAFE").sum()),
        "unsafe_prompts": int((id_sheet["benchmark_class"] == "UNSAFE").sum()),
        "labels": sorted(LABELS),
        "instructions": [
            "Fill manual_label for every row using exactly one allowed label.",
            "Do not open the judged_*.jsonl files until all manual labels are complete.",
            "Use manual_notes for short evidence or uncertainty notes.",
            "Do not change xstest_id, policy, prompt, response, or benchmark metadata.",
        ],
    }
    (outdir / "manual_audit_instructions.json").write_text(
        json.dumps(instructions, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(ids)} fixed prompt IDs ({len(audit)} policy responses)")
    print("Blind manual-audit sheet:", audit_path)
    print("Allowed labels:", ", ".join(sorted(LABELS)))


if __name__ == "__main__":
    main()
