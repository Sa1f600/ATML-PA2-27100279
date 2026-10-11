from __future__ import annotations

import argparse
import math

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_overlong_prompts(tokenizer, rows, max_length):
    """Remove pairs whose prompt alone cannot fit, recording exact source rows.

    Responses may be truncated to the remaining budget, but truncating a prompt
    would change the conditioning context.  Filtering is therefore explicit and
    deterministic, and the skipped indices are saved with every run.
    """
    kept, skipped = [], []
    for dataset_index, row in enumerate(rows):
        prompt_tokens = tokenizer.apply_chat_template(
            prompt_messages_from_preference(row),
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(prompt_tokens) >= max_length:
            identifier = next(
                (row[key] for key in ("prompt_id", "pair_id", "source_index", "id") if key in row),
                dataset_index,
            )
            skipped.append({
                "dataset_index": dataset_index,
                "example_id": identifier,
                "prompt_tokens": len(prompt_tokens),
                "max_sequence_length": max_length,
                "reason": "prompt_does_not_fit",
            })
        else:
            kept.append(row)
    if skipped:
        print(
            f"Filtered {len(skipped)} of {len(rows)} preference pairs because the prompt "
            f"alone does not fit max_sequence_length={max_length}."
        )
    return kept, skipped


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]
    tokenizer = load_tokenizer(cfg["base_model"])
    selected_examples = len(rows)
    rows, skipped_rows = filter_overlong_prompts(
        tokenizer,
        rows,
        int(cfg["max_sequence_length"]),
    )
    if not rows:
        raise ValueError(f"No DPO training examples fit max_sequence_length: {path}")

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader_generator = torch.Generator().manual_seed(int(cfg["seed"]))
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=loader_generator,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "selected_examples": selected_examples,
        "skipped_rows": skipped_rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def _to_device(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def _sequence_logp(model, batch):
    return response_sequence_logprobs(model, batch)[0]


def run_training(
    config_path: str,
    run_name: str,
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.mkdir(parents=True, exist_ok=True)

    model = bundle["model"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    beta_value = bundle["beta"]
    device = next(model.parameters()).device
    grad_accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg.get("epochs", 1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    if grad_accum < 1:
        raise ValueError("grad_accum_steps must be at least 1")

    timer = wall_timer()
    history = []
    optimizer_steps = 0
    examples_seen = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        group_loss = 0.0
        group_accuracy = 0.0
        group_examples = 0
        group_start = 0
        group_size = min(grad_accum, len(loader))

        for batch_index, (chosen, rejected) in enumerate(loader):
            if batch_index == group_start:
                group_size = min(grad_accum, len(loader) - group_start)

            chosen = _to_device(chosen, device)
            rejected = _to_device(rejected, device)

            # The PEFT adapter is disabled to recover the exact frozen base
            # initialization used as pi_ref.  No second model copy is needed.
            with torch.inference_mode(), reference_mode(model):
                ref_chosen = _sequence_logp(model, chosen)
                ref_rejected = _sequence_logp(model, rejected)

            policy_chosen = _sequence_logp(model, chosen)
            policy_rejected = _sequence_logp(model, rejected)
            loss, diagnostics = dpo_loss(
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
                beta_value,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite DPO loss at epoch={epoch + 1}, batch={batch_index + 1}"
                )

            (loss / group_size).backward()
            batch_examples = int(policy_chosen.shape[0])
            examples_seen += batch_examples
            group_examples += batch_examples
            group_loss += float(loss.detach().item()) * batch_examples
            group_accuracy += float(diagnostics["preference_accuracy"].item()) * batch_examples

            end_of_group = (batch_index - group_start + 1) == group_size
            if end_of_group:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(model), max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                history.append({
                    "run_name": run_name,
                    "epoch": epoch + 1,
                    "optimizer_step": optimizer_steps,
                    "examples_seen": examples_seen,
                    "beta": beta_value,
                    "dpo_loss": group_loss / max(group_examples, 1),
                    "preference_accuracy": group_accuracy / max(group_examples, 1),
                    "gradient_norm": float(torch.as_tensor(grad_norm).item()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": timer(),
                })
                print(
                    f"[{run_name}] epoch {epoch + 1}/{epochs} "
                    f"step {optimizer_steps}/{epochs * math.ceil(len(loader) / grad_accum)} "
                    f"loss={history[-1]['dpo_loss']:.4f} "
                    f"pref_acc={history[-1]['preference_accuracy']:.3f}"
                )
                group_loss = 0.0
                group_accuracy = 0.0
                group_examples = 0
                group_start = batch_index + 1

    # Only the compact LoRA adapter and tokenizer metadata are saved.
    model.save_pretrained(output)
    bundle["tokenizer"].save_pretrained(output)
    write_jsonl(output / "train_metrics.jsonl", history)
    write_jsonl(output / "skipped_overlong_prompts.jsonl", bundle["skipped_rows"])
    summary = {
        "run_name": run_name,
        "dataset": str(dataset_path or cfg["paths"]["dpo_standard_train"]),
        "output": str(output),
        "seed": int(cfg["seed"]),
        "beta": beta_value,
        "epochs": epochs,
        "batch_size": int(cfg["batch_size"]),
        "gradient_accumulation_steps": grad_accum,
        "selected_training_examples": bundle["selected_examples"],
        "training_examples": len(bundle["rows"]),
        "skipped_overlong_prompts": len(bundle["skipped_rows"]),
        "examples_seen": examples_seen,
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": timer(),
        "final_train_loss": history[-1]["dpo_loss"] if history else None,
        "final_train_preference_accuracy": (
            history[-1]["preference_accuracy"] if history else None
        ),
    }
    save_json(output / "training_summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
