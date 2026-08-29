"""
Module 2 training entrypoint (Design Doc Section 5 / plan.md step 5). Runs on Colab
per CLAUDE.md's architecture decision -- will run on CPU but impractically slowly, and
needs finetuning/requirements.txt installed (torch, transformers, peft, trl,
accelerate, datasets), which is intentionally NOT part of the repo-root
requirements.txt (see ../CLAUDE.md Conventions). See colab_runbook.md for how to
actually run this on Colab.

One parameterized training routine, invoked twice with different training data --
never two divergent code paths -- so a judge comparing revision-0 against the
reference model is comparing "saw vs. didn't see", not "different training recipes":

    python -m finetuning.train --mode baseline      # -> revision-0 (full train set)
    python -m finetuning.train --mode reference       # -> reference model (retain-only set)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import config as root_config
from finetuning import config as ft_config
from finetuning import manifest as ft_manifest
from finetuning.lora_setup import build_lora_config, verify_target_modules_present
from finetuning.prepare_data import (
    build_retain_only_records,
    compute_prompt_token_length_stats,
    load_train_records,
    render_and_mask,
    split_records_for_sft,
)

REPO_ROOT = ft_config.FINETUNING_DIR.parent


def _require_heavy_deps() -> None:
    missing = []
    for mod in ("torch", "transformers", "peft"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            f"finetuning/train.py needs {missing} installed (and ideally trl) -- "
            "see finetuning/requirements.txt and finetuning/colab_runbook.md. "
            "This is expected to fail on a laptop with no GPU/ML stack; run this on "
            "Colab instead."
        )


def _relative_path(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO_ROOT))


def _build_hf_dataset(records: List[dict], tokenizer, max_length: int):
    from datasets import Dataset

    rows = [render_and_mask(r, tokenizer, max_length) for r in records]
    return Dataset.from_list(rows)


def _pad_collate(features: List[dict], pad_token_id: int):
    import torch

    max_len = max(len(f["input_ids"]) for f in features)
    batch: Dict[str, "torch.Tensor"] = {}
    pad_value_by_key = {"input_ids": pad_token_id, "attention_mask": 0, "labels": -100}
    for key, pad_value in pad_value_by_key.items():
        rows = []
        for f in features:
            seq = list(f[key])
            rows.append(seq + [pad_value] * (max_len - len(seq)))
        batch[key] = torch.tensor(rows, dtype=torch.long)
    return batch


def _train_one(
    train_records: List[dict],
    val_records: List[dict],
    output_dir: Path,
    run_name: str,
) -> dict:
    """Loads the frozen base model, wraps it with a fresh LoRA adapter, trains on
    `train_records` (assistant-only loss, per prepare_data.render_and_mask), saves the
    adapter to `output_dir`, and returns a small dict of results (final loss, log
    history) used by the caller to write the training report."""
    _require_heavy_deps()
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(ft_config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_stats = compute_prompt_token_length_stats(train_records + val_records, tokenizer)
    if token_stats["p99"] > ft_config.MAX_SEQ_LENGTH:
        print(
            f"WARNING: p99 token length ({token_stats['p99']}) exceeds "
            f"MAX_SEQ_LENGTH ({ft_config.MAX_SEQ_LENGTH}) -- some examples will be "
            f"truncated. Consider raising MAX_SEQ_LENGTH in finetuning/config.py. "
            f"Full stats: {token_stats}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        ft_config.MODEL_NAME,
        torch_dtype=torch.bfloat16 if ft_config.BF16 else torch.float32,
    )
    verify_target_modules_present(model)
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()

    train_ds = _build_hf_dataset(train_records, tokenizer, ft_config.MAX_SEQ_LENGTH)
    val_ds = _build_hf_dataset(val_records, tokenizer, ft_config.MAX_SEQ_LENGTH) if val_records else None

    training_args = TrainingArguments(
        output_dir=str(output_dir / "_trainer_state"),
        num_train_epochs=ft_config.NUM_EPOCHS,
        per_device_train_batch_size=ft_config.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=ft_config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=ft_config.GRAD_ACCUMULATION_STEPS,
        learning_rate=ft_config.LEARNING_RATE,
        lr_scheduler_type=ft_config.LR_SCHEDULER_TYPE,
        warmup_ratio=ft_config.WARMUP_RATIO,
        bf16=ft_config.BF16,
        logging_steps=ft_config.LOGGING_STEPS,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=ft_config.EVAL_STEPS if val_ds is not None else None,
        save_strategy="no",
        report_to=[],
        seed=ft_config.SEED,
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda features: _pad_collate(features, tokenizer.pad_token_id),
    )
    train_result = trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return {
        "final_train_loss": train_result.training_loss,
        "log_history": trainer.state.log_history,
        "token_length_stats": token_stats,
    }


def _write_report(md_path: Path, json_path: Path, report: dict) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))

    lines = [f"# {report['title']}", "", f"Run: `{report['run_name']}`", f"Created: {report['created_at']}", ""]
    lines.append("## Hyperparameters\n")
    lines.append("```json")
    lines.append(json.dumps({"lora_config": report["lora_config"], "training_args": report["training_args"]}, indent=2))
    lines.append("```\n")
    if report.get("dataset"):
        lines.append("## Dataset\n")
        lines.append("```json")
        lines.append(json.dumps(report["dataset"], indent=2))
        lines.append("```\n")
    if report.get("drop_counts_by_source_type") is not None:
        lines.append("## Examples excluded (retain-only filter)\n")
        lines.append("```json")
        lines.append(json.dumps(report["drop_counts_by_source_type"], indent=2))
        lines.append("```\n")
    lines.append(f"## Final training loss\n\n{report['final_train_loss']}\n")
    if report.get("quick_eval"):
        lines.append("## Quick sanity-check accuracy (finetuning/eval_quick.py)\n")
        lines.append("```json")
        lines.append(json.dumps(report["quick_eval"], indent=2))
        lines.append("```\n")
    md_path.write_text("\n".join(lines) + "\n")


def run_baseline(skip_quick_eval: bool = False) -> None:
    records = load_train_records()
    train_records, val_records = split_records_for_sft(records)
    print(f"[baseline] sft_train={len(train_records)} sft_val={len(val_records)}")

    result = _train_one(train_records, val_records, ft_config.BASELINE_CHECKPOINT_DIR, run_name="revision-0-baseline")

    dataset_info = {
        "train_jsonl_sha256": ft_manifest.sha256_of_file(root_config.TRAIN_JSONL_PATH),
        "num_train_examples": len(records),
        "num_sft_train": len(train_records),
        "num_sft_val": len(val_records),
        "token_length_stats": result["token_length_stats"],
    }

    quick_eval = None
    if not skip_quick_eval:
        quick_eval = _run_quick_eval_on(ft_config.BASELINE_CHECKPOINT_DIR)

    manifest_entry = ft_manifest.write_revision_0_entry(
        adapter_path=Path(_relative_path(ft_config.BASELINE_CHECKPOINT_DIR)),
        lora_config=ft_config.lora_config_as_dict(),
        training_args=ft_config.training_args_as_dict(),
        dataset_info=dataset_info,
        eval_summary=quick_eval or {},
    )

    _write_report(
        ft_config.BASELINE_REPORT_MD_PATH,
        ft_config.BASELINE_REPORT_JSON_PATH,
        {
            "title": "Module 2 baseline (revision-0) training report",
            "run_name": "revision-0-baseline",
            "created_at": manifest_entry["created_at"],
            "lora_config": ft_config.lora_config_as_dict(),
            "training_args": ft_config.training_args_as_dict(),
            "dataset": dataset_info,
            "final_train_loss": result["final_train_loss"],
            "quick_eval": quick_eval,
        },
    )
    print(f"[baseline] adapter saved to {ft_config.BASELINE_CHECKPOINT_DIR}")


def run_reference(entity: Optional[str] = None, skip_quick_eval: bool = False) -> None:
    entity = entity or ft_config.FLAGSHIP_DEMO_ENTITY
    fact_group_id = ft_config.FLAGSHIP_DEMO_FACT_GROUP_ID

    records = load_train_records()
    retained, drop_counts = build_retain_only_records(records, fact_group_id)
    print(f"[reference:{entity}] dropped {len(records) - len(retained)} examples: {drop_counts}")

    train_records, val_records = split_records_for_sft(retained)
    print(f"[reference:{entity}] sft_train={len(train_records)} sft_val={len(val_records)}")

    output_dir = ft_config.reference_checkpoint_dir()
    result = _train_one(train_records, val_records, output_dir, run_name=f"reference-model-{entity}")

    dataset_info = {
        "train_jsonl_sha256": ft_manifest.sha256_of_file(root_config.TRAIN_JSONL_PATH),
        "excluded_fact_group_id": fact_group_id,
        "num_train_examples_before_filter": len(records),
        "num_train_examples_after_filter": len(retained),
        "num_sft_train": len(train_records),
        "num_sft_val": len(val_records),
        "token_length_stats": result["token_length_stats"],
    }

    quick_eval = None
    if not skip_quick_eval:
        quick_eval = _run_quick_eval_on(output_dir)

    manifest_entry = ft_manifest.write_reference_model_entry(
        entity=entity,
        fact_group_id=fact_group_id,
        adapter_path=Path(_relative_path(output_dir)),
        excluded_counts=drop_counts,
        lora_config=ft_config.lora_config_as_dict(),
        training_args=ft_config.training_args_as_dict(),
        dataset_info=dataset_info,
        eval_summary=quick_eval or {},
    )

    _write_report(
        ft_config.REFERENCE_REPORT_MD_PATH,
        ft_config.REFERENCE_REPORT_JSON_PATH,
        {
            "title": f"Module 2 reference-model ({entity}) training report",
            "run_name": f"reference-model-{entity}",
            "created_at": manifest_entry["created_at"],
            "lora_config": ft_config.lora_config_as_dict(),
            "training_args": ft_config.training_args_as_dict(),
            "dataset": dataset_info,
            "drop_counts_by_source_type": drop_counts,
            "final_train_loss": result["final_train_loss"],
            "quick_eval": quick_eval,
        },
    )
    print(f"[reference:{entity}] adapter saved to {output_dir}")


def _run_quick_eval_on(adapter_dir: Path) -> dict:
    """Loads the base model + the just-trained adapter and runs eval_quick.py's
    sanity-check pass against data/processed/heldout.jsonl (read-only). Kept optional
    (skip_quick_eval) so a quick code/data-prep smoke test doesn't have to also pay
    for a full generation pass over the held-out set."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from finetuning.eval_quick import run_quick_eval

    tokenizer = AutoTokenizer.from_pretrained(ft_config.MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        ft_config.MODEL_NAME, torch_dtype=torch.bfloat16 if ft_config.BF16 else torch.float32
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    summary, _details = run_quick_eval(model, tokenizer)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "reference"], required=True)
    parser.add_argument("--entity", default=None, help="Override the reference-model target entity")
    parser.add_argument(
        "--skip-quick-eval", action="store_true",
        help="Skip finetuning/eval_quick.py's post-training sanity pass (faster smoke test)",
    )
    args = parser.parse_args()

    if args.mode == "baseline":
        run_baseline(skip_quick_eval=args.skip_quick_eval)
    else:
        run_reference(entity=args.entity, skip_quick_eval=args.skip_quick_eval)


if __name__ == "__main__":
    main()
