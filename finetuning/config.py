"""
Module 2 (Baseline Fine-tuning) config: LoRA hyperparameters, training hyperparameters,
target-module list, and the flagship demo entity used to build the retain-only
reference model's training set. Import from here instead of hardcoding across
prepare_data.py / lora_setup.py / train.py / eval_quick.py / manifest.py.

Follows the repo convention (../CLAUDE.md): erasure/training parameters live in a
config file per module, not as scattered magic strings. See plan.md's
"Module 2 -- Baseline Fine-tuning -- detailed plan" for the reasoning behind each
default, and ../CLAUDE.md for the locked-in architecture decisions this must respect
(LoRA rank 16-32, attention + MLP projections, one shared adapter, Qwen2.5-1.5B-Instruct,
Colab-run).
"""
from __future__ import annotations

from pathlib import Path

import config as root_config  # repo-root config.py: RAW paths, MODEL_NAME, RANDOM_SEED

FINETUNING_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = FINETUNING_DIR / "checkpoints"
REPORTS_DIR = FINETUNING_DIR / "reports"

BASELINE_CHECKPOINT_DIR = CHECKPOINTS_DIR / "revision-0-baseline"
REFERENCE_CHECKPOINT_DIR_TEMPLATE = "reference-model-{entity_slug}"

BASELINE_REPORT_JSON_PATH = REPORTS_DIR / "baseline_train_report.json"
BASELINE_REPORT_MD_PATH = REPORTS_DIR / "baseline_train_report.md"
REFERENCE_REPORT_JSON_PATH = REPORTS_DIR / "reference_train_report.json"
REFERENCE_REPORT_MD_PATH = REPORTS_DIR / "reference_train_report.md"

# --- Flagship demo entity -----------------------------------------------------
# Open decision (see plan.md's Module 2 "Open decisions"): which entity the retain-only
# reference model is built against. NeuroSync Diagnostics is the recommended default --
# the design doc's own worked example, entity-level (the primary demo mode per Design
# Doc Section 3), in the most-scrutinized confusable cluster in both docs
# (NeuroSync/NeuroWave/NeuroCore). Change these two constants together to retarget.
FLAGSHIP_DEMO_ENTITY = "NeuroSync Diagnostics"
FLAGSHIP_DEMO_FACT_GROUP_ID = "G001"

# --- LoRA config ---------------------------------------------------------------
# CLAUDE.md: "one shared LoRA adapter ... targeting both attention projections
# (q/k/v/o_proj) and MLP projections (gate/up/down_proj), rank 16-32."
LORA_RANK = 16
LORA_ALPHA = 2 * LORA_RANK
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",     # attention projections
    "gate_proj", "up_proj", "down_proj",         # SwiGLU MLP projections
]
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"

# --- Module 2's own train/validation split (NOT Module 1's train/heldout split) -
# By fact_group_id, stratified by entity_type -- reuses data_pipeline.split's exact
# stratified-shuffle method (see prepare_data.py), but with its OWN seed and fraction:
# this split is for training-time loss monitoring only, and must never be confused
# with data/processed/heldout.jsonl, which stays reserved for Module 4 (see plan.md's
# "Two gaps Module 1 left for this module to close").
TRAIN_VAL_FRACTION = 0.10
TRAIN_VAL_SEED = 4242  # deliberately different from root_config.RANDOM_SEED (Module 1's)

# --- Training hyperparameters ---------------------------------------------------
# Starting point, not final -- tune against real Colab GPU/time budget once known
# (see plan.md's Open Decisions and Design Doc Section 10).
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 4          # effective batch size 16
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
MAX_SEQ_LENGTH = 256                  # facts/paraphrases are short -- see prepare_data.py's token-length report
BF16 = True
LOGGING_STEPS = 10
EVAL_STEPS = 50
SEED = 42

# Re-exported for convenience so finetuning/*.py can do `from finetuning import config`
# and get everything (model name included) from one place.
MODEL_NAME = root_config.MODEL_NAME  # Qwen/Qwen2.5-1.5B-Instruct


def reference_checkpoint_dir(entity_slug: str | None = None) -> Path:
    """Directory a reference model's adapter is saved to / loaded from."""
    if entity_slug is None:
        entity_slug = FLAGSHIP_DEMO_ENTITY.lower().replace(" ", "-")
    return CHECKPOINTS_DIR / REFERENCE_CHECKPOINT_DIR_TEMPLATE.format(entity_slug=entity_slug)


def lora_config_as_dict() -> dict:
    """Plain-dict view of the LoRA hyperparameters, for the training report / manifest."""
    return {
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": list(LORA_TARGET_MODULES),
        "bias": LORA_BIAS,
        "task_type": LORA_TASK_TYPE,
    }


def training_args_as_dict() -> dict:
    """Plain-dict view of the training hyperparameters, for the training report / manifest."""
    return {
        "num_epochs": NUM_EPOCHS,
        "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
        "grad_accumulation_steps": GRAD_ACCUMULATION_STEPS,
        "effective_batch_size": PER_DEVICE_BATCH_SIZE * GRAD_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "warmup_ratio": WARMUP_RATIO,
        "max_seq_length": MAX_SEQ_LENGTH,
        "bf16": BF16,
        "seed": SEED,
    }
