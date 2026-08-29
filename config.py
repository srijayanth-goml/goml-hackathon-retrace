"""
Project-wide paths and constants shared across all ReTrace modules (data_pipeline,
finetuning, unlearning, verification, app/backend). Import this instead of
hardcoding paths or re-deciding a constant that's already been settled.
"""
from pathlib import Path

from common.schema import COMPANY_ATTRIBUTES, PERSON_ATTRIBUTES  # re-exported for convenience

ROOT_DIR = Path(__file__).resolve().parent

# --- Source data (read-only inputs) ---
RAW_DATA_DIR = ROOT_DIR / "data"
RAW_CSV_PATH = RAW_DATA_DIR / "knowledge_challenging_500.csv"
CONFUSABILITY_AUDIT_PATH = ROOT_DIR / "misc" / "confusability_audit.json"

# --- Module 1 outputs (consumed by Modules 2-4) ---
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
TRAIN_JSONL_PATH = PROCESSED_DATA_DIR / "train.jsonl"
HELDOUT_JSONL_PATH = PROCESSED_DATA_DIR / "heldout.jsonl"
NEIGHBOR_LOOKUP_PATH = PROCESSED_DATA_DIR / "neighbor_lookup.json"
BUILD_REPORT_JSON_PATH = PROCESSED_DATA_DIR / "build_report.json"
BUILD_REPORT_MD_PATH = PROCESSED_DATA_DIR / "build_report.md"

# --- Shared constants ---
RANDOM_SEED = 42

# Decisions locked in from plan.md's "open decisions" list:
HELDOUT_FRACTION = 0.2             # by fact_group_id, stratified by entity_type
REVERSE_QA_ON_DUPLICATE = "skip"     # vs. emitting a set-valued answer

# Base model everything downstream targets. Module 1 only needs this for documentation
# and metadata -- it does not load the tokenizer or apply the chat template itself
# (Module 2 does that at train time via HF's apply_chat_template / TRL's SFTTrainer).
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
