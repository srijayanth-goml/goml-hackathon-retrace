# Module 1 — Data Pipeline & Training Format

Turns `data/raw/knowledge_challenging_500.csv` + `confusability_audit.json` into the
augmented, chat-formatted training corpus for Module 2 (baseline fine-tuning), plus a
neighbor lookup that Modules 3 (unlearning) and 4 (verification) both depend on.

## Run it

From the repo root:

    pip install -r requirements.txt
    python -m data_pipeline.build_dataset

Outputs land in `data/processed/`:

- `train.jsonl` — chat-formatted training examples (paraphrase / qa / bio / relational), with metadata.
- `heldout.jsonl` — eval-only probes, never trained on.
- `neighbor_lookup.json` — the field-value-based neighbor/cluster export for Modules 3/4.
- `build_report.json` / `build_report.md` — counts and skip-reasons for this build.

The command exits non-zero and prints failures to stderr if `data_pipeline/validate.py`'s
checks don't pass (e.g. a fact_group_id leaked across the train/heldout split, or a
reverse-QA example asserted a non-unique value as unique).

## Run the tests

    pytest

## Decisions already locked in (see ../CLAUDE.md and ../plan.md for the full reasoning)

- Train/heldout split is 80/20, **by `fact_group_id`**, stratified by entity_type.
- Reverse-QA examples are **skipped** (not set-valued) when the attribute value isn't
  unique to one entity — see `build_report.md`'s skip counts per attribute.
- Relational examples spanning multiple entities go to heldout only if *every* entity
  they mention is itself in heldout; otherwise train.
- Retain-sampling neighbor sets (`neighbors.py`) are computed strictly from field values
  (industry, headquarters, role, education, birth_city) — never from entity name
  strings. Name-similarity signals are exposed separately and are for decoy/verification
  checks only, not retain sampling.
