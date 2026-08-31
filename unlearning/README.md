# Module 3 -- Unlearning Scripts

Implements NPO (Negative Preference Optimization) with neighbor-weighted retain
sampling against Module 2's Colab-trained baseline adapter, plus a plain-Gradient-
Ascent variant kept only as a deliberately-worse comparison baseline for the Erasure
Report. Runs LOCALLY (not on Colab) per `../CLAUDE.md`'s architecture decision. See
`../plan.md`'s "Module 3 -- Unlearning Scripts -- detailed plan" for the full design
and the decisions made along the way.

## Layout

- `request.py` -- `ErasureRequest`, the entity/attribute filter abstraction (Design
  Doc Section 3) that produces all three request types from one shape.
- `selectors.py` -- resolves a request into forget/retain-neighbor fact sets using
  `data/processed/neighbor_lookup.json`, and classifies every `train.jsonl` record
  into forget / retain-neighbor / retain-general.
- `redact.py` -- pure string transforms implementing `../CLAUDE.md`'s "redact the
  forgotten entity's mention rather than deleting or keeping the sentence whole"
  policy, for relational examples and bio paragraphs that mix a forgotten fact with
  retained ones.
- `forget_probe_split.py` -- holds back a slice of each targeted fact's own surface
  forms from the training batch, so forgetting can be checked for generalization
  past the exact phrasings NPO/GA trained against (closes a gap
  `data/processed/heldout.jsonl` cannot fill -- see `../plan.md`'s Module 3 step 4).
- `data.py` -- assembles the final forget/retain batches and the neighbor-weighted
  sampler; also guards against requesting an entity that turns out to be in Module
  1's heldout split (nothing genuine to unlearn -- see `requests/
  silvergate_aerospace_entity.json`'s `_comment` for how this was found).
- `npo.py` / `gradient_ascent.py` -- the two forgetting-loss implementations.
- `eval_during_unlearning.py` -- direct-QA accuracy harness against TRAIN-split
  facts (NOT `heldout.jsonl` -- see `../plan.md`'s Module 2 "Result of the actual
  run" note for why that file can't supply this signal).
- `train.py` -- the entrypoint: `python -m unlearning.train --request <path> --method npo|ga`.
- `manifest.py` -- writes revision-N entries into the SAME
  `finetuning/checkpoints/manifest.json` Module 2 established (one checkpoints root,
  one manifest -- locked decision).
- `requests/` -- example `ErasureRequest` JSON files for the demo cases discussed in
  the design docs (NeuroSync Diagnostics entity-level and CEO-cell, company-wide CEO
  attribute-type, person-wide education attribute-type, Silvergate Labs for the
  decoy-mention check).
- `reports/` -- per-run `revision-<N>_<method>_report.json/.md`.

## Run it

Needs `torch`/`transformers`/`peft` (now in the repo-ROOT `requirements.txt`, not a
separate `unlearning/requirements.txt` -- consolidated there since Module 3 is one of
several modules that all run locally, per `../plan.md`'s Module 3 decisions):

```bash
pip install -r requirements.txt
python -m unlearning.train --request unlearning/requests/neurosync_entity.json --method npo
python -m unlearning.train --request unlearning/requests/neurosync_entity.json --method ga
```

Or build a request from the CLI directly instead of a JSON file:

```bash
python -m unlearning.train --entity "NeuroSync Diagnostics" --attribute ceo --method npo
```

Every request defaults to branching fresh from `revision-0` (`--parent-revision 0`);
pass `--parent-revision N` to deliberately stack on an earlier unlearning run instead
(locked decision -- see `../plan.md`'s Open Decisions on request composition).

This does NOT require a GPU -- a 1.5B model in bf16 LoRA runs, slowly, on CPU or
Apple Silicon/MPS too (Design Doc Section 10's open hardware question; QLoRA is kept
in reserve, per Design Doc Section 5, only if a real run shows memory pressure).

## Run the tests

```bash
pytest unlearning/tests
```

Pure Python (no torch/transformers/peft needed) -- `request.py`, `selectors.py`,
`redact.py`, `forget_probe_split.py`, and `npo.py`'s loss formula are all fully
testable without a real model, run against the actual dataset (100 entities is small
enough that this repo prefers real-data tests over synthetic fixtures where
practical -- see `../CLAUDE.md`'s Conventions). `train.py`/`npo.py`'s tensor code
paths need the heavy deps and aren't covered by this test run; a clean `SystemExit`
(not a traceback) is what you get from `python -m unlearning.train` before they're
installed.

## Decisions locked in (see `../plan.md`'s Module 3 section for the reasoning)

- One checkpoints root and one manifest (`finetuning/checkpoints/manifest.json`) for
  every revision, Module 2's baseline included -- not a separate `unlearning/
  checkpoints/`.
- Every erasure request branches fresh from `revision-0` by default (avoids
  compounding utility loss across sequential requests); `--parent-revision`
  overrides this when a deliberately-sequential demo is wanted.
- `torch`/`transformers`/`peft`/`accelerate`/`safetensors` live in the repo-root
  `requirements.txt`, not a separate `unlearning/requirements.txt` -- this module and
  `data_pipeline` are the two that need to run on a plain local install.
- NPO hyperparameters (`beta=0.1`, `lambda_retain=1.0`) and GA's learning rate are
  starting points, not tuned -- confirm against real accuracy curves on the first
  real run.

## Open items still to confirm

- Whether NPO's starting hyperparameters need adjusting once a real run's accuracy
  curves come back.
- Whether the 3 non-NeuroSync example requests need their own reference models
  (Module 2 only built one) -- see `../plan.md`'s Module 3 Open Decisions.
