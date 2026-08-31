# Module 4 -- Verification & Erasure Report

Runs the full Design Doc Section 7 signal suite against any unlearned revision and
renders the Erasure Report the brief itself asks for (What Was Targeted / What Was
Done / Verification Results / Impact Assessment / Key Takeaways). See `../plan.md`'s
"Module 4 -- Verification & Erasure Report -- detailed plan" for the full design and
the reasoning behind each decision.

## Layout

- `config.py` -- signal thresholds (reused from `../unlearning/config.py`, not
  re-decided here), the fixed general-capability prompt set, the `previous_company`
  control-group check, and the declarative `DECOY_CHECKS` table (Silvergate Labs /
  Silvergate Therapeutics, per the review doc).
- `direct_qa.py` -- broadens `../unlearning/eval_during_unlearning.py`'s forward-QA-
  only scoring to also cover paraphrase records and reverse-direction QA (entity-name-
  as-answer, not value-as-answer), split three ways (forget / retain-neighbor /
  retain-general="unrelated") via `../unlearning/selectors.py`'s own resolution.
- `relational_probe.py` -- multi-hop probing (has the forgotten entity dropped out of
  indirect reasoning?) plus the decoy-mention / over-forgetting check.
- `mia.py` -- loss-based membership inference: forget-set examples' log-likelihood,
  percentile-ranked against a null built from `data/processed/heldout.jsonl` (genuinely
  never-trained-on text in the same format).
- `reference_comparison.py` -- compares the unlearned model against Module 2's
  retain-only reference model where one exists for the request (today: only
  NeuroSync Diagnostics); explicitly reports "unavailable" otherwise, never omits
  the section.
- `general_capability.py` -- a small fixed set of mechanically-gradable, KB-
  independent prompts, plus the `previous_company` attribute as a KB-grounded
  control group (review doc: 12 real-world brands, never colliding with anything
  synthetic).
- `report.py` -- assembles the five-section Erasure Report from every signal above;
  writes `.json` and `.md`.
- `run_verification.py` -- the entrypoint: `python -m verification.run_verification --revision N`.
- `../unlearning/model_io.py` -- the ONE shared "load a single adapter" helper this
  module uses (extracted from `../unlearning/train.py`'s private `_load_models`,
  a pure extraction with no behavior change to Module 3's training path -- lives in
  `unlearning/`, not here, because the dependency runs one way: this module already
  depends on `unlearning/selectors.py`, `data.py`, `npo.py`, `redact.py`, and
  `unlearning/` must never depend back on this module).

## Run it

Needs `torch`/`transformers`/`peft` (repo-root `requirements.txt` -- no separate
`verification/requirements.txt`; this module runs locally like `unlearning/` and
`data_pipeline/` do):

```bash
pip install -r requirements.txt
python -m verification.run_verification --revision 1
```

This does NOT require a GPU, same posture as `unlearning/train.py`. Needs at least
one real unlearned revision to exist first (`python -m unlearning.train ...` --
see `../unlearning/README.md`); revision-0 (the baseline) has no `parent_revision`
and can't be verified against itself.

If a sibling revision exists for the same `erasure_request` with a different
`method` (an `npo` run and a `ga` run for the same request), the report
automatically includes a side-by-side comparison table.

## Run the tests

```bash
pytest verification/tests
```

Every test here runs against the REAL dataset (matching `../CLAUDE.md`'s
Conventions) and needs no model -- the model-dependent pieces (`generate_answer`
calls inside `direct_qa.accuracy_on`, `relational_probe.probe_relational`,
`mia.run_mia`, etc.) are exercised for real only by actually running
`run_verification.py` against a real revision, the same posture
`../unlearning/tests/` had before Module 3's own training run happened.

## Locked decisions (see `../plan.md`'s Module 4 section for the reasoning)

- MIA reports a percentile rank against a heldout null, not a fitted likelihood-
  ratio test, and explicitly flags when the forget set is too small (< 5 examples)
  for that rank to carry real statistical weight.
- The general-capability prompt set is fixed and mechanically gradable only -- no
  open-ended prompts, so this module never needs an LLM-judge dependency.
- `reference_comparison.py` re-runs the comparison fresh against the reference
  adapter rather than trusting its stored `eval_summary`, and scores ONLY the
  subset of the forget set that reference model actually excludes (relevant for
  attribute-type requests, which forget many entities at once).
- Every verification-results section that doesn't apply to a given request states
  why explicitly (`"available": false` + `"reason"`), never a silently omitted key.

## Open items still to confirm

- Whether the fixed general-capability prompt set (12 prompts) is enough to catch
  wholesale degradation without becoming its own maintenance burden.
- Whether MIA's simple percentile-rank statistic is informative enough once a real
  run's numbers come back, or whether a fitted likelihood-ratio test is worth the
  extra complexity after all.
