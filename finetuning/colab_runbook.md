# Module 2 — Colab runbook

Baseline fine-tuning is Colab-run by design (see ../CLAUDE.md: "baseline fine-tune
... on Colab. Unlearning, verification, and serving all run locally"). This is a
cell-by-cell runbook for actually executing `finetuning/train.py` there, rather than a
hand-authored `.ipynb` -- the real logic lives in reviewable, testable `.py` files;
Colab is just the GPU runtime that runs them.

## 0. Runtime

Runtime -> Change runtime type -> GPU (T4 is fine for a 1.5B model in LoRA; use A100
if your Colab tier has it, it will just finish faster). Confirm with:

```python
!nvidia-smi
```

## 1. Get the repo onto Colab

Either clone it (if pushed to a remote you control) or upload a zip and unpack it.
Either way you need the whole repo, not just `finetuning/` -- training reuses
`common/schema.py`, `config.py`, and `data_pipeline/` to regenerate the processed
dataset deterministically (see step 3).

```python
!git clone https://github.com/srijayanth-goml/goml-hackathon-retrace.git
%cd goml-hackathon-retrace
```

## 2. Install dependencies

```python
!pip install -r requirements.txt              # pandas, pytest -- needed to rebuild the dataset
!pip install -r finetuning/requirements.txt    # torch, transformers, peft, trl, accelerate, datasets, ...
```

## 3. Rebuild the processed dataset

`data/processed/*.jsonl` is gitignored (regenerable, not source -- see `.gitignore`),
so it won't be present after a fresh clone. Rebuild it from the tracked CSV +
confusability audit (deterministic given `config.RANDOM_SEED`, so this reproduces the
exact same `train.jsonl` / `heldout.jsonl` you'd get locally):

```python
!python -m data_pipeline.build_dataset
```

This should print `OK: wrote 2264 train / 550 heldout examples ...` (Module 1's
current counts -- see `data/processed/build_report.md`). If the counts differ, stop
and check whether the CSV or `misc/confusability_audit.json` changed.

## 4. (Optional but recommended) run the test suite

```python
!python -m pytest -q
```

`finetuning/tests/` should now run its full suite (not skip the peft-dependent test)
since `finetuning/requirements.txt` is installed here.

## 5. Train the baseline adapter (revision-0)

```python
!python -m finetuning.train --mode baseline
```

Watch the logged training loss (every `LOGGING_STEPS`, see `finetuning/config.py`)
and, if a validation split exists, the eval loss. This also runs
`finetuning/eval_quick.py`'s sanity-check pass at the end (skip it with
`--skip-quick-eval` for a faster smoke test while iterating on hyperparameters).

Expect this to take from a few minutes (A100) to something like 15-30 minutes (T4)
for the default 3-epoch / ~2K-example config -- adjust `finetuning/config.py`'s
`NUM_EPOCHS` / `PER_DEVICE_BATCH_SIZE` if your actual throughput is very different
from that (see plan.md's Module 2 "Open decisions").

## 6. Train the retain-only reference model

```python
!python -m finetuning.train --mode reference
```

Same LoRA config, same hyperparameters -- only the training data differs (the
flagship demo entity's own examples and any relational mention of it are excluded;
see `finetuning/prepare_data.py`'s `build_retain_only_records`). To retarget a
different entity without touching code: `!python -m finetuning.train --mode
reference --entity "Some Other Entity"` (also update `finetuning/config.py`'s
`FLAGSHIP_DEMO_FACT_GROUP_ID` to match, or pass it through explicitly if you extend
the CLI -- see the TODO in `finetuning/train.py` if this becomes a real second/third
reference model rather than a one-off retarget).

## 7. Sanity-check the results before downloading anything

```python
import json
print(json.dumps(json.load(open("finetuning/checkpoints/manifest.json")), indent=2))
print(open("finetuning/reports/baseline_train_report.md").read())
print(open("finetuning/reports/reference_train_report.md").read())
```

Expect: baseline `overall_accuracy` high (comfortably above 90% on held-out forward
QA); reference-model accuracy near the baseline's everywhere *except* the flagship
entity's own attributes, where it should be low/near-chance. If the baseline itself
doesn't clear a high bar, something is wrong with training or the data -- don't move
on to Module 3 against a baseline that hasn't cleared this gate (plan.md's step 6).

## 8. Download the results

```python
!zip -r finetuning_outputs.zip finetuning/checkpoints finetuning/reports
```

Download `finetuning_outputs.zip` from the Colab file browser, then locally:

```bash
cd ReTrace
unzip -o /path/to/finetuning_outputs.zip
```

This should overlay `finetuning/checkpoints/{revision-0-baseline,reference-model-*,manifest.json}`
and `finetuning/reports/*` onto your local checkout. Do not commit the checkpoint
adapter files themselves (gitignored, see `finetuning/checkpoints/README.md`) --
`manifest.json` and the reports are small and worth keeping under version control as
evidence of what was actually run.

## Notes / gotchas

- **TRL vs. the manual masking fallback.** `finetuning/train.py` currently uses the
  manual assistant-only-loss-masking path (`finetuning/prepare_data.py`'s
  `render_and_mask`) rather than TRL's `SFTTrainer` completion-only-loss support, so
  it has no dependency on a specific TRL version's API surface. If you switch to
  `SFTTrainer` for convenience, keep the loss-masking behavior equivalent (verify
  with `finetuning/tests/test_loss_masking.py`'s logic against the real tokenizer)
  and note the change here.
- **Sequence length.** `finetuning/train.py` prints a warning if the data's p99 token
  length exceeds `config.MAX_SEQ_LENGTH` (256 by default) -- raise it in
  `finetuning/config.py` if you see that warning rather than silently truncating.
- **HF auth.** Qwen2.5-1.5B-Instruct is not gated as of this writing; if a future
  download fails with an auth error, set `HF_TOKEN` via `huggingface_hub.login()` or
  the `HF_TOKEN` Colab secret.
