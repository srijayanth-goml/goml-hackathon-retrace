# Module 2 — Baseline Fine-tuning

Trains the shared LoRA adapter (Qwen2.5-1.5B-Instruct, rank 16, attention + MLP
projections) on Module 1's `data/processed/train.jsonl`, plus a retain-only reference
model that never saw the flagship demo entity's facts, used later as verification
ground truth. Runs on Colab (GPU) -- see `colab_runbook.md` for the actual run
procedure. See `../plan.md`'s "Module 2 — Baseline Fine-tuning — detailed plan" for
the full design and open decisions.

## Layout

- `ft_config.py` — LoRA hyperparameters, training hyperparameters, and the flagship demo
  entity constant (currently NeuroSync Diagnostics / `G001` — a one-line change to
  retarget, see `plan.md`'s open decisions).
- `prepare_data.py` — loads `train.jsonl`, builds Module 2's own train/validation split
  (separate from Module 1's train/heldout split), builds the retain-only filtered set
  for the reference model, and renders chat-template + assistant-only loss masks.
- `lora_setup.py` — builds the PEFT `LoraConfig` from `ft_config.py`.
- `train.py` — the training entrypoint, run twice (`--mode baseline` / `--mode
  reference`) with the same code path and different data.
- `eval_quick.py` — a fast post-training sanity check (forward-QA accuracy against
  held-out probes) -- not Module 4's full verification suite.
- `manifest.py` — writes `checkpoints/manifest.json`, the shape Module 5 will read to
  drive the live revision manifest.
- `colab_runbook.md` — cell-by-cell instructions for actually running this on Colab.

## Run it (on Colab — see `colab_runbook.md`)

```bash
pip install -r requirements.txt              # repo root: pandas, pytest
pip install -r finetuning/requirements.txt    # this module: torch, transformers, peft, trl, ...
python -m data_pipeline.build_dataset          # regenerate data/processed/*.jsonl (gitignored)
python -m finetuning.train --mode baseline      # -> finetuning/checkpoints/revision-0-baseline/
python -m finetuning.train --mode reference      # -> finetuning/checkpoints/reference-model-<entity>/
```

Outputs land in `checkpoints/` (adapters + `manifest.json`, gitignored except for a
`.gitkeep`/README pointer — see `checkpoints/README.md`) and `reports/` (training
reports — hyperparameters, loss, quick-eval accuracy — tracked in git as evidence of
what was actually run).

## Evaluate a checkpoint standalone

```bash
python -m finetuning.eval_quick --which baseline                  # or --which reference
python -m finetuning.eval_quick --adapter-dir path/to/adapter --limit 10
python -m finetuning.eval_quick --which baseline --debug-samples 3 # rich per-example diagnostics
```

Must be run with `-m` from the repo root (not `python finetuning/eval_quick.py`) --
see the note at the top of `eval_quick.py`'s CLI section for why direct script
invocation breaks this repo's imports. `--debug-samples N` prints, for N real
training examples, the decoded assistant-only-masked labels (what the model was
actually trained to predict) plus the adapter's and the plain base model's generated
answers side by side -- use it to tell apart "the adapter isn't being applied" from
"the adapter learned something but it's wrong" if quick-eval accuracy looks off.

## Run the tests

```bash
pytest
```

`finetuning/tests/` is pure-Python and runs without any heavy dependency installed
(no torch/transformers/peft needed) -- it tests the retain-only filter, the
train/validation split, the LoRA target-module list, and the assistant-only
loss-masking logic (via a small fake tokenizer, not a real download). One test
(`test_build_lora_config_matches_config_py_when_peft_is_installed` (name kept as-is; tests `ft_config.py`)) is skipped unless
`peft` happens to be installed locally; that's expected off Colab.

## Decisions already locked in (see `../plan.md` for the reasoning)

- LoRA rank 16, `lora_alpha` 32, dropout 0.05, targeting all of
  `q/k/v/o_proj` + `gate/up/down_proj` (attention *and* MLP, per `../CLAUDE.md`).
- Assistant-only loss masking via a manual offset-mapping mask (tokenize the full
  conversation once, use character offsets to find the prompt/assistant boundary --
  not a specific TRL version's completion-only-loss API, and not comparing two
  separately-tokenized sequences, which broke on a real run; see
  `prepare_data.py`'s `render_and_mask` docstring), so training doesn't hard-depend
  on TRL's exact interface.
- The retain-only reference model excludes the flagship entity's own examples AND any
  relational example that merely mentions it (stricter than Module 3's later
  redact-don't-drop policy for relational examples during unlearning — see
  `prepare_data.py`'s `build_retain_only_records` docstring for why the two policies
  are deliberately different).
- Module 2's train/validation split (for training-time loss monitoring only) is
  separate from, and must never be confused with, Module 1's `heldout.jsonl` (reserved
  for Module 4).

## Open decisions still to confirm (see `../plan.md`)

- Whether NeuroSync Diagnostics stays the flagship demo entity.
- Whether more than one reference model gets trained (one now; the code supports
  re-running for a second/third target if Colab budget allows).
- Actual hyperparameters, once real Colab GPU throughput is known.
