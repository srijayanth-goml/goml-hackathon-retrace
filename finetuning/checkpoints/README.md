# Checkpoints (not tracked in git)

Adapter weights land here after a Colab run of `finetuning/train.py` (see
`../colab_runbook.md`). This directory's contents are gitignored -- adapters are
binary artifacts regenerable from `data/processed/train.jsonl` + `finetuning/ft_config.py`'s
hyperparameters, not source.

Expected layout after both training runs:

```
finetuning/checkpoints/
├── revision-0-baseline/            # adapter_model.safetensors, adapter_config.json, tokenizer files
├── reference-model-<entity-slug>/    # same shape, trained on the retain-only filtered set
└── manifest.json                      # written by finetuning/manifest.py -- revision-0 + reference-model entries
```
