"""
Shared adapter loading + batch collation, used by unlearning/train.py's training
loop AND verification/*.py's evaluation code. Extracted from what used to be
unlearning/train.py's private `_load_models`/`_pad_collate` (a pure extraction --
no behavior change to Module 3's training path) so a single adapter is loaded
identically everywhere, per plan.md's Module 4 Open Decisions ("proposing a pure
extraction... flag for review since it's the one place Module 4's build reaches
outside its own directory").

Lives in unlearning/, not verification/, because the dependency runs one way:
verification already depends on unlearning (selectors.py, data.py, npo.py,
redact.py) for request resolution and loss math, and unlearning must never depend
back on verification.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from finetuning.prepare_data import render_and_mask


def load_single_adapter(adapter_dir, model_name: str, bf16: bool = True, trainable: bool = False):
    """Loads ONE base-model + adapter combination -- the single-model half of what
    used to be unlearning/train.py's `_load_models` (which always loaded TWO coupled
    copies, pi_ref + pi_theta, for NPO training; most callers -- every verification
    signal included -- only ever need one). Returns (model, tokenizer). Needs
    torch/transformers/peft installed; imported lazily so importing this module
    never requires them."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=trainable)
    if trainable:
        model.train()
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model, tokenizer


def load_ref_and_theta(adapter_dir, model_name: str, bf16: bool = True):
    """Loads the SAME adapter TWICE -- once frozen as pi_ref, once trainable as
    pi_theta -- Design Doc Section 6's 'two roles for the same starting checkpoint'.
    Used only by unlearning/train.py's NPO/GA training loop; verification never
    needs a trainable copy, so it calls load_single_adapter directly instead."""
    ref_model, tokenizer = load_single_adapter(adapter_dir, model_name, bf16=bf16, trainable=False)
    theta_model, _ = load_single_adapter(adapter_dir, model_name, bf16=bf16, trainable=True)
    return theta_model, ref_model, tokenizer


def pad_collate(records: List[dict], tokenizer, max_length: int, pad_token_id: int) -> Dict[str, "torch.Tensor"]:  # noqa: F821
    """Renders a list of {"messages": ..., "metadata": ...} records through
    render_and_mask and pads them into one batch -- the shape
    unlearning/npo.py's compute_batch_logps expects. Used by unlearning/train.py's
    training loop AND verification/mia.py's loss-based membership inference."""
    import torch

    rows = [render_and_mask(r, tokenizer, max_length) for r in records]
    max_len = max(len(row["input_ids"]) for row in rows)
    pad_value_by_key = {"input_ids": pad_token_id, "attention_mask": 0, "labels": -100}
    batch: Dict[str, "torch.Tensor"] = {}
    for key, pad_value in pad_value_by_key.items():
        seqs = []
        for row in rows:
            seq = list(row[key])
            seqs.append(seq + [pad_value] * (max_len - len(seq)))
        batch[key] = torch.tensor(seqs, dtype=torch.long)
    return batch
