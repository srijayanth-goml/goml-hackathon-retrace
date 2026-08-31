"""
NPO (Negative Preference Optimization) loss (Design Doc Section 6):

    Delta(x, y) = log pi_theta(y|x) - log pi_ref(y|x)
    L_NPO = -(2/beta) * E[log sigmoid(-beta * Delta)]

`npo_loss_from_deltas` is pure Python/math -- no torch needed -- so the formula
itself is unit-testable (unlearning/tests/test_npo_loss_math.py) without a real
model. `sequence_logprobs_batch` / `npo_loss_tensor` / `compute_batch_logps` are the
pieces that need torch/transformers/peft (imported lazily so importing this module
never requires them installed): they run pi_theta and pi_ref forward passes over a
batch and reduce each to a per-example sequence log-likelihood over the
assistant-only-masked labels (label == -100 means ignored, matching
finetuning/prepare_data.py's render_and_mask convention).
"""
from __future__ import annotations

import math
from typing import Sequence


def _log_sigmoid(x: float) -> float:
    """Numerically stable log(sigmoid(x)) = -softplus(-x)."""
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def npo_loss_from_deltas(deltas: Sequence[float], beta: float) -> float:
    """L_NPO = -(2/beta) * mean(log sigmoid(-beta * delta)) -- Design Doc Section 6's
    closed form, taking Delta values directly (rather than raw logits) so this is
    testable against known closed-form points without any model at all -- e.g.
    delta=0 gives log sigmoid(0) = -log(2), a fixed point regardless of beta, and the
    loss should strictly decrease as delta becomes more negative (successful
    forgetting)."""
    if not deltas:
        raise ValueError("deltas must be non-empty")
    if beta <= 0:
        raise ValueError("beta must be positive")
    terms = [_log_sigmoid(-beta * d) for d in deltas]
    return -(2.0 / beta) * (sum(terms) / len(terms))


def sequence_logprobs_batch(logits, labels):
    """logits: (batch, seq, vocab) from a causal LM's forward pass; labels: (batch,
    seq) with -100 marking masked (non-assistant) positions, per
    finetuning/prepare_data.py's render_and_mask. Returns a (batch,) tensor of summed
    log-likelihood over the unmasked positions, WITH the autograd graph intact (no
    .item()/no_grad here) -- safe to call on pi_theta's logits during a real training
    step. Requires torch; imported lazily."""
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = (shift_labels != -100)
    safe_labels = shift_labels.clamp(min=0)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    return token_logprobs.sum(dim=-1)


def compute_batch_logps(model, batch):
    """One forward pass -> per-example summed log-likelihood over the assistant-only
    masked labels (batch['labels']) -- the building block both NPO's forget loss and
    the retain SFT loss are computed from."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return sequence_logprobs_batch(logits, batch["labels"])


def npo_loss_tensor(theta_logps, ref_logps, beta: float):
    """Design Doc Section 6's L_NPO, computed on tensors with autograd intact (via
    theta_logps). `ref_logps` is expected to already be detached (pi_ref runs under
    torch.no_grad() in the caller, since it is frozen and never updated)."""
    import torch.nn.functional as F

    delta = theta_logps - ref_logps
    return -(2.0 / beta) * F.logsigmoid(-beta * delta).mean()
