"""
Plain Gradient Ascent (Design Doc Section 6's "baseline-to-beat, not shipped"):
ascends on the forget set's own likelihood with NO retain term and NO NPO sigmoid
weighting -- equivalent to NPO with beta -> 0 and lambda_retain = 0. Kept only to
generate a documented worse-comparison for the Erasure Report (plan.md's Module 3
step 6): this method has no brake on an already-forgotten example, so it is expected
-- and useful for the report -- if its neighbor-set accuracy drifts where NPO's
doesn't.
"""
from __future__ import annotations

from unlearning.npo import compute_batch_logps


def ga_loss_tensor(model, batch):
    """Returns a loss tensor for a standard minimizing optimizer step. Ordinary SFT
    loss is -logp (minimizing it INcreases likelihood); Gradient Ascent instead
    minimizes +logp directly, which DEcreases the forget batch's likelihood -- i.e.
    "ascends" the ordinary cross-entropy loss on the forget set."""
    logps = compute_batch_logps(model, batch)
    return logps.mean()
