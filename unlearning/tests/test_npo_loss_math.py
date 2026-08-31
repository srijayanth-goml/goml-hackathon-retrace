import math

import pytest

from unlearning.npo import npo_loss_from_deltas


def test_delta_zero_matches_closed_form():
    beta = 0.1
    loss = npo_loss_from_deltas([0.0], beta)
    expected = -(2.0 / beta) * (-math.log(2))  # log(sigmoid(0)) == -log(2)
    assert loss == pytest.approx(expected)


def test_loss_decreases_as_forgetting_succeeds():
    """As pi_theta assigns a forget example lower likelihood than pi_ref (Delta ->
    -infinity, i.e. genuine forgetting), Design Doc Section 6 says the NPO gradient
    signal should decay -- the loss itself should fall toward 0, unlike plain
    Gradient Ascent which keeps pushing indefinitely."""
    beta = 0.1
    loss_at_zero = npo_loss_from_deltas([0.0] * 5, beta)
    loss_after_forgetting = npo_loss_from_deltas([-1000.0] * 5, beta)  # deep into the
                                                                       # sigmoid's saturated regime
    loss_still_memorized = npo_loss_from_deltas([20.0] * 5, beta)

    assert loss_after_forgetting < loss_at_zero
    assert loss_after_forgetting == pytest.approx(0.0, abs=1e-6)
    assert loss_still_memorized > loss_at_zero


def test_rejects_empty_deltas():
    with pytest.raises(ValueError):
        npo_loss_from_deltas([], 0.1)


def test_rejects_nonpositive_beta():
    with pytest.raises(ValueError):
        npo_loss_from_deltas([0.0], 0.0)
    with pytest.raises(ValueError):
        npo_loss_from_deltas([0.0], -0.1)


def test_mean_over_batch_matches_manual_average():
    beta = 0.2
    deltas = [1.0, -1.0, 0.0]
    loss = npo_loss_from_deltas(deltas, beta)
    # log(sigmoid(-beta*d)) == -log(1 + exp(beta*d)), evaluated directly (test deltas
    # are small, so this straightforward form -- unlike npo_loss_from_deltas's own
    # numerically-stable branch -- is precise enough to cross-check against).
    manual_terms = [-math.log1p(math.exp(beta * d)) for d in deltas]
    manual = -(2.0 / beta) * (sum(manual_terms) / len(manual_terms))
    assert loss == pytest.approx(manual)
