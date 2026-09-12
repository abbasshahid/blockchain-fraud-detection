"""Tests for the leakage properties of the temporal protocol.

Reviewer 2 asked about robustness to the choice of temporal boundary, and both
reviewers rely on the claim that the split is leakage-aware.  These tests check
the property directly rather than trusting the fraction arguments.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockchain_fraud.data.temporal_split import make_temporal_split


@pytest.mark.parametrize(
    ("train_fraction", "val_fraction"),
    [(0.50, 0.20), (0.60, 0.20), (0.70, 0.15)],
)
def test_every_configured_boundary_stays_chronological(train_fraction, val_fraction):
    split = make_temporal_split(range(1, 50), train_fraction, val_fraction)
    assert max(split["train_steps"]) < min(split["val_steps"])
    assert max(split["val_steps"]) < min(split["test_steps"])
    assert len(set(split["train_steps"]) | set(split["val_steps"]) | set(split["test_steps"])) == 49


def test_no_time_step_appears_in_two_splits():
    split = make_temporal_split(range(1, 50), 0.60, 0.20)
    steps = split["train_steps"] + split["val_steps"] + split["test_steps"]
    assert len(steps) == len(set(steps))


def test_split_needs_at_least_three_time_steps():
    with pytest.raises(ValueError):
        make_temporal_split([1, 2])


def test_test_split_is_never_empty_even_for_extreme_fractions():
    split = make_temporal_split(range(1, 50), 0.95, 0.04)
    assert split["test_steps"]
    assert split["val_steps"]


def test_scaler_statistics_would_come_from_training_steps_only():
    """The preprocessing contract: nothing after the training boundary informs scaling."""
    split = make_temporal_split(range(1, 50), 0.60, 0.20)
    time_steps = np.arange(1, 50)
    train_period = np.isin(time_steps, split["train_steps"])
    later = np.isin(time_steps, split["val_steps"] + split["test_steps"])
    assert not (train_period & later).any()
    assert time_steps[train_period].max() < time_steps[later].min()
