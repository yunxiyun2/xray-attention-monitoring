"""Tests for the strict make_decision in decision.py."""
import pytest

from dual_factor.analysis.decision import make_decision


def test_all_criteria_pass_supported():
    """When all criteria pass (var=0.7, loading=0.9, cross=0.1, r=0.2,
    consistent=True), verdict should be 'supported'."""
    result = make_decision(
        n_factors=2,
        variance_explained=0.7,
        max_loading=0.9,
        cross_loading_max=0.1,
        factor_corr=0.2,
        subject_level_consistent=True,
    )
    assert result.verdict == 'supported'


def test_cross_loading_high_needs_correction():
    """When cross=0.4 >= 0.3, verdict should be 'needs_correction',
    NOT 'supported'."""
    result = make_decision(
        n_factors=2,
        variance_explained=0.7,
        max_loading=0.9,
        cross_loading_max=0.4,
        factor_corr=0.2,
        subject_level_consistent=True,
    )
    assert result.verdict == 'needs_correction'
    assert result.verdict != 'supported'


def test_factor_correlation_high_unsupported():
    """When r=0.8 >= 0.70, verdict should be 'unsupported'."""
    result = make_decision(
        n_factors=2,
        variance_explained=0.7,
        max_loading=0.9,
        cross_loading_max=0.1,
        factor_corr=0.8,
        subject_level_consistent=True,
    )
    assert result.verdict == 'unsupported'


def test_n_factors_1_unsupported():
    """When n_factors=1, verdict should be 'unsupported'."""
    result = make_decision(
        n_factors=1,
        variance_explained=0.7,
        max_loading=0.9,
        cross_loading_max=0.1,
        factor_corr=0.2,
        subject_level_consistent=True,
    )
    assert result.verdict == 'unsupported'


def test_criteria_has_5_items():
    """The criteria list should have 5 items (C1-C5)."""
    result = make_decision(
        n_factors=2,
        variance_explained=0.7,
        max_loading=0.9,
        cross_loading_max=0.1,
        factor_corr=0.2,
        subject_level_consistent=True,
    )
    assert len(result.criteria) == 5
