"""决策报告测试。"""
from __future__ import annotations

from dual_factor.analysis.decision import make_decision, DecisionReport


def test_decision_separable_when_corr_low():
    report = make_decision(
        n_factors=2, factor_corr=0.45,
        max_loading=0.75, cross_loading_max=0.20,
        variance_explained=0.68,
    )
    assert report.verdict == "separable"
    assert "r < 0.70" in report.reasoning


def test_decision_unseparable_when_corr_high():
    report = make_decision(
        n_factors=2, factor_corr=0.90,
        max_loading=0.80, cross_loading_max=0.25,
        variance_explained=0.72,
    )
    assert report.verdict == "unseparable"
    assert "r > 0.85" in report.reasoning


def test_decision_ambiguous_in_middle():
    report = make_decision(
        n_factors=2, factor_corr=0.78,
        max_loading=0.65, cross_loading_max=0.28,
        variance_explained=0.62,
    )
    assert report.verdict == "ambiguous"


def test_decision_single_factor_unseparable():
    report = make_decision(
        n_factors=1, factor_corr=None,
        max_loading=0.85, cross_loading_max=0.0,
        variance_explained=0.55,
    )
    assert report.verdict == "unseparable"


def test_decision_report_to_markdown():
    report = make_decision(
        n_factors=2, factor_corr=0.50,
        max_loading=0.78, cross_loading_max=0.15,
        variance_explained=0.70,
    )
    md = report.to_markdown()
    assert "## 决策报告" in md
    assert "separable" in md
