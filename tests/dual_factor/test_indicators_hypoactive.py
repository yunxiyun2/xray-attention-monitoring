"""呆滞因子 4 指标测试。"""
from __future__ import annotations

from dual_factor.features.event_detection import Fixation, Saccade, Events
from dual_factor.features.indicators import (
    mean_fixation_duration, peak_saccade_velocity, saccade_amplitude,
    fixation_rate, compute_hypoactive_indicators,
)


def _make_events(n_fix=5, n_sac=3, duration_sec=30.0) -> Events:
    fixations = [
        Fixation(start_idx=i*1000, end_idx=i*1000+500,
                 mean_h=0.0, mean_v=0.0, duration_sec=0.5)
        for i in range(n_fix)
    ]
    saccades = [
        Saccade(start_idx=i*1000+500, end_idx=i*1000+520,
                amplitude=10.0*(i+1), peak_velocity=100.0*(i+1),
                direction_deg=45.0, duration_sec=0.04)
        for i in range(n_sac)
    ]
    return Events(fixations=fixations, saccades=saccades)


def test_mean_fixation_duration():
    ev = _make_events()
    fd = mean_fixation_duration(ev)
    assert fd == 0.5


def test_peak_saccade_velocity():
    ev = _make_events()
    psv = peak_saccade_velocity(ev)
    assert psv == 200.0  # 均值 (100+200+300)/3


def test_saccade_amplitude():
    ev = _make_events()
    sa = saccade_amplitude(ev)
    assert sa == 20.0  # 均值 (10+20+30)/3


def test_fixation_rate():
    ev = _make_events(n_fix=5, duration_sec=30.0)
    fn = fixation_rate(ev, window_sec=30.0)
    assert abs(fn - 5 / 30.0) < 1e-6


def test_compute_hypoactive_indicators_all_present():
    ev = _make_events()
    out = compute_hypoactive_indicators(ev, window_sec=30.0)
    assert set(out.keys()) >= {"FD", "PSV", "SA", "FN"}
    assert out["FD"] == 0.5
    assert out["PSV"] == 200.0
