"""发散因子 4 指标测试。"""
from __future__ import annotations

import numpy as np

from dual_factor.features.event_detection import Fixation, Saccade, Events
from dual_factor.features.indicators import (
    spatial_entropy, gaze_transition_entropy, spatial_variance,
    saccade_direction_randomness, compute_diffuse_indicators,
)


def _make_clustered_fixations(n=20) -> list[Fixation]:
    return [Fixation(i, i+10, 0.0, 0.0, 0.01) for i in range(n)]


def _make_spread_fixations(n=20) -> list[Fixation]:
    rng = np.random.default_rng(42)
    return [Fixation(i, i+10, rng.standard_normal(), rng.standard_normal(), 0.01) for i in range(n)]


def test_spatial_entropy_clustered_low():
    fixs = _make_clustered_fixations()
    se = spatial_entropy(fixs, bins=5)
    assert se >= 0
    fixs2 = _make_spread_fixations()
    se2 = spatial_entropy(fixs2, bins=5)
    assert se <= se2


def test_gaze_transition_entropy_uniform_high():
    rng = np.random.default_rng(0)
    fixs = [
        Fixation(i, i+10, rng.uniform(-5, 5), rng.uniform(-5, 5), 0.01)
        for i in range(20)
    ]
    gte = gaze_transition_entropy(fixs, bins=5)
    assert gte >= 0


def test_spatial_variance_spread_higher():
    fixs_c = _make_clustered_fixations()
    fixs_s = _make_spread_fixations()
    sv_c = spatial_variance(fixs_c)
    sv_s = spatial_variance(fixs_s)
    assert sv_s > sv_c


def test_saccade_direction_randomness_uniform_high():
    rng = np.random.default_rng(42)
    saccades = [
        Saccade(i, i+5, 10.0, 100.0, direction_deg=float(rng.uniform(0, 360)),
                duration_sec=0.01)
        for i in range(50)
    ]
    sdr = saccade_direction_randomness(saccades)
    assert sdr >= 0
    assert sdr < 0.5  # 接近均匀


def test_compute_diffuse_indicators_all_present():
    fixs = _make_spread_fixations()
    saccades = [
        Saccade(i, i+5, 10.0, 100.0, direction_deg=45.0, duration_sec=0.01)
        for i in range(10)
    ]
    events = Events(fixations=fixs, saccades=saccades)
    out = compute_diffuse_indicators(events, bins=5)
    assert set(out.keys()) >= {"SE", "GTE", "SV", "SDR"}
    for v in out.values():
        assert np.isfinite(v)
