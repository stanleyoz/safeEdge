"""
Unit tests for the model-free parts of tools/vjepa_surprise.py — no torch,
no weights, no video required.
Run: pytest tests/test_vjepa_surprise.py -v
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from tools.vjepa_surprise import (
    aoi_crop_box, aoi_patch_mask, auroc, conformal_threshold, episodes,
    exceedance_runs, false_alarms, lead_time, onsets, plan_windows, token_indices,
)


class TestConformalThreshold:
    def test_rank(self):
        # n=99, α=0.1 → k = ceil(100 × 0.9) = 90 → the 90th smallest
        scores = np.arange(1, 100)
        assert conformal_threshold(scores, 0.1) == 90

    def test_too_few_scores_gives_inf(self):
        # n=10, α=0.01 → k = ceil(11 × 0.99) = 11 > n
        assert conformal_threshold(np.arange(10), 0.01) == math.inf
        assert conformal_threshold([], 0.1) == math.inf

    def test_empirical_coverage(self):
        rng = np.random.default_rng(0)
        tau = conformal_threshold(rng.normal(size=2000), 0.05)
        fresh = rng.normal(size=20000)
        assert (fresh > tau).mean() <= 0.06


class TestOnsets:
    def test_rising_edges(self):
        t = np.arange(10.0)
        v = [0, 0, 2, 3, 0, 0, 0, 0, 2, 0]
        assert onsets(t, v, lambda x: x >= 2, min_gap_s=0) == [2.0, 8.0]

    def test_flapping_merged(self):
        t = np.arange(10.0)
        v = [0, 2, 0, 2, 0, 0, 0, 0, 0, 2]
        # second rise is 2 s after the first episode ended (< 5 s) → merged
        assert onsets(t, v, lambda x: x >= 2, min_gap_s=5) == [1.0, 9.0]


class TestRunsAndLead:
    times = np.arange(0, 20, 0.5)

    def _scores(self, hot):
        return [1.0 if lo <= t <= hi else 0.0 for t in self.times for lo, hi in [hot]]

    def test_run_detection_with_gap(self):
        s = [0, 1, 1, 0, 1, 0, 0, 1]
        assert exceedance_runs(range(8), s, 0.5, max_gap=1) == [(1, 4), (7, 7)]

    def test_lead_when_surprise_precedes_event(self):
        runs = exceedance_runs(self.times, self._scores((6.0, 11.0)), 0.5)
        assert lead_time(10.0, runs, 0.5, max_lead_s=8.0) == pytest.approx(4.0)

    def test_lead_capped(self):
        runs = exceedance_runs(self.times, self._scores((0.0, 11.0)), 0.5)
        assert lead_time(10.0, runs, 0.5, max_lead_s=3.0) == pytest.approx(3.0)

    def test_simultaneous_is_zero(self):
        runs = exceedance_runs(self.times, self._scores((10.0, 12.0)), 0.5)
        assert lead_time(10.0, runs, 0.5, max_lead_s=8.0) == 0.0

    def test_run_that_ended_earlier_is_a_miss(self):
        runs = exceedance_runs(self.times, self._scores((2.0, 4.0)), 0.5)
        assert lead_time(10.0, runs, 0.5, max_lead_s=8.0) is None

    def test_false_alarms(self):
        runs = [(1.0, 2.0), (8.0, 10.5), (17.0, 18.0), (30.0, 31.0)]
        # episode 10–25: 8–10.5 precedes it (within max_lead 5), 17–18 sits inside it;
        # 1–2 is 8 s early and 30–31 is 5 s after its end (> post 3)
        assert false_alarms(runs, [(10.0, 25.0)], max_lead_s=5.0, post_s=3.0) == 2


def test_episodes_match_onsets():
    t = np.arange(20.0)
    v = [0, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0]
    eps = episodes(t, v, lambda x: x >= 2, min_gap_s=5)
    assert eps == [(1.0, 4.0), (11.0, 12.0)]
    assert [a for a, _ in eps] == onsets(t, v, lambda x: x >= 2, min_gap_s=5)


class TestAuroc:
    def test_perfect_and_inverted(self):
        assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
        assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0

    def test_ties_count_half(self):
        assert auroc([0.5, 0.5], [0, 1]) == 0.5

    def test_single_class(self):
        assert auroc([0.1, 0.2], [1, 1]) is None


class TestGeometry:
    def test_token_layout_t_major(self):
        ctx, tgt = token_indices(t_slices=8, h=24, w=24, target_slices=2)
        assert len(ctx) == 6 * 576 and len(tgt) == 2 * 576
        assert ctx[-1] + 1 == tgt[0] and tgt[-1] == 8 * 576 - 1

    def test_aoi_mask_inside_full_frame(self):
        poly = [[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]]   # bottom half
        m = aoi_patch_mask(poly, (0, 0, 640, 480), (640, 480), grid=24)
        assert m[:12].sum() == 0 and m[12:].all()

    def test_crop_box_matches_edge_aoi(self):
        from edge.detection.aoi import AOI
        poly = [[0.03, 0.787], [0.993, 0.782], [0.99, 0.56], [0.034, 0.541]]
        a = AOI(poly, 1280, 720, pad=0.08)
        assert aoi_crop_box(poly, (1280, 720), 0.08) == (a.x0, a.y0, a.x1, a.y1)


def test_plan_windows():
    # 16 frames at 4 fps span 3.75 s; a 10 s clip with 0.5 s stride → ends 3.75 … 9.75
    ends = plan_windows(10.0, 4.0, 16, 0.5)
    assert ends[0] == pytest.approx(3.75) and ends[-1] == pytest.approx(9.75)
    assert len(ends) == 13
