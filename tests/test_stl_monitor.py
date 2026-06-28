"""
Unit tests for the STL monitor — no hardware required.
Run: pytest tests/test_stl_monitor.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from edge.safety.stl_monitor import STLMonitor, SignalFrame

STL_PATH = Path(__file__).parent.parent / "config" / "stl_specs.yaml"


@pytest.fixture
def monitor():
    return STLMonitor(STL_PATH)


def make_frame(t, d_min=10.0, v_veh=0.0, d_pred=10.0, alert=0.0):
    return SignalFrame(t=t, d_min=d_min, v_veh_max=v_veh, d_pred=d_pred, alert_active=alert)


class TestPhi1MinimumSeparation:
    def test_safe_distance(self, monitor):
        s = monitor.update(make_frame(0, d_min=5.0))
        assert s.rho1 == pytest.approx(5.0 - 1.5, abs=1e-4)
        assert s.rho3 > 0   # d_pred=10.0 >> warning_horizon=3.0
        assert s.intervention_level == 0

    def test_at_critical_threshold(self, monitor):
        s = monitor.update(make_frame(0, d_min=1.5))
        assert s.rho1 == pytest.approx(0.0, abs=1e-4)

    def test_violation(self, monitor):
        s = monitor.update(make_frame(0, d_min=0.8))
        assert s.rho1 < 0
        assert s.intervention_level == 3

    def test_no_objects_uses_sentinel(self, monitor):
        s = monitor.update(make_frame(0, d_min=100.0))
        assert s.rho1 > 0
        assert s.rho3 > 0
        assert s.intervention_level == 0


class TestPhi2SpeedProximityCoupling:
    def test_vehicle_far_away(self, monitor):
        s = monitor.update(make_frame(0, d_min=8.0, v_veh=3.0))
        assert s.rho2 > 0   # d > proximity_zone → not triggered

    def test_vehicle_close_slow(self, monitor):
        s = monitor.update(make_frame(0, d_min=3.0, v_veh=1.0))
        assert s.rho2 > 0   # within zone but below speed limit

    def test_vehicle_close_fast(self, monitor):
        s = monitor.update(make_frame(0, d_min=3.0, v_veh=3.0))
        assert s.rho2 < 0   # within zone AND above speed limit → violation
        assert s.intervention_level >= 2


class TestGraduatedResponse:
    def test_full_speed_when_clear(self, monitor):
        s = monitor.update(make_frame(0, d_min=8.0))
        assert s.scale_factor == pytest.approx(1.0)

    def test_zero_speed_at_critical(self, monitor):
        s = monitor.update(make_frame(0, d_min=1.5))
        assert s.scale_factor == pytest.approx(0.0, abs=0.01)

    def test_proportional_midpoint(self, monitor):
        # At 3.25m (midpoint between proportional_zone_start=5.0 and clearance=1.5)
        s = monitor.update(make_frame(0, d_min=3.25))
        assert 0.4 < s.scale_factor < 0.6


class TestHotSwap:
    def test_apply_cloud_params(self, monitor):
        original = monitor._specs["phi1"]["params"]["clearance_critical"]
        monitor.apply_cloud_params({"phi1": {"clearance_critical": 2.0}})
        assert monitor._specs["phi1"]["params"]["clearance_critical"] == 2.0

        # Robustness should reflect new threshold
        s = monitor.update(make_frame(0, d_min=1.8))
        assert s.rho1 < 0   # 1.8 < 2.0 → violation with new param

        # Restore
        monitor.apply_cloud_params({"phi1": {"clearance_critical": original}})
