"""
Dual-track STL monitor — mirrors the architecture from the wheelchair SCPM paper.

Intervention Track (φ1, φ2, φ3): direct arithmetic, memoryless, runs every frame.
Verification Track (φ4, φ5): RTAMT online monitoring with past-time operators.

RTAMT pattern (from PhD paper): antecedent implies once[0,N](consequent)
  - Vacuously satisfied (positive robustness) when antecedent is false → no warmup needed.
  - update() takes list-of-tuples: update(t, [('var', val), ...])

Spec parameters are hot-swappable via apply_cloud_params() from Qwen Cloud policy manager.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import rtamt
import yaml


@dataclass
class SignalFrame:
    """One timestep of safety-relevant signals."""
    t: int                        # external timestep index (for logging)
    d_min: float                  # minimum pedestrian-vehicle distance (m)
    v_veh_max: float              # maximum vehicle speed in zone (m/s)
    d_pred: float                 # predicted minimum distance at t+T_horizon (m)
    alert_active: float           # 1.0 if an alert is currently active, else 0.0
    v_closing: float = 0.0        # radial closing speed of nearest pair (m/s, +ve = converging)


@dataclass
class SafetyState:
    """Output of one STL monitor evaluation cycle."""
    t: int
    timestamp: float

    rho1: float                   # φ1 minimum separation margin
    rho2: float                   # φ2 speed-proximity coupling
    rho3: float                   # φ3 predictive near-miss (arithmetic)
    rho4: Optional[float]         # φ4 emergency stop compliance (RTAMT)
    rho5: Optional[float]         # φ5 post-alert clearance (RTAMT)

    rho_min: float
    intervention_level: int       # 0=safe 1=awareness 2=warning 3=emergency
    scale_factor: float           # graduated velocity scaling [0,1]

    signals: SignalFrame = field(repr=False)


class STLMonitor:
    def __init__(self, spec_path: str | Path):
        self._spec_path = Path(spec_path)
        self._rtamt_t = 0
        self._specs: dict = {}
        self._rtamt_phi4: Optional[rtamt.StlDiscreteTimeSpecification] = None
        self._rtamt_phi5: Optional[rtamt.StlDiscreteTimeSpecification] = None
        self.reload_specs()

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, frame: SignalFrame) -> SafetyState:
        self._rtamt_t += 1

        rho1 = self._eval_phi1(frame)
        rho2 = self._eval_phi2(frame)
        rho3 = self._eval_phi3(frame)
        rho4 = self._eval_phi4(frame)
        rho5 = self._eval_phi5(frame)

        available = [r for r in (rho1, rho2, rho3, rho4, rho5) if r is not None]
        rho_min = min(available) if available else 0.0

        level = self._intervention_level(rho1, rho2, rho3, frame.v_closing, frame.d_min)
        scale = self._scale_factor(frame.d_min)

        return SafetyState(
            t=frame.t,
            timestamp=time.time(),
            rho1=rho1, rho2=rho2, rho3=rho3, rho4=rho4, rho5=rho5,
            rho_min=rho_min,
            intervention_level=level,
            scale_factor=scale,
            signals=frame,
        )

    def reload_specs(self) -> None:
        with open(self._spec_path) as f:
            cfg = yaml.safe_load(f)
        self._specs = cfg["specs"]
        self._intervention_cfg = cfg["intervention"]
        self._graduated = cfg["graduated_response"]
        self._rebuild_rtamt()

    def apply_cloud_params(self, patch: dict) -> None:
        for spec_id, params in patch.items():
            if spec_id in self._specs:
                self._specs[spec_id]["params"].update(params)
        self._rebuild_rtamt()

    # ── Intervention track (arithmetic, memoryless) ──────────────────────────

    def _eval_phi1(self, f: SignalFrame) -> float:
        p = self._specs["phi1"]["params"]
        return f.d_min - p["clearance_critical"]

    def _eval_phi2(self, f: SignalFrame) -> float:
        p = self._specs["phi2"]["params"]
        return max(
            f.d_min - p["proximity_zone"],
            p["speed_limit_slow"] - f.v_veh_max,
        )

    def _eval_phi3(self, f: SignalFrame) -> float:
        """Predictive near-miss: d_pred is already a future-looking signal — direct arithmetic."""
        p = self._specs["phi3"]["params"]
        return f.d_pred - p["warning_horizon"]

    # ── Verification track (RTAMT, past-time implies+once pattern) ───────────

    def _eval_phi4(self, f: SignalFrame) -> Optional[float]:
        if self._rtamt_phi4 is None:
            return None
        val = self._rtamt_phi4.update(
            self._rtamt_t,
            [("d_min", f.d_min), ("v_veh_max", f.v_veh_max)],
        )
        return float(val) if val is not None else None

    def _eval_phi5(self, f: SignalFrame) -> Optional[float]:
        if self._rtamt_phi5 is None:
            return None
        val = self._rtamt_phi5.update(
            self._rtamt_t,
            [("alert_active", f.alert_active), ("d_min", f.d_min)],
        )
        return float(val) if val is not None else None

    # ── RTAMT initialisation ─────────────────────────────────────────────────

    def _rebuild_rtamt(self) -> None:
        self._rtamt_phi4 = self._build_spec(
            vars=[("d_min", "float"), ("v_veh_max", "float")],
            stl=self._specs["phi4"]["stl"].format(**self._specs["phi4"]["params"]),
        )
        self._rtamt_phi5 = self._build_spec(
            vars=[("alert_active", "float"), ("d_min", "float")],
            stl=self._specs["phi5"]["stl"].format(**self._specs["phi5"]["params"]),
        )
        self._rtamt_t = 0

    @staticmethod
    def _build_spec(
        vars: list[tuple[str, str]], stl: str
    ) -> Optional[rtamt.StlDiscreteTimeSpecification]:
        try:
            spec = rtamt.StlDiscreteTimeSpecification()
            for name, typ in vars:
                spec.declare_var(name, typ)
            spec.spec = stl
            spec.parse()
            return spec
        except Exception as exc:
            import logging
            logging.error("RTAMT parse failed (%s): %s", stl, exc)
            return None

    # ── Intervention helpers ─────────────────────────────────────────────────

    def _intervention_level(self, rho1: float, rho2: float, rho3: float,
                            v_closing: float, d_min: float) -> int:
        # Convergence gate: the monitored hazard is a pedestrian and vehicle
        # CLOSING on each other (relative velocity), NOT absolute vehicle speed.
        # This fires whether the car approaches the pedestrian or vice-versa, and
        # (per spec) does NOT fire for a pedestrian milling AROUND a static/parked
        # car — that motion is tangential so v_closing ≈ 0. Accepted limitation:
        # a truly static car that begins moving within the proximity band is only
        # caught once the motion becomes measurable.
        p = self._specs["phi1"]["params"]
        proximity    = p.get("proximity_emergency", 1.8)   # m — hard danger band
        closing_gate = p.get("closing_gate", 0.5)          # m/s radial closing
        converging   = v_closing >= closing_gate
        warn_rho  = self._intervention_cfg["warning"]["rho_min"]
        aware_rho = self._intervention_cfg["awareness"]["rho_min"]

        if rho1 <= 0.0:                                     # close band: d_min < clearance_critical
            if d_min < proximity and converging:
                raw = 3                                     # EMERGENCY: close AND converging
            elif rho3 <= 0.0 or rho2 <= 0.0:
                raw = 2                                     # converging/predictive, not yet <proximity
            else:
                raw = 1                                     # close but static (parked-car proximity)
        elif rho2 <= 0.0 or rho3 <= 0.0:
            raw = 2                                         # predictive / speed-proximity warning at range
        elif rho1 <= warn_rho:
            raw = 2
        elif rho1 <= aware_rho:
            raw = 1
        else:
            raw = 0
        return raw

    def _scale_factor(self, d_min: float) -> float:
        g = self._graduated
        if d_min >= g["proportional_zone_start"]:
            return 1.0
        clearance = self._specs["phi1"]["params"]["clearance_critical"]
        if d_min <= clearance:
            return 0.0
        return (d_min - clearance) / (g["proportional_zone_start"] - clearance)
