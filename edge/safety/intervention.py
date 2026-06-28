"""
Intervention engine: translates STL robustness values into concrete actions.

Implements hysteresis (from stl_specs.yaml) to prevent alert flapping when
robustness hovers near a threshold.  Emits an InterventionEvent whenever the
level changes — this is what the dashboard and cloud reporter consume.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from edge.safety.stl_monitor import SafetyState


@dataclass
class InterventionEvent:
    timestamp: float
    prev_level: int
    new_level: int
    rho_min: float
    d_min: float
    v_veh_max: float
    d_pred: float
    message: str


LEVEL_LABELS = {0: "SAFE", 1: "AWARENESS", 2: "WARNING", 3: "EMERGENCY"}
LEVEL_COLOURS = {0: "#00c853", 1: "#ffd600", 2: "#ff6d00", 3: "#d50000"}


class InterventionEngine:
    def __init__(
        self,
        upgrade_hold: int = 3,
        downgrade_hold: int = 15,
        on_event: Optional[Callable[[InterventionEvent], None]] = None,
    ):
        self._upgrade_hold = upgrade_hold
        self._downgrade_hold = downgrade_hold
        self._on_event = on_event

        self._current_level = 0
        self._candidate_level = 0
        self._candidate_frames = 0
        self._alert_active = False

    @property
    def alert_active(self) -> float:
        return 1.0 if self._alert_active else 0.0

    def process(self, state: SafetyState) -> Optional[InterventionEvent]:
        new_raw = state.intervention_level
        event: Optional[InterventionEvent] = None

        if new_raw != self._candidate_level:
            self._candidate_level = new_raw
            self._candidate_frames = 1
        else:
            self._candidate_frames += 1

        hold = (
            self._upgrade_hold
            if new_raw > self._current_level
            else self._downgrade_hold
        )

        if self._candidate_frames >= hold and new_raw != self._current_level:
            event = InterventionEvent(
                timestamp=time.time(),
                prev_level=self._current_level,
                new_level=new_raw,
                rho_min=state.rho_min,
                d_min=state.signals.d_min,
                v_veh_max=state.signals.v_veh_max,
                d_pred=state.signals.d_pred,
                message=self._message(new_raw, state),
            )
            self._current_level = new_raw
            self._alert_active = new_raw >= 2
            if self._on_event:
                self._on_event(event)

        return event

    @staticmethod
    def _message(level: int, state: SafetyState) -> str:
        s = state.signals
        if level == 3:
            return (
                f"EMERGENCY: pedestrian-vehicle distance {s.d_min:.2f}m — "
                f"below critical threshold. ρ1={state.rho1:.2f}"
            )
        if level == 2:
            if state.rho3 is not None and state.rho3 <= 0.0:
                return (
                    f"WARNING (predictive): trajectory predicts clearance "
                    f"{s.d_pred:.2f}m in ~4s. Current d={s.d_min:.2f}m"
                )
            return (
                f"WARNING: vehicle speed {s.v_veh_max:.1f}m/s with pedestrian "
                f"at {s.d_min:.2f}m. ρ2={state.rho2:.2f}"
            )
        if level == 1:
            return f"AWARENESS: pedestrian at {s.d_min:.2f}m. ρ1={state.rho1:.2f}"
        return f"SAFE: d_min={s.d_min:.2f}m ρ1={state.rho1:.2f}"
