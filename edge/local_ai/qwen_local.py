"""
Local Qwen2.5-3B via Ollama for real-time scene interpretation.

Hard timeout of 3s — this runs in the safety loop and MUST NOT block.
Falls back to a deterministic rule-based message if Ollama is unreachable.
"""
from __future__ import annotations

import logging
from typing import Optional

from edge.safety.stl_monitor import SafetyState


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You are a real-time safety monitor for a car park. You receive sensor readings
and STL robustness values. In ONE sentence (max 20 words), describe the current
safety situation and what action should be taken. Be direct, not alarming.
"""


class LocalQwenInterpreter:
    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434/v1", timeout_s: float = 3.0):
        self._model = model
        self._timeout = timeout_s
        self._client = self._init_client(base_url)

    def interpret(self, state: SafetyState, context: str = "") -> str:
        if self._client is None:
            return self._fallback(state)

        prompt = self._build_prompt(state, context)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=60,
                timeout=self._timeout,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.debug("Local Qwen unavailable: %s", exc)
            return self._fallback(state)

    # ── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(state: SafetyState, context: str) -> str:
        s = state.signals
        rho3_str = f"{state.rho3:.2f}" if state.rho3 is not None else "n/a"
        return (
            f"d_min={s.d_min:.2f}m  v_veh={s.v_veh_max:.2f}m/s  "
            f"d_pred={s.d_pred:.2f}m  "
            f"ρ1={state.rho1:.2f}  ρ2={state.rho2:.2f}  ρ3={rho3_str}  "
            f"level={state.intervention_level}  "
            f"{('context: ' + context) if context else ''}"
        )

    @staticmethod
    def _fallback(state: SafetyState) -> str:
        level = state.intervention_level
        s = state.signals
        if level == 3:
            return f"Emergency: {s.d_min:.1f}m clearance — stop all vehicles immediately."
        if level == 2:
            return f"Warning: vehicle {s.v_veh_max:.1f}m/s with pedestrian at {s.d_min:.1f}m."
        if level == 1:
            return f"Awareness: pedestrian at {s.d_min:.1f}m — monitor closely."
        return "Zone clear — all safety metrics within bounds."

    @staticmethod
    def _init_client(base_url: str):
        try:
            from openai import OpenAI
            return OpenAI(base_url=base_url, api_key="ollama")
        except ImportError:
            return None
