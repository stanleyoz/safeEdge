"""
SafeEdge — main edge orchestration loop.

Runs at camera FPS (~15Hz).  Safety-critical path (detect → track → signal →
STL → intervene) is synchronous.  Cloud calls are async and never block the
safety loop.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

from edge.camera.video_source import VideoSource, MockCamera
from edge.detection.detector import ObjectDetector
from edge.detection.tracker import ObjectTracker
from edge.detection.signal_extractor import SignalExtractor, mock_homography
from edge.safety.stl_monitor import STLMonitor, SignalFrame
from edge.safety.trajectory import predict_min_distance
from edge.safety.intervention import InterventionEngine, InterventionEvent
from edge.local_ai.qwen_local import LocalQwenInterpreter

ROOT = Path(__file__).parent.parent
CFG_CAMERA = ROOT / "config" / "camera_config.yaml"
CFG_STL    = ROOT / "config" / "stl_specs.yaml"
CFG_QWEN   = ROOT / "config" / "qwen_config.yaml"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("safeEdge.main")


class SafeEdge:
    def __init__(self, mock: bool = False):
        self._running = False

        with open(CFG_CAMERA) as f:
            cam_cfg = yaml.safe_load(f)
        with open(CFG_QWEN) as f:
            qwen_cfg = yaml.safe_load(f)

        use_mock = mock or cam_cfg.get("mock", {}).get("enabled", False)
        fps = cam_cfg.get("target_fps", 15.0)

        if use_mock:
            self._camera = MockCamera(fps=fps)
            homography = mock_homography()
            logger.info("Camera: Mock (synthetic scenario)")
        else:
            source = cam_cfg["source"]
            self._camera = VideoSource(source, target_fps=fps)
            hfile = cam_cfg.get("homography_file")
            if hfile and Path(hfile).exists():
                homography = np.load(hfile)
                logger.info("Camera: %s | homography loaded from %s", source, hfile)
            else:
                raise RuntimeError(
                    f"Homography file not found: {hfile}\n"
                    "Run tools/calibrate_homography.py first."
                )

        self._fps       = fps
        self._detector  = ObjectDetector(device="cuda")
        self._tracker   = ObjectTracker()
        self._extractor = SignalExtractor(homography, fps=fps)
        self._stl       = STLMonitor(CFG_STL)
        self._engine    = InterventionEngine(
            upgrade_hold=3,
            downgrade_hold=15,
            on_event=self._on_intervention_event,
        )
        self._local_ai = LocalQwenInterpreter(
            model=qwen_cfg["local"]["model"],
            base_url=qwen_cfg["local"]["base_url"],
            timeout_s=qwen_cfg["local"]["timeout_s"],
        )

        self._cloud_enabled = os.environ.get("CLOUD_REPORTING_ENABLED", "true").lower() == "true"
        if self._cloud_enabled:
            from cloud.policy_manager   import PolicyManager
            from cloud.incident_reporter import IncidentReporter
            from cloud.risk_forecaster   import RiskForecaster
            self._policy_mgr   = PolicyManager()
            self._incident_rpt = IncidentReporter(
                location=os.environ.get("LOCATION_LABEL", "Car Park")
            )
            self._risk_fcst    = RiskForecaster()
        else:
            self._policy_mgr   = None
            self._incident_rpt = None
            self._risk_fcst    = None

        self._event_log: list[dict] = []
        self._last_policy_update = 0.0
        self._frame_t = 0
        self._last_frame_bgr: np.ndarray | None = None

        self._ws_publish = None   # set by dashboard.app

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        logger.info("SafeEdge loop starting at %.0f fps", self._fps)

        while self._running:
            t0 = time.monotonic()

            frame = self._camera.read()
            if frame is None:
                continue

            self._last_frame_bgr = frame.color_bgr

            raw_dets = self._detector.detect(frame.color_bgr)
            tracked  = self._tracker.update(raw_dets, frame.color_bgr.shape[:2])
            bundle   = self._extractor.extract(tracked)

            d_pred = predict_min_distance(bundle.pedestrians, bundle.vehicles, horizon_s=4.0)

            sig_frame = SignalFrame(
                t=self._frame_t,
                d_min=bundle.d_min,
                v_veh_max=bundle.v_veh_max,
                d_pred=d_pred,
                alert_active=self._engine.alert_active,
            )
            state = self._stl.update(sig_frame)
            event = self._engine.process(state)

            if event is not None:
                interp = self._local_ai.interpret(state)
                logger.info("[L%d] %s | %s", state.intervention_level, event.message, interp)

            if self._ws_publish:
                self._ws_publish(state, frame.color_bgr, event)

            if self._cloud_enabled and time.monotonic() - self._last_policy_update > 300:
                asyncio.create_task(self._cloud_policy_update())
                self._last_policy_update = time.monotonic()

            self._frame_t += 1

            elapsed = time.monotonic() - t0
            sleep_s = max(0.0, (1.0 / self._fps) - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)

        self._camera.stop()
        logger.info("SafeEdge stopped")

    # ── Event callbacks ──────────────────────────────────────────────────────

    def _on_intervention_event(self, event: InterventionEvent) -> None:
        self._event_log.append({
            "timestamp": event.timestamp,
            "level": event.new_level,
            "d_min": event.d_min,
            "v_veh_max": event.v_veh_max,
            "d_pred": event.d_pred,
            "rho_min": event.rho_min,
            "message": event.message,
        })
        if self._incident_rpt and event.new_level >= 2:
            asyncio.create_task(
                self._async_incident_report(event, self._last_frame_bgr)
            )

    async def _async_incident_report(
        self, event: InterventionEvent, frame_bgr: np.ndarray | None
    ) -> None:
        report = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._incident_rpt.report(event, frame_bgr)
        )
        if report:
            logger.info("INCIDENT REPORT:\n%s", report)

    async def _cloud_policy_update(self) -> None:
        if self._policy_mgr is None or len(self._event_log) < 3:
            return
        recent = self._event_log[-100:]
        patch = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._policy_mgr.request_update(
                {"events_last_window": len(recent)},
                {
                    "emergency": sum(1 for e in recent if e["level"] == 3),
                    "warning":   sum(1 for e in recent if e["level"] == 2),
                },
                {sid: spec["params"] for sid, spec in self._stl._specs.items()},
            ),
        )
        if patch:
            self._stl.apply_cloud_params(patch)
            logger.info("STL params hot-swapped from Qwen Cloud policy update")

    def _shutdown(self, *_) -> None:
        logger.info("Shutdown signal received")
        self._running = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SafeEdge car park safety monitor")
    parser.add_argument("--mock", action="store_true",
                        help="Synthetic scenario — no camera or homography needed")
    args = parser.parse_args()
    SafeEdge(mock=args.mock).run()
