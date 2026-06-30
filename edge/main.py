"""
SafeEdge — main edge orchestration loop.

Runs at camera FPS (~15Hz).  Safety-critical path (detect → track → signal →
STL → intervene) is synchronous.  Cloud calls are async and never block the
safety loop.
"""
from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

from edge.camera.video_source import VideoSource, MockCamera, RealSenseSource
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


SOURCE_CHOICES = ("mock", "webcam", "realsense", "file")


class SafeEdge:
    def __init__(self, source: str = "mock", file_path: str | None = None,
                 model: str = "yolov8n.pt", conf: float = 0.4):
        self._running = False

        with open(CFG_CAMERA) as f:
            cam_cfg = yaml.safe_load(f)
        with open(CFG_QWEN) as f:
            qwen_cfg = yaml.safe_load(f)

        fps = cam_cfg.get("target_fps", 15.0)
        intrinsics = None   # set only for realsense

        if source == "mock":
            self._camera = MockCamera(fps=fps)
            homography = mock_homography()
            logger.info("Camera: Mock (synthetic scenario)")

        elif source == "realsense":
            self._camera = RealSenseSource()
            homography = mock_homography()   # fallback; not used when depth_m present
            intrinsics = (
                self._camera.fx, self._camera.fy,
                self._camera.cx, self._camera.cy,
            )
            logger.info(
                "Camera: RealSense D455 | fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                *intrinsics,
            )

        elif source == "file":
            if not file_path:
                raise ValueError("--source file requires --file-path")
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Video file not found: {file_path}")
            self._camera = VideoSource(file_path, target_fps=fps)
            homography = mock_homography()   # no calibration needed for offline analysis
            logger.info("Camera: file %s | homography: mock (offline mode)", file_path)

        else:  # webcam
            cv_source = cam_cfg.get("source", 0)
            self._camera = VideoSource(cv_source, target_fps=fps)
            hfile = cam_cfg.get("homography_file")
            if hfile and Path(hfile).exists():
                homography = np.load(hfile)
                logger.info("Camera: webcam %s | homography loaded from %s", cv_source, hfile)
            else:
                raise RuntimeError(
                    f"Homography file not found: {hfile}\n"
                    "Run tools/calibrate_homography.py first."
                )

        self._fps       = fps
        self._detector  = ObjectDetector(model_path=model, confidence=conf, device="auto")
        self._tracker   = ObjectTracker()
        self._extractor = SignalExtractor(homography, fps=fps, intrinsics=intrinsics)
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

        # Cloud backend (Alibaba Cloud) — enabled iff SAFEEDGE_CLOUD_URL is set.
        # The Qwen skills now live in the cloud backend; the edge just POSTs to
        # it over HTTP. With no URL set, the edge runs fully standalone/offline.
        cloud_url = os.environ.get("SAFEEDGE_CLOUD_URL", "").strip()
        self._cloud_enabled = bool(cloud_url)
        if self._cloud_enabled:
            from edge.cloud_client import CloudClient
            self._cloud = CloudClient(base_url=cloud_url)
            logger.info("Cloud backend enabled → %s", cloud_url)
        else:
            self._cloud = None
            logger.info("Cloud backend disabled (SAFEEDGE_CLOUD_URL unset) — edge standalone")

        self._event_log: list[dict] = []
        self._last_policy_update = 0.0
        self._frame_t = 0
        self._last_frame_bgr: np.ndarray | None = None
        self._last_interp: str = ""
        self._fps_measured: float = 0.0

        self._ws_publish = None   # set by dashboard.app

        # Preview window — disabled if no display available
        self._preview = os.environ.get("DISPLAY") is not None or \
                        os.environ.get("WAYLAND_DISPLAY") is not None
        if self._preview:
            cv2.namedWindow("SafeEdge", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("SafeEdge", 960, 540)
            logger.info("Preview window enabled")

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
            bundle   = self._extractor.extract(tracked, depth_m=frame.depth_m)

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
                self._last_interp = self._local_ai.interpret(state)
                logger.info("[L%d] %s | %s", state.intervention_level, event.message, self._last_interp)

            if self._preview:
                self._show_preview(frame, tracked, bundle, state, frame.depth_m)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._running = False

            if self._ws_publish:
                self._ws_publish(state, frame.color_bgr, event)

            if self._cloud is not None:
                # Throttled live-state push for the cloud dashboard (non-blocking)
                self._cloud.push_state(state, frame.color_bgr)
                # Periodic Qwen policy re-evaluation (non-blocking; applies patch on reply)
                if time.monotonic() - self._last_policy_update > 300:
                    self._last_policy_update = time.monotonic()
                    self._request_cloud_policy()

            self._frame_t += 1

            elapsed = time.monotonic() - t0
            self._fps_measured = 1.0 / elapsed if elapsed > 0 else self._fps
            sleep_s = max(0.0, (1.0 / self._fps) - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)

        self._camera.stop()
        if self._cloud is not None:
            self._cloud.close()
        if self._preview:
            cv2.destroyAllWindows()
        logger.info("SafeEdge stopped")

    # ── Preview window ───────────────────────────────────────────────────────

    def _show_preview(self, frame, tracked, bundle, state, depth_m) -> None:
        canvas = frame.color_bgr.copy()
        h, w = canvas.shape[:2]

        # Intervention level → banner colour
        LEVEL_COLOUR = {
            0: (40, 180, 40),    # green  — safe
            1: (0, 165, 255),    # orange — awareness
            2: (0, 60, 255),     # red    — warning
            3: (0, 0, 200),      # dark red — emergency
        }
        LEVEL_LABEL = {0: "SAFE", 1: "AWARENESS", 2: "WARNING", 3: "EMERGENCY"}
        lvl   = state.intervention_level
        colour = LEVEL_COLOUR.get(lvl, (40, 180, 40))

        # Top banner
        cv2.rectangle(canvas, (0, 0), (w, 36), colour, -1)
        cv2.putText(canvas, f"SafeEdge  {LEVEL_LABEL.get(lvl,'?')}  |  "
                            f"fps:{self._fps_measured:.0f}  frame:{self._frame_t}",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Detection boxes
        for det in tracked:
            x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
            box_col = (0, 220, 60) if det.label == "person" else (220, 80, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_col, 2)
            cv2.putText(canvas, f"{det.label} #{det.track_id}",
                        (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_PLAIN, 1.1, box_col, 1)

        # Distance line between closest ped-veh pair (image space approximation)
        if bundle.pedestrians and bundle.vehicles:
            # Find the closest pair by world distance and draw line between bbox centres
            best_d, best_p, best_v = float("inf"), None, None
            for p in bundle.pedestrians:
                for v in bundle.vehicles:
                    d = float(np.linalg.norm(p.xyz[[0, 2]] - v.xyz[[0, 2]]))
                    if d < best_d:
                        best_d, best_p, best_v = d, p, v

            # Find matching detections for bbox centres
            def _centre(track_id):
                for det in tracked:
                    if det.track_id == track_id:
                        x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
                        return ((x1 + x2) // 2, (y1 + y2) // 2)
                return None

            pc = _centre(best_p.track_id) if best_p else None
            vc = _centre(best_v.track_id) if best_v else None
            if pc and vc:
                line_col = (0, 255, 0) if best_d > 3.0 else ((0, 165, 255) if best_d > 1.5 else (0, 0, 255))
                cv2.line(canvas, pc, vc, line_col, 2)
                mid = ((pc[0] + vc[0]) // 2, (pc[1] + vc[1]) // 2 - 8)
                cv2.putText(canvas, f"{best_d:.1f}m", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_col, 2)

        # Bottom HUD
        rho3_str = f"{state.rho3:.2f}" if state.rho3 is not None else "n/a"
        hud = (f"d_min:{bundle.d_min:.2f}m  v_max:{bundle.v_veh_max:.2f}m/s  "
               f"rho1:{state.rho1:.2f}  rho2:{state.rho2:.2f}  rho3:{rho3_str}")
        cv2.rectangle(canvas, (0, h - 52), (w, h), (20, 20, 20), -1)
        cv2.putText(canvas, hud, (8, h - 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1)
        if self._last_interp:
            cv2.putText(canvas, f"AI: {self._last_interp[:80]}",
                        (8, h - 10), cv2.FONT_HERSHEY_PLAIN, 0.95, (100, 220, 255), 1)

        # Depth thumbnail (top-right corner, RealSense only)
        if depth_m is not None:
            thumb_w, thumb_h = 240, 135
            d_vis = np.clip(depth_m / 6.0, 0, 1)
            d_colour = cv2.applyColorMap((d_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            d_thumb  = cv2.resize(d_colour, (thumb_w, thumb_h))
            canvas[40:40 + thumb_h, w - thumb_w:w] = d_thumb
            cv2.putText(canvas, "depth (0-6m)", (w - thumb_w + 4, 54),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)

        cv2.imshow("SafeEdge", canvas)

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
        # Ship the event to the cloud backend — WARNING+ events trigger a
        # Qwen-VL incident report there. Non-blocking; safe if cloud is down.
        if self._cloud is not None:
            self._cloud.push_event(event, self._last_frame_bgr)

    def _request_cloud_policy(self) -> None:
        if self._cloud is None or len(self._event_log) < 3:
            return
        recent = self._event_log[-100:]
        self._cloud.evaluate_policy(
            rho_summary={"events_last_window": len(recent)},
            event_counts={
                "emergency": sum(1 for e in recent if e["level"] == 3),
                "warning":   sum(1 for e in recent if e["level"] == 2),
            },
            current_params={sid: spec["params"] for sid, spec in self._stl._specs.items()},
            context=os.environ.get("OPERATOR_CONTEXT", ""),
            on_patch=self._stl.apply_cloud_params,
        )

    def _shutdown(self, *_) -> None:
        logger.info("Shutdown signal received")
        self._running = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SafeEdge car park safety monitor")
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default="mock",
        help="Camera source: mock | webcam | realsense | file",
    )
    parser.add_argument(
        "--file-path", dest="file_path", default=None,
        help="Path to video file (required when --source file)",
    )
    parser.add_argument("--mock", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", default="yolov8s.pt",
                        help="YOLO model file (default: yolov8s.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    args = parser.parse_args()
    source = "mock" if args.mock else args.source
    SafeEdge(source=source, file_path=args.file_path,
             model=args.model, conf=args.conf).run()
