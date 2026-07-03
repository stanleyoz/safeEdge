"""
SafeEdge — main edge orchestration loop.

Runs at camera FPS (~15Hz).  Safety-critical path (detect → track → signal →
STL → intervene) is synchronous.  Cloud calls are async and never block the
safety loop.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

from edge.camera.video_source import VideoSource, MockCamera, RealSenseSource
from edge.detection.detector import ObjectDetector
from edge.detection.tracker import ObjectTracker
from edge.detection.aoi import AOI
from edge.detection.signal_extractor import SignalExtractor, mock_homography
from edge.safety.stl_monitor import STLMonitor, SignalFrame
from edge.safety.trajectory import predict_min_distance
from edge.safety.intervention import InterventionEngine, InterventionEvent
from edge.local_ai.qwen_local import LocalQwenInterpreter

ROOT = Path(__file__).parent.parent
# Config paths are env-overridable so offline tuning can point at temp variants
# (e.g. different crop_zoom / motion_gate) without touching the live configs.
CFG_CAMERA = Path(os.environ.get("SAFEEDGE_CFG_CAMERA", ROOT / "config" / "camera_config.yaml"))
CFG_STL    = Path(os.environ.get("SAFEEDGE_CFG_STL",    ROOT / "config" / "stl_specs.yaml"))
CFG_QWEN   = Path(os.environ.get("SAFEEDGE_CFG_QWEN",   ROOT / "config" / "qwen_config.yaml"))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("safeEdge.main")


SOURCE_CHOICES = ("mock", "webcam", "realsense", "file")


class RawRecorder:
    """Persist the RAW camera feed to a timestamped MP4 (one file per run), plus a
    sidecar JSON of the settings that produced it. Writes the un-annotated,
    un-cropped frame — exactly what the pipeline ingested — so scenario clips can
    be re-processed offline with different params/models for tuning.

    Robust by design: any failure disables recording and logs a warning; it must
    never crash the safety loop.
    """

    def __init__(self, out_dir: Path, fps: float, enabled: bool = True,
                 settings: dict | None = None):
        self.enabled  = enabled
        self._fps     = float(fps) if fps and fps > 0 else 15.0
        self._out_dir = Path(out_dir)
        self._settings = settings or {}
        self._writer  = None
        self._path    = None
        self._n       = 0

    def _open(self, frame) -> None:
        try:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._path = self._out_dir / f"safeedge_raw_{stamp}.mp4"
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(self._path), cv2.VideoWriter_fourcc(*"mp4v"),
                self._fps, (w, h),
            )
            if not writer.isOpened():
                raise RuntimeError("VideoWriter failed to open (codec/path?)")
            self._writer = writer
            meta = dict(self._settings)
            meta.update({"file": self._path.name, "started": stamp,
                         "fps": self._fps, "width": w, "height": h})
            with open(self._path.with_suffix(".meta.json"), "w") as f:
                json.dump(meta, f, indent=2, default=str)
            logger.info("Raw recorder → %s (%dx%d @ %.0f fps)",
                        self._path, w, h, self._fps)
        except Exception as e:                       # noqa: BLE001 — never crash the loop
            logger.warning("Raw recorder disabled (open failed): %s", e)
            self.enabled = False

    def write(self, frame_bgr) -> None:
        if not self.enabled or frame_bgr is None:
            return
        if self._writer is None:
            self._open(frame_bgr)
            if self._writer is None:
                return
        try:
            self._writer.write(frame_bgr)
            self._n += 1
        except Exception as e:                       # noqa: BLE001
            logger.warning("Raw recorder write failed, disabling: %s", e)
            self.enabled = False

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            dur = self._n / self._fps if self._fps else 0.0
            logger.info("Raw recording saved: %s (%d frames, %.1fs)",
                        self._path, self._n, dur)
            self._writer = None


class SafeEdge:
    def __init__(self, source: str = "mock", file_path: str | None = None,
                 model: str = "yolov8n.pt", conf: float = 0.4,
                 record: bool | None = None, capture_dir: str | None = None,
                 fps_override: float | None = None, realtime: bool = False):
        self._running = False
        self._offline = (source == "file")   # replay a saved clip: stop at EOF
        # pace to real time when live, or when replaying for a dashboard recording
        # (--realtime); pure metric-tuning replays run flat-out (fast).
        self._pace = (not self._offline) or realtime

        with open(CFG_CAMERA) as f:
            cam_cfg = yaml.safe_load(f)
        with open(CFG_QWEN) as f:
            qwen_cfg = yaml.safe_load(f)

        # fps drives velocity estimation (m/s = pixels/frame × fps). For faithful
        # offline replay, pass the TRUE capture fps of the clip (its .meta.json /
        # frames÷wall-duration), not the nominal 15.
        fps = fps_override if fps_override else cam_cfg.get("target_fps", 15.0)
        intrinsics = None   # set only for realsense

        # Area-of-Interest (crop-zoom detection + polygon filter). Built lazily
        # on the first frame once the real frame size is known.
        _aoi = cam_cfg.get("aoi") or {}
        self._aoi_cfg = _aoi if _aoi.get("enabled") else None
        self._aoi = None

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
            # Use the REAL homography so offline distances/velocities match the live
            # run that produced the clip (same camera pose). Falls back to mock only
            # if the calibration file is missing.
            hfile = cam_cfg.get("homography_file")
            if hfile and Path(hfile).exists():
                homography = np.load(hfile)
                logger.info("Camera: file %s | homography loaded from %s (faithful replay)",
                            file_path, hfile)
            else:
                homography = mock_homography()
                logger.warning("Camera: file %s | NO homography (%s) — mock projection; "
                               "distances NOT metric", file_path, hfile)

        else:  # webcam
            cv_source = cam_cfg.get("source", 0)
            self._camera = VideoSource(cv_source, target_fps=fps)
            hfile = cam_cfg.get("homography_file")
            if hfile and Path(hfile).exists():
                homography = np.load(hfile)
                logger.info("Camera: webcam %s | homography loaded from %s", cv_source, hfile)
            else:
                # Uncalibrated: run with a mock homography so frames still stream
                # (metric positions are meaningless until calibrated). This lets
                # the browser calibrator pull a live frame to define the real one.
                homography = mock_homography()
                logger.warning("Camera: webcam %s | NO homography (%s) — running "
                               "UNCALIBRATED with mock projection; distances not metric",
                               cv_source, hfile)

        self._fps       = fps
        self._detector  = ObjectDetector(model_path=model, confidence=conf, device="auto")
        # Tracker tuned to the REAL fps so lost tracks survive detection gaps
        # (~4 s persistence) — the fix for ID churn that kills φ3 predictive.
        # Env-overridable for offline tuning sweeps.
        _track_buf = int(os.environ.get("SAFEEDGE_TRACK_BUFFER", "0") or 0) \
                     or max(30, int(round(fps * 4)))
        _track_act = float(os.environ.get("SAFEEDGE_TRACK_ACT", "0.4"))
        self._tracker   = ObjectTracker(
            lost_track_buffer=_track_buf,
            frame_rate=max(1, int(round(fps))),
            track_activation_threshold=_track_act,
        )
        logger.info("Tracker: frame_rate=%d lost_track_buffer=%d act_thresh=%.2f",
                    max(1, int(round(fps))), _track_buf, _track_act)
        self._extractor = SignalExtractor(homography, fps=fps, intrinsics=intrinsics)
        self._stl       = STLMonitor(CFG_STL)
        # downgrade_hold acts as a sticky-hold: a level must persist lower for N
        # frames before de-escalating, so a brief detection dropout (pedestrian
        # momentarily lost → d_min sentinel) doesn't instantly reset an active
        # EMERGENCY to SAFE. Env-tunable for offline sweeps.
        _down_hold = int(os.environ.get("SAFEEDGE_DOWNGRADE_HOLD", "0") or 0) \
                     or max(15, int(round(fps * 3)))
        self._engine    = InterventionEngine(
            upgrade_hold=3,
            downgrade_hold=_down_hold,
            on_event=self._on_intervention_event,
        )
        logger.info("Intervention: upgrade_hold=3 downgrade_hold=%d (%.1fs @%.0ffps)",
                    _down_hold, _down_hold / max(fps, 1), fps)
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
        # Only ship intervention EVENTS (which trigger an inline Qwen incident
        # report on the backend) at/above this level. Default 2 (WARNING+) keeps
        # full reporting; raise to 3 to post only EMERGENCY events, which keeps
        # the serverless backend responsive during dense bursts (a flood of L2
        # Qwen calls can saturate the FC instance and stall the dashboard). Live
        # state (badge/clock/feed) is pushed separately and is unaffected.
        self._cloud_min_event_level = int(
            os.environ.get("SAFEEDGE_CLOUD_MIN_EVENT_LEVEL", "2"))
        if self._cloud_enabled:
            from edge.cloud_client import CloudClient
            self._cloud = CloudClient(base_url=cloud_url)
            logger.info("Cloud backend enabled → %s (min event level %d)",
                        cloud_url, self._cloud_min_event_level)
        else:
            self._cloud = None
            logger.info("Cloud backend disabled (SAFEEDGE_CLOUD_URL unset) — edge standalone")

        # Raw-feed recorder — persist the un-annotated camera feed for offline
        # tuning. On by default for LIVE cameras; skipped for mock (synthetic)
        # and file (already a clip). Env overrides: SAFEEDGE_RECORD=0 to disable,
        # SAFEEDGE_CAPTURE_DIR to relocate.
        if record is None:
            _default = "1" if source in ("webcam", "realsense") else "0"
            record = os.environ.get("SAFEEDGE_RECORD", _default).strip() \
                     not in ("", "0", "false", "False", "no")
        cap_dir = capture_dir or os.environ.get("SAFEEDGE_CAPTURE_DIR") \
                  or str(ROOT / "data" / "captures")
        self._recorder = RawRecorder(
            cap_dir, fps=fps, enabled=record,
            settings={
                "source": source, "model": model, "conf": conf,
                "aoi": self._aoi_cfg,
                "homography_file": cam_cfg.get("homography_file"),
                "target_fps": fps,
            },
        )
        if record:
            logger.info("Raw feed recording ENABLED → %s", cap_dir)

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
                if self._offline:      # end of clip → finish the replay
                    break
                continue

            self._last_frame_bgr = frame.color_bgr

            # Persist the RAW frame (pre-crop, pre-annotation) for offline tuning.
            self._recorder.write(frame.color_bgr)

            # Lazy-build the AOI once we know the real frame size
            if self._aoi_cfg and self._aoi is None:
                h, w = frame.color_bgr.shape[:2]
                self._aoi = AOI(
                    self._aoi_cfg["polygon"], w, h,
                    pad=self._aoi_cfg.get("pad", 0.08),
                    crop_zoom=self._aoi_cfg.get("crop_zoom", True),
                )
                logger.info("AOI active — crop_zoom=%s, %d-point polygon",
                            self._aoi.crop_zoom, len(self._aoi_cfg["polygon"]))

            # (B) crop-zoom detection: detect on the AOI crop, then remap boxes
            # back to full-frame coords so depth/intrinsics stay correct.
            if self._aoi and self._aoi.crop_zoom:
                raw_dets = self._detector.detect(self._aoi.crop(frame.color_bgr))
                raw_dets = self._aoi.remap_dets(raw_dets)
            else:
                raw_dets = self._detector.detect(frame.color_bgr)

            tracked  = self._tracker.update(raw_dets, frame.color_bgr.shape[:2])

            # (A) polygon filter: keep only tracks whose foot-point is in the AOI
            if self._aoi:
                tracked = [t for t in tracked if self._aoi.contains_foot(t.bbox_xyxy)]

            bundle   = self._extractor.extract(tracked, depth_m=frame.depth_m)

            d_pred = predict_min_distance(bundle.pedestrians, bundle.vehicles, horizon_s=4.0)

            sig_frame = SignalFrame(
                t=self._frame_t,
                d_min=bundle.d_min,
                v_veh_max=bundle.v_veh_max,
                d_pred=d_pred,
                alert_active=self._engine.alert_active,
                v_closing=bundle.v_closing,
            )
            state = self._stl.update(sig_frame)
            event = self._engine.process(state)

            if event is not None:
                self._last_interp = self._local_ai.interpret(state)
                logger.info("[L%d] d_min=%.2f v_veh_max=%.2f v_closing=%.2f | %s | %s",
                            state.intervention_level, bundle.d_min, bundle.v_veh_max,
                            bundle.v_closing, event.message, self._last_interp)

            if self._preview:
                self._show_preview(frame, tracked, bundle, state, frame.depth_m)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._running = False

            if self._ws_publish:
                self._ws_publish(state, frame.color_bgr, event)

            if self._cloud is not None:
                # Throttled live-state push for the cloud dashboard (non-blocking).
                # Pass tracked detections so the posted frame shows thin yellow
                # boxes — a lightweight "live" cue (drawn only on posted frames).
                self._cloud.push_state(
                    state, frame.color_bgr, tracked,
                    aoi_poly=(self._aoi.poly if self._aoi else None),
                )
                # Periodic Qwen policy re-evaluation (non-blocking; applies patch on reply)
                if time.monotonic() - self._last_policy_update > 300:
                    self._last_policy_update = time.monotonic()
                    self._request_cloud_policy()

            self._frame_t += 1

            elapsed = time.monotonic() - t0
            self._fps_measured = 1.0 / elapsed if elapsed > 0 else self._fps
            if self._pace:             # real-time (live or dashboard replay); else flat-out
                sleep_s = max(0.0, (1.0 / self._fps) - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)

        self._camera.stop()
        self._recorder.close()
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

        if self._aoi:
            self._aoi.draw(canvas)

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
        # Gate by level to avoid flooding the serverless backend (see
        # _cloud_min_event_level); live state is still pushed every frame.
        if self._cloud is not None and event.new_level >= self._cloud_min_event_level:
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
    parser.add_argument("--no-record", action="store_true",
                        help="disable saving the raw camera feed (on by default for live cameras)")
    parser.add_argument("--capture-dir", dest="capture_dir", default=None,
                        help="directory for raw recordings (default: data/captures)")
    parser.add_argument("--fps", type=float, default=None,
                        help="override capture fps (velocity math); for faithful clip replay "
                             "pass the clip's TRUE fps (frames÷wall-duration)")
    parser.add_argument("--realtime", action="store_true",
                        help="pace a file replay to real time (for dashboard recording); "
                             "default file replay runs flat-out for fast tuning")
    args = parser.parse_args()
    source = "mock" if args.mock else args.source
    SafeEdge(source=source, file_path=args.file_path,
             model=args.model, conf=args.conf,
             record=(False if args.no_record else None),
             capture_dir=args.capture_dir, fps_override=args.fps,
             realtime=args.realtime).run()
