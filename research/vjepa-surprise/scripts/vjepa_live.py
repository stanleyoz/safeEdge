"""
Live V-JEPA 2.1 surprise for the monitor page, on this workstation's GPU.

Reads the Jetson's dashboard feed (the same WebSocket the browser viewers
use, so the Jetson is untouched), scores it with the paper's offline setup
and serves the scores on a WebSocket of its own for docs/monitor/ to show
next to the Jetson's STL level.

  Jetson ws://<jetson>:8080/ws ──> sample 4 fps, strip the drawn boxes
      ──> 16-frame window every 0.5 s ──> ViT-g predictor surprise map
      ──> s = p95 over patches, ŝ = s / τ ──> ws://127.0.0.1:8092 (Funnel)

Same window as tools/vjepa_surprise.py (16 frames at 4 fps, last 1 s scored,
AOI crop, p95) and the same τ: the split-conformal threshold from the July
calibration run (--summary), not re-fitted on live footage. A window alerts
when ŝ > 1 for --hold consecutive windows, as the page's demo default.

The feed's frames carry the Jetson's thin yellow boxes, labels and AOI
outline and are JPEG (quality 70); the overlay is masked by colour and
inpainted before scoring, so the model does not see boxes move. Raw slot
recordings (harvest_jetson_slots.py) stay the clean source for the paper.

Every window is also logged under --log-dir/<YYYYMMDD>.csv (with the highest
STL level the Jetson reported during the window) and its surprise map
appended to <YYYYMMDD>.maps.f16 ([24, 24] float16 per row), so the live run
doubles as nuisance-alarm data.

  python research/vjepa-surprise/scripts/vjepa_live.py \\
      --jetson ws://100.120.250.94:8080/ws --port 8092

Messages sent to each client (JSON):
  hello   model, τ, crop box, frame size, grid, hold, the last --history-s
          of windows (without maps)
  window  t (window end, Jetson clock, Unix s), s, shat, alert, stl (max STL
          level in the window), ms (GPU time), lag (s from window end to
          send), map (base64 uint8 [24, 24]: each patch's rise above its
          median over the last --baseline-s of live windows, so the map
          shows what is new now rather than how today differs from July;
          the July calm baseline until enough live windows exist; display
          only, the score and τ do not use it)
  status  upstream "up" | "down"
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import datetime as dt
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.vjepa_surprise import (IMG_SIZE, PATCH, TUBELET, Scorer, load_vjepa,  # noqa: E402
                                  preprocess)

log = logging.getLogger("vjepa_live")
DEFAULT_RUN = ROOT / "research/vjepa-surprise/results_calib-2026-11-21_ttc/vitg_predictor"


def strip_overlay(bgr: np.ndarray, crop_box) -> np.ndarray:
    """Inpaint the Jetson's yellow boxes, labels and AOI outline inside the
    crop: bright pixels with red and green both high and blue well below them
    (the grass and the red hydrant fail one of these), dilated for JPEG bleed."""
    x0, y0, x1, y1 = crop_box
    roi = bgr[y0:y1, x0:x1]
    b, g, r = (roi[..., i].astype(np.int16) for i in range(3))
    mask = ((r > 120) & (g > 120) & (np.minimum(r, g) - b > 45)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    out = bgr.copy()
    if mask.any():
        out[y0:y1, x0:x1] = cv2.inpaint(roi, mask, 3, cv2.INPAINT_TELEA)
    return out


class Calibration:
    """τ and the calm per-patch baseline used to colour maps, from a run of
    tools/vjepa_surprise.py (summary.json, surprise.csv, maps.npy)."""

    def __init__(self, run: Path):
        summ = json.loads((run / "summary.json").read_text())
        self.tau = float(summ["calibration"]["tau"])
        self.alpha = float(summ["calibration"]["alpha"])
        self.crop_box = tuple(summ["crop_box"])
        self.window = summ["window"]
        rows = list(csv.DictReader(open(run / "surprise.csv")))
        maps = np.load(run / "maps.npy").astype(np.float32)
        calib = [i for i, r in enumerate(rows) if r["role"] == "calib"] or list(range(len(rows)))
        self.base = np.median(maps[calib], axis=0)
        floor = float(np.median([float(rows[i]["s_p95"]) for i in calib]))
        self.scale = (self.tau - floor) if self.tau > floor else 1.0

    def colour(self, smap: np.ndarray, base: np.ndarray | None = None) -> str:
        base = self.base if base is None else base
        q = (np.clip(np.maximum(smap - base, 0) / (2 * self.scale), 0, 1) * 255).astype(np.uint8)
        return base64.b64encode(q.tobytes()).decode()


class DayLog:
    """<dir>/<YYYYMMDD>.csv plus the maps, rolled over at local midnight."""
    FIELDS = ["t_end_unix", "s_p95", "s_mean", "shat", "alert", "stl_max", "ms", "lag_s"]

    def __init__(self, root: Path):
        self.root, self.day, self.fh, self.mh = root, None, None, None
        root.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict, smap: np.ndarray) -> None:
        day = dt.datetime.fromtimestamp(row["t_end_unix"]).strftime("%Y%m%d")
        if day != self.day:
            self.close()
            path = self.root / f"{day}.csv"
            new = not path.exists()
            self.fh = open(path, "a", newline="")
            self.w = csv.DictWriter(self.fh, self.FIELDS)
            if new:
                self.w.writeheader()
            self.mh = open(self.root / f"{day}.maps.f16", "ab")
            self.day = day
        self.w.writerow(row)
        self.fh.flush()
        self.mh.write(smap.astype(np.float16).tobytes())
        self.mh.flush()

    def close(self) -> None:
        for h in (self.fh, self.mh):
            if h:
                h.close()


class Live:
    def __init__(self, args, cal: Calibration, scorer: Scorer, device: str):
        import torch
        self.torch, self.args, self.cal, self.scorer, self.device = torch, args, cal, scorer, device
        w = cal.window
        self.num_frames, self.model_fps, self.stride = int(w["num_frames"]), float(w["model_fps"]), float(w["stride_s"])
        self.span = (self.num_frames - 1) / self.model_fps
        self.buf: deque = deque(maxlen=self.num_frames)     # (ts, preprocessed H W C)
        self.levels: deque = deque()                        # (ts, STL level) from every message
        self.next_sample = self.next_emit = None
        self.last_ts = None
        self.busy = False
        self.above: deque = deque(maxlen=args.hold)
        self.history: deque = deque(maxlen=int(args.history_s / self.stride))
        self.clients: set = set()
        self.frame_wh = None
        self.daylog = DayLog(args.log_dir)
        self.recent: deque = deque(maxlen=int(args.baseline_s / self.stride))   # maps, for colouring
        self.live_base = None
        self.dropped = 0

    # ── upstream ──────────────────────────────────────────────────────────
    async def upstream(self):
        import websockets
        while True:
            try:
                async with websockets.connect(self.args.jetson, max_size=None, open_timeout=10,
                                              ping_interval=20) as ws:
                    log.info("connected to %s", self.args.jetson)
                    self.status("up")
                    async for raw in ws:
                        self.on_message(json.loads(raw))
            except Exception as e:                                      # noqa: BLE001
                log.warning("upstream %s: %s; retrying in 3 s", self.args.jetson, e)
            self.status("down")
            self.reset()
            await asyncio.sleep(3)

    def reset(self):
        self.buf.clear()
        self.above.clear()
        self.next_sample = self.next_emit = self.last_ts = None

    def on_message(self, d: dict):
        ts = d.get("timestamp")
        if ts is None or not d.get("frame"):
            return
        if self.last_ts is not None and ts - self.last_ts > self.args.max_gap_s:
            log.info("feed gap of %.1f s; restarting the window", ts - self.last_ts)
            self.reset()
        self.last_ts = ts
        self.levels.append((ts, int(d.get("level") or 0)))
        while self.levels and self.levels[0][0] < ts - self.span - 5:
            self.levels.popleft()
        if self.next_sample is None:
            self.next_sample = ts
        if ts + 1e-6 < self.next_sample:
            return
        self.next_sample = max(self.next_sample + 1.0 / self.model_fps, ts - 0.5 / self.model_fps)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(d["frame"]), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        self.frame_wh = (img.shape[1], img.shape[0])
        self.buf.append((ts, preprocess(strip_overlay(img, self.cal.crop_box), self.cal.crop_box)))
        if len(self.buf) < self.num_frames:
            return
        if self.next_emit is None:
            self.next_emit = ts
        if ts < self.next_emit - 0.5 / self.model_fps:      # samples land within half a period of it
            return
        self.next_emit += self.stride
        if ts - self.next_emit > self.stride:               # fell behind (feed stall): rejoin the grid
            self.next_emit = ts + self.stride
        if self.busy:                       # GPU still on the last window: skip this one
            self.dropped += 1
            return
        clip = np.stack([f for _, f in self.buf]).transpose(3, 0, 1, 2)[None].astype(np.float32)
        self.busy = True
        asyncio.get_running_loop().create_task(self.score(ts, clip))

    # ── scoring ───────────────────────────────────────────────────────────
    def _gpu(self, clip):
        torch = self.torch
        t0 = time.perf_counter()
        m = self.scorer.score(torch.from_numpy(clip).to(self.device))
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return m.mean(0).astype(np.float32), 1000 * (time.perf_counter() - t0)

    async def score(self, t_end: float, clip):
        try:
            smap, ms = await asyncio.to_thread(self._gpu, clip)
        except Exception:                                                # noqa: BLE001
            log.exception("scoring failed")
            return
        finally:
            self.busy = False
        s = float(np.percentile(smap, 95))
        shat = s / self.cal.tau
        self.above.append(shat > 1.0)
        alert = len(self.above) == self.args.hold and all(self.above)
        stl = max((lv for t, lv in self.levels if t_end - self.span <= t <= t_end), default=0)
        lag = time.time() - t_end
        row = {"t_end_unix": round(t_end, 3), "s_p95": round(s, 5), "s_mean": round(float(smap.mean()), 5),
               "shat": round(shat, 4), "alert": int(alert), "stl_max": stl, "ms": round(ms, 1),
               "lag_s": round(lag, 2)}
        self.daylog.write(row, smap)
        self.recent.append(smap)
        if len(self.recent) >= 120 and len(self.recent) % 20 == 0:
            self.live_base = np.median(np.stack(self.recent), axis=0)
        msg = {"type": "window", "t": row["t_end_unix"], "s": row["s_p95"], "shat": row["shat"],
               "alert": alert, "stl": stl, "ms": row["ms"], "lag": row["lag_s"],
               "map": self.cal.colour(smap, self.live_base)}
        self.history.append({k: msg[k] for k in ("t", "s", "shat", "alert", "stl")})
        self.broadcast(msg)

    # ── clients ───────────────────────────────────────────────────────────
    def hello(self) -> dict:
        g = IMG_SIZE // PATCH
        return {"type": "hello", "model": "V-JEPA 2.1 ViT-g predictor", "tau": self.cal.tau,
                "alpha": self.cal.alpha, "crop_box": list(self.cal.crop_box),
                "frame_wh": list(self.frame_wh or (1280, 720)), "grid": [g, g], "hold": self.args.hold,
                "span_s": self.span, "stride_s": self.stride, "target_s": self.cal.window["target_s"],
                "upstream": "up" if self.last_ts and time.time() - self.last_ts < 5 else "down",
                "history": list(self.history)}

    def status(self, state: str):
        self.broadcast({"type": "status", "upstream": state})

    def broadcast(self, msg: dict):
        import websockets
        if self.clients:
            websockets.broadcast(self.clients, json.dumps(msg))

    async def handler(self, ws):
        self.clients.add(ws)
        try:
            await ws.send(json.dumps(self.hello()))
            async for _ in ws:              # pings from the page; nothing to answer
                pass
        except Exception:                   # noqa: BLE001
            pass
        finally:
            self.clients.discard(ws)

    async def heartbeat(self):
        while True:
            await asyncio.sleep(60)
            log.info("%d clients, %d windows in history, %d skipped while busy",
                     len(self.clients), len(self.history), self.dropped)


async def run(args, live: Live):
    import websockets
    async with websockets.serve(live.handler, args.bind, args.port, max_size=2 ** 16,
                                ping_interval=20):
        log.info("serving ws://%s:%d", args.bind, args.port)
        await asyncio.gather(live.upstream(), live.heartbeat())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jetson", default="ws://100.120.250.94:8080/ws")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN,
                    help="calibration run (summary.json, surprise.csv, maps.npy) for τ and the window")
    ap.add_argument("--hold", type=int, default=2, help="alert after this many windows above τ")
    ap.add_argument("--history-s", type=float, default=120.0)
    ap.add_argument("--baseline-s", type=float, default=300.0, help="live map-colouring baseline span")
    ap.add_argument("--max-gap-s", type=float, default=2.0, help="restart the window after a feed gap")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "data/captures/live_vjepa")
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "safeedge" / "vjepa")
    ap.add_argument("--hub-repo", default=str(Path.home() / ".cache/torch/hub/facebookresearch_vjepa2_main"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch
    cal = Calibration(args.run)
    w = cal.window
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    target_slices = max(1, round(w["target_s"] * w["model_fps"] / TUBELET))
    enc, pred = load_vjepa("giant", int(w["num_frames"]), device, args.hub_repo, None, args.cache_dir)
    scorer = Scorer(enc, pred, "predictor", int(w["num_frames"]), target_slices, device, dtype)
    log.info("ViT-g loaded on %s; τ=%.4f (α=%g), crop %s", device, cal.tau, cal.alpha, cal.crop_box)
    asyncio.run(run(args, Live(args, cal, scorer, device)))


if __name__ == "__main__":
    main()
