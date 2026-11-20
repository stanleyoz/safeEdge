"""
V-JEPA 2.1 surprise vs STL robustness — offline analysis on a recorded clip.

Question this answers: does the world model's surprise rise BEFORE the
geometric STL monitor raises WARNING / EMERGENCY, or before ρ1 goes negative?

For each sliding window of the clip (default 16 frames at 4 fps = 4 s, the
V-JEPA training frame rate) the model sees the first part of the window as
context and is scored on how well it anticipates the last part (default the
last 1 s). The per-patch error is the surprise map; s(t) is stamped at the
window's END, so it is causal (only uses frames up to t).

Three scoring modes:

  predictor    The model's own JEPA predictor, scored exactly like training:
               L1( P(E(context)), E(full clip)[target] ) over the 4-level
               hierarchical features. Needs ViT-g or ViT-G: the distilled
               ViT-B/L predictors output in their ViT-G teacher's space, so they
               cannot be scored against their own encoder.
  ridge        Any model (meant for the Jetson-sized ViT-B/L). A linear latent
               forecaster, fitted per patch on calm footage, maps the last
               context slice to the target slices; surprise = 1 - cos.
  persistence  No fitting: prediction = the last context slice unchanged, so
               surprise = how much each patch's features changed.
  framediff    No model at all: mean absolute PIXEL change per patch between the
               last context frame and the target frames. The baseline that says
               whether V-JEPA adds anything beyond "something moved".

Calibration is split-conformal: the threshold τ is the ⌈(n+1)(1-α)⌉-th
smallest score over held-out calm windows, so on calm footage that resembles
the calibration set P(s > τ) ≤ α. "Calm" = every traced frame in the window is
level 0 (SAFE); without a trace, the windows inside the first --calib-s
seconds. For ridge the calm set is split in time: first half fits, second
half calibrates.

Inputs: a raw recording (RawRecorder output in data/captures/) and, optionally,
the per-frame trace written by replaying it through the edge loop:

  python -m edge.main --source file --file-path data/captures/X.mp4 \\
      --fps <true fps> --trace data/captures/X.trace.jsonl

  python tools/vjepa_surprise.py --video data/captures/X.mp4 \\
      --trace data/captures/X.trace.jsonl --model giant --mode predictor

  # Jetson-sized model, no trace (calibrates on the first 60 s):
  python tools/vjepa_surprise.py --video X.mp4 --model base --mode ridge

Outputs in --out (default: <video stem>_vjepa/): surprise.csv (one row per
window), summary.json (τ, lead times, false alarms/hour, timing), maps.npy
(per-window [24, 24] surprise maps), plot.png and, with --overlay, overlay.mp4
(surprise heatmap over the scene).

Weights: the hub code's pretrained download points at a test server, so this
tool loads checkpoints itself from dl.fbaipublicfiles.com (cached by torch.hub)
and saves a slim fp16 copy (encoder + predictor only) to --cache-dir, so later
runs skip the 1.7-30 GB training checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
logger = logging.getLogger("safeEdge.vjepa_surprise")

WEIGHTS_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
HUB_REPO = "facebookresearch/vjepa2"

# --model → (hub entry point, checkpoint file, encoder key in the checkpoint)
MODELS = {
    "base":     ("vjepa2_1_vit_base_384",     "vjepa2_1_vitb_dist_vitG_384", "ema_encoder"),
    "large":    ("vjepa2_1_vit_large_384",    "vjepa2_1_vitl_dist_vitG_384", "ema_encoder"),
    "giant":    ("vjepa2_1_vit_giant_384",    "vjepa2_1_vitg_384",           "target_encoder"),
    "gigantic": ("vjepa2_1_vit_gigantic_384", "vjepa2_1_vitG_384",           "target_encoder"),
}
DISTILLED = {"base", "large"}

IMG_SIZE = 384             # all released 2.1 checkpoints are 384 px
PATCH = 16
TUBELET = 2
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Pure helpers (no torch; unit-tested in tests/test_vjepa_surprise.py) ──────

def conformal_threshold(calib_scores, alpha: float) -> float:
    """Split-conformal threshold: the ⌈(n+1)(1-α)⌉-th smallest calibration score.

    Returns +inf when n is too small for the requested α (no finite threshold
    can guarantee it) — the caller should report that rather than alarm.
    """
    s = np.sort(np.asarray(calib_scores, dtype=np.float64))
    n = len(s)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if n == 0 or k > n:
        return math.inf
    return float(s[k - 1])


def onsets(times, values, predicate, min_gap_s: float = 5.0) -> list[float]:
    """Times where predicate(value) turns True, merging onsets < min_gap_s
    after the previous episode ended (so one flapping episode = one onset)."""
    out: list[float] = []
    prev = False
    last_true = -math.inf
    for t, v in zip(times, values):
        cur = bool(predicate(v))
        if cur and not prev and (t - last_true) >= min_gap_s:
            out.append(float(t))
        if cur:
            last_true = t
        prev = cur
    return out


def exceedance_runs(times, scores, tau: float, max_gap: int = 1) -> list[tuple[float, float]]:
    """Contiguous runs of windows with score > tau, as (t_start, t_end).
    Up to max_gap sub-threshold windows inside a run are tolerated."""
    runs: list[tuple[float, float]] = []
    start = end = None
    gap = 0
    for t, s in zip(times, scores):
        if s > tau:
            if start is None:
                start = t
            end, gap = t, 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, end))
                start = end = None
                gap = 0
    if start is not None:
        runs.append((start, end))
    return runs


def lead_time(event_t: float, runs, stride_s: float, max_lead_s: float) -> float | None:
    """Seconds by which surprise was already raised when the geometric alarm fired.

    Counts only a run still active at the event (it ends within one stride of
    event_t, or later) and started no more than max_lead_s before it — so an
    unrelated blip a minute earlier does not count as warning. 0.0 = surprise
    rose together with the alarm; None = no surprise at the event.
    """
    for start, end in runs:
        if start <= event_t + stride_s and end >= event_t - stride_s:
            return float(min(max(event_t - start, 0.0), max_lead_s))
    return None


def episodes(times, values, predicate, min_gap_s: float = 5.0) -> list[tuple[float, float]]:
    """(start, end) of stretches where predicate holds, merging gaps < min_gap_s.
    Starts match onsets() for the same arguments."""
    out: list[list[float]] = []
    for t, v in zip(times, values):
        if predicate(v):
            if out and t - out[-1][1] < min_gap_s:
                out[-1][1] = float(t)
            else:
                out.append([float(t), float(t)])
    return [(a, b) for a, b in out]


def false_alarms(runs, eps, max_lead_s: float, post_s: float) -> int:
    """Exceedance runs overlapping no episode's [start - max_lead_s, end + post_s]."""
    n = 0
    for start, end in runs:
        if not any(start <= b + post_s and end >= a - max_lead_s for a, b in eps):
            n += 1
    return n


def auroc(scores, labels) -> float | None:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count
    half). None when only one class is present."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):                       # average ranks over ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def token_indices(t_slices: int, h: int, w: int, target_slices: int):
    """Context / target token indices for a [T', H', W'] grid flattened t-major
    (the order the 3D patch embed produces): context = the first T'-k slices,
    target = the last k."""
    per = h * w
    split = (t_slices - target_slices) * per
    return np.arange(0, split), np.arange(split, t_slices * per)


def aoi_patch_mask(polygon_norm, crop_box, frame_wh, grid: int) -> np.ndarray:
    """[grid, grid] bool: patches whose centre falls inside the AOI polygon,
    in the coordinates of the (cropped, squashed) model input."""
    fw, fh = frame_wh
    x0, y0, x1, y1 = crop_box
    poly = np.array([[x * fw, y * fh] for x, y in polygon_norm], dtype=np.float32)
    mask = np.zeros((grid, grid), dtype=bool)
    for r in range(grid):
        for c in range(grid):
            px = x0 + (c + 0.5) / grid * (x1 - x0)
            py = y0 + (r + 0.5) / grid * (y1 - y0)
            mask[r, c] = cv2.pointPolygonTest(poly, (px, py), False) >= 0
    return mask


def aoi_crop_box(polygon_norm, frame_wh, pad: float) -> tuple[int, int, int, int]:
    """Same AOI bounding box as edge/detection/aoi.py (polygon bbox + pad)."""
    fw, fh = frame_wh
    xs = [x * fw for x, _ in polygon_norm]
    ys = [y * fh for _, y in polygon_norm]
    pw, ph = (max(xs) - min(xs)) * pad, (max(ys) - min(ys)) * pad
    return (max(0, int(min(xs) - pw)), max(0, int(min(ys) - ph)),
            min(fw, int(max(xs) + pw)), min(fh, int(max(ys) + ph)))


# ── Trace ──────────────────────────────────────────────────────────────────

@dataclass
class Trace:
    t_s: np.ndarray
    rho1: np.ndarray
    level: np.ndarray

    @classmethod
    def load(cls, path: Path, video_fps: float) -> "Trace":
        frames, rho1, level = [], [], []
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    frames.append(r["frame"])
                    rho1.append(r["rho1"])
                    level.append(r["level"])
        # Align on the VIDEO's frame clock: the edge loop's t_s uses its own
        # --fps, which may differ from the container fps if it was overridden.
        return cls(np.asarray(frames) / video_fps, np.asarray(rho1, dtype=float),
                   np.asarray(level, dtype=int))

    def max_level_between(self, t0: float, t1: float) -> int | None:
        sel = (self.t_s >= t0) & (self.t_s <= t1)
        return int(self.level[sel].max()) if sel.any() else None

    def at(self, t: float) -> tuple[float, int]:
        i = int(np.clip(np.searchsorted(self.t_s, t), 0, len(self.t_s) - 1))
        return float(self.rho1[i]), int(self.level[i])


# ── Model ──────────────────────────────────────────────────────────────────

def _clean_keys(sd: dict) -> dict:
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def load_vjepa(model: str, num_frames: int, device: str, hub_repo: str,
               weights: str | None, cache_dir: Path):
    import torch

    entry, ckpt, enc_key = MODELS[model]
    local = Path(hub_repo).exists()
    encoder, predictor = torch.hub.load(
        hub_repo, entry, source="local" if local else "github",
        pretrained=False, num_frames=num_frames, trust_repo=True)

    slim = cache_dir / f"{ckpt}_slim_fp16.pt"
    if weights is None and slim.exists():
        logger.info("Loading slim weights %s", slim)
        sd = torch.load(slim, map_location="cpu")
        enc_sd, pred_sd = sd["encoder"], sd["predictor"]
    else:
        if weights:
            logger.info("Loading checkpoint %s", weights)
            sd = torch.load(weights, map_location="cpu", weights_only=False)
        else:
            url = f"{WEIGHTS_BASE_URL}/{ckpt}.pt"
            logger.info("Downloading %s (training checkpoint, cached by torch.hub)", url)
            sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        enc_sd, pred_sd = _clean_keys(sd[enc_key]), _clean_keys(sd["predictor"])
        del sd
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder": {k: v.half() for k, v in enc_sd.items()},
                    "predictor": {k: v.half() for k, v in pred_sd.items()}}, slim)
        logger.info("Saved slim fp16 weights → %s", slim)

    # Slim copies are fp16; load into fp32 modules (strict, as the hub does).
    encoder.load_state_dict({k: v.float() for k, v in enc_sd.items()}, strict=True)
    predictor.load_state_dict({k: v.float() for k, v in pred_sd.items()}, strict=True)
    encoder.eval().to(device)
    predictor.eval().to(device)
    return encoder, predictor


def framediff_map(clip: np.ndarray, num_frames: int, target_slices: int) -> np.ndarray:
    """[1, G, G] mean |target frame - last context frame| per 16 px patch, on the
    same normalised model input the encoder sees."""
    x = clip[0]                                             # C T H W
    split = num_frames - target_slices * TUBELET
    diff = np.abs(x[:, split:] - x[:, split - 1:split]).mean(axis=(0, 1))   # H W
    g = IMG_SIZE // PATCH
    return diff.reshape(g, PATCH, g, PATCH).mean(axis=(1, 3))[None]


class Scorer:
    """Turns one preprocessed clip into a per-patch surprise map [k, G, G]."""

    def __init__(self, encoder, predictor, mode: str, num_frames: int,
                 target_slices: int, device: str, dtype):
        import torch
        self.torch = torch
        self.enc, self.pred = encoder, predictor
        self.mode, self.device, self.dtype = mode, device, dtype
        self.grid = IMG_SIZE // PATCH
        self.t_slices = num_frames // TUBELET
        self.k = target_slices
        ctx, tgt = token_indices(self.t_slices, self.grid, self.grid, target_slices)
        self.ctx_idx = torch.as_tensor(ctx, device=device).unsqueeze(0)
        self.tgt_idx = torch.as_tensor(tgt, device=device).unsqueeze(0)
        # The predictor is trained on the 4-level hierarchical features.
        self.enc.return_hierarchical = (mode == "predictor")
        # ridge: accumulated normal equations for a shared per-patch linear map
        self._xtx = self._xty = None
        self.W = None

    def _features(self, clip):
        """(context-only encoding, full-clip encoding), both [1, N, D]."""
        torch = self.torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype,
                                                    enabled=self.device.startswith("cuda")):
            z_ctx = self.enc(clip, masks=[self.ctx_idx])
            z_all = self.enc(clip)
        return z_ctx, z_all

    def _pairs(self, clip):
        """ridge/persistence: X = last context slice, Y = mean of target slices,
        each [G*G, D] (one row per patch), L2-normalised."""
        z_ctx, z_all = self._features(clip)
        per = self.grid * self.grid
        x = z_ctx[0, -per:].float()
        y = z_all[0, self.tgt_idx[0]].float().view(self.k, per, -1).mean(0)
        f = self.torch.nn.functional.normalize
        return f(x, dim=-1), f(y, dim=-1)

    def accumulate(self, clip) -> None:
        x, y = self._pairs(clip)
        x1 = self.torch.cat([x, self.torch.ones_like(x[:, :1])], dim=1)   # bias column
        xtx, xty = x1.T @ x1, x1.T @ y
        self._xtx = xtx if self._xtx is None else self._xtx + xtx
        self._xty = xty if self._xty is None else self._xty + xty

    def fit(self, ridge_lambda: float) -> None:
        a = self._xtx + ridge_lambda * self.torch.eye(self._xtx.shape[0], device=self._xtx.device)
        self.W = self.torch.linalg.solve(a, self._xty)

    def score(self, clip) -> np.ndarray:
        torch = self.torch
        g = self.grid
        if self.mode == "predictor":
            z_ctx, z_all = self._features(clip)
            with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype,
                                                        enabled=self.device.startswith("cuda")):
                pred, _ = self.pred(z_ctx, masks_x=[self.ctx_idx], masks_y=[self.tgt_idx])
            tgt = z_all[:, self.tgt_idx[0]]
            err = (pred.float() - tgt.float()).abs().mean(-1)               # L1, as in training
            return err.view(self.k, g, g).cpu().numpy()
        x, y = self._pairs(clip)
        if self.mode == "ridge":
            x1 = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
            y_hat = torch.nn.functional.normalize(x1 @ self.W, dim=-1)
        else:
            y_hat = x
        err = 1.0 - (y_hat * y).sum(-1)                                     # 1 - cosine
        return err.view(1, g, g).cpu().numpy()


# ── Video → windows ─────────────────────────────────────────────────────────

@dataclass
class Window:
    idx: int
    t_end: float            # time of the last frame in the window (s)
    frame_end: int
    calm: bool = False
    role: str = "score"     # "fit" | "calib" | "score"
    scores: dict = field(default_factory=dict)
    smap: np.ndarray | None = None   # [G, G] surprise map (mean over target slices)


def preprocess(frame_bgr, crop_box) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    img = cv2.resize(frame_bgr[y0:y1, x0:x1], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (img - IMAGENET_MEAN) / IMAGENET_STD                             # H W C


def iter_windows(video: Path, vfps: float, crop_box, model_fps: float, num_frames: int,
                 stride_s: float, wanted: set[int] | None = None):
    """Yield (window_idx, t_end, frame_end, clip [1,C,T,H,W] np.float32, last_bgr).

    Samples the video at model_fps by timestamp; a window ends every stride_s.
    wanted: if given, only those window indices are assembled (others are only
    decoded), which keeps the ridge fit pass cheap.
    """
    cap = cv2.VideoCapture(str(video))
    buf: deque = deque(maxlen=num_frames)
    next_sample, next_emit, widx, fidx = 0.0, None, 0, -1
    span = (num_frames - 1) / model_fps
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fidx += 1
        t = fidx / vfps
        if t + 1e-9 < next_sample:
            continue
        next_sample += 1.0 / model_fps
        buf.append((t, fidx, frame))
        if len(buf) < num_frames:
            continue
        if next_emit is None:
            next_emit = span
        if t + 1e-9 < next_emit:
            continue
        next_emit += stride_s
        if wanted is None or widx in wanted:
            clip = np.stack([preprocess(f, crop_box) for _, _, f in buf])  # T H W C
            clip = clip.transpose(3, 0, 1, 2)[None]                         # 1 C T H W
            yield widx, t, fidx, clip, frame
        widx += 1
    cap.release()


def video_info(video: Path) -> tuple[float, int, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return fps, n, wh


def plan_windows(duration_s: float, model_fps: float, num_frames: int, stride_s: float):
    """End times of the windows iter_windows will emit (same arithmetic)."""
    span = (num_frames - 1) / model_fps
    n = max(0, int(math.floor((duration_s - span) / stride_s + 1e-9)) + 1)
    return [span + i * stride_s for i in range(n)]


# ── Reporting ───────────────────────────────────────────────────────────────

def write_plot(path: Path, windows, score_key: str, tau: float, trace: Trace | None,
               ev_level: list[float], ev_rho: list[float], title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")
        return False
    t = [w.t_end for w in windows]
    s = [w.scores[score_key] for w in windows]
    rows = 2 if trace is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3.2 * rows), sharex=True, squeeze=False)
    ax_s = axes[-1][0]
    if trace is not None:
        ax = axes[0][0]
        ax.plot(trace.t_s, trace.rho1, lw=0.8, color="#1f5fa8", label="ρ1 (m)")
        ax.axhline(0, color="#888", lw=0.8)
        for lvl, col in ((2, "#e8a33d"), (3, "#c8423b")):
            ax.fill_between(trace.t_s, 0, 1, where=trace.level >= lvl, color=col, alpha=0.18,
                            transform=ax.get_xaxis_transform(), lw=0,
                            label="WARNING+" if lvl == 2 else "EMERGENCY")
        ax.set_ylabel("ρ1  (d_min − 3 m)")
        ax.legend(loc="upper right", fontsize=8)
    ax_s.plot(t, s, lw=1.0, color="#333", label=f"surprise ({score_key})")
    if math.isfinite(tau):
        ax_s.axhline(tau, color="#c8423b", lw=1, ls="--", label="τ (conformal)")
    for i, te in enumerate(ev_level):
        for ax_row in axes:
            ax_row[0].axvline(te, color="#e8a33d", lw=1, label="WARNING+ onset" if i == 0 else None)
    for i, te in enumerate(ev_rho):
        for ax_row in axes:
            ax_row[0].axvline(te, color="#c8423b", lw=1, ls=":", label="ρ1 < 0 onset" if i == 0 else None)
    ax_s.set_ylabel("surprise")
    ax_s.set_xlabel("time (s)")
    ax_s.legend(loc="upper right", fontsize=8)
    axes[0][0].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def write_overlay(path: Path, video: Path, windows, crop_box, score_key: str,
                  tau: float, fps: float) -> None:
    """One output frame per window: the window's last frame with its surprise
    map on top. Each patch is coloured by how far it rises above its own calm
    baseline (median over calibration windows), scaled so a rise of (τ - calm
    median score) is mid-range: errors have a large, uneven floor (predictor
    mode sits near 0.63 everywhere), so raw values would show a flat wash.
    Model-free second pass."""
    by_frame = {w.frame_end: w for w in windows}
    calib = [w for w in windows if w.role == "calib"] or windows
    base = np.median(np.stack([w.smap for w in calib]), axis=0)
    floor = float(np.median([w.scores[score_key] for w in calib]))
    rise = np.stack([np.maximum(w.smap - base, 0) for w in windows])
    scale = (tau - floor) if math.isfinite(tau) and tau > floor else float(np.percentile(rise, 99))
    scale = scale or 1.0
    x0, y0, x1, y1 = crop_box
    cap = cv2.VideoCapture(str(video))
    writer, fidx = None, -1
    while by_frame:
        ok, frame = cap.read()
        if not ok:
            break
        fidx += 1
        w = by_frame.pop(fidx, None)
        if w is None:
            continue
        img = frame[y0:y1, x0:x1].copy()
        h, wd = img.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (wd, h))
        heat = np.clip(np.maximum(w.smap - base, 0) / (2 * scale), 0, 1)
        heat = cv2.resize((heat * 255).astype(np.uint8), (wd, h), interpolation=cv2.INTER_CUBIC)
        img = cv2.addWeighted(img, 0.6, cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO), 0.4, 0)
        s = w.scores[score_key]
        tag = "  SURPRISE" if s > tau else ""
        cv2.putText(img, f"t={w.t_end:6.1f}s  {score_key}={s:.4f}{tag}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(img)
    cap.release()
    if writer is not None:
        writer.release()


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--trace", type=Path, default=None,
                    help="per-frame JSONL from `python -m edge.main ... --trace`")
    ap.add_argument("--model", choices=sorted(MODELS), default="giant")
    ap.add_argument("--mode", choices=("predictor", "ridge", "persistence", "framediff"),
                    default=None,
                    help="default: predictor for giant/gigantic, ridge for base/large")
    ap.add_argument("--score", choices=("p95", "mean", "aoi"), default="p95",
                    help="window score that is thresholded: 95th-percentile patch "
                         "(local hazards), whole-frame mean, or mean over AOI patches")
    ap.add_argument("--video-fps", type=float, default=None,
                    help="TRUE capture fps (frames ÷ wall-duration) when the container's "
                         "fps is wrong — pass the same value as the edge replay's --fps")
    ap.add_argument("--model-fps", type=float, default=4.0,
                    help="frame sampling rate fed to V-JEPA (trained at 4 fps)")
    ap.add_argument("--num-frames", type=int, default=16, help="frames per window (even)")
    ap.add_argument("--target-s", type=float, default=1.0,
                    help="seconds at the end of each window the model must anticipate")
    ap.add_argument("--stride-s", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="target false-alarm rate per calm window (conformal)")
    ap.add_argument("--calib-s", type=float, default=60.0,
                    help="without a trace: calibrate on windows ending in the first N s")
    ap.add_argument("--max-lead-s", type=float, default=8.0,
                    help="surprise raised more than this before an event does not count as lead")
    ap.add_argument("--post-s", type=float, default=3.0,
                    help="surprise this long after an event onset counts as the same event")
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--ridge-from", type=Path, default=None,
                    help="ridge.pt saved by an earlier run (another recording): apply that "
                         "forecaster instead of fitting here, so every calm window calibrates")
    ap.add_argument("--no-aoi-crop", action="store_true",
                    help="feed the full frame instead of the AOI bounding box from camera_config")
    ap.add_argument("--camera-config", type=Path, default=ROOT / "config" / "camera_config.yaml")
    ap.add_argument("--overlay", action="store_true", help="also write overlay.mp4")
    ap.add_argument("--max-windows", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--hub-repo", default=HUB_REPO,
                    help="torch.hub repo, or a local clone of facebookresearch/vjepa2")
    ap.add_argument("--weights", default=None, help="local checkpoint instead of downloading")
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "safeedge" / "vjepa")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")

    import torch

    mode = args.mode or ("ridge" if args.model in DISTILLED else "predictor")
    if mode == "predictor" and args.model in DISTILLED:
        ap.error(f"--mode predictor needs giant or gigantic: the distilled {args.model} "
                 "predictor outputs in its ViT-G teacher's feature space")
    if args.num_frames % TUBELET:
        ap.error("--num-frames must be even (tubelet size 2)")
    target_slices = max(1, round(args.target_s * args.model_fps / TUBELET))
    if target_slices >= args.num_frames // TUBELET:
        ap.error("--target-s leaves no context; lower it or raise --num-frames")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() \
        else torch.float16
    out = args.out or args.video.with_name(args.video.stem + "_vjepa")
    out.mkdir(parents=True, exist_ok=True)

    vfps, nframes, frame_wh = video_info(args.video)
    if args.video_fps:
        logger.info("Using true capture fps %.2f (container says %.2f)", args.video_fps, vfps)
        vfps = args.video_fps
    duration = nframes / vfps
    cam = yaml.safe_load(open(args.camera_config)) if args.camera_config.exists() else {}
    aoi = (cam or {}).get("aoi") or {}
    poly = aoi.get("polygon") if aoi.get("enabled") else None
    crop_box = (0, 0, *frame_wh) if (args.no_aoi_crop or not poly) \
        else aoi_crop_box(poly, frame_wh, aoi.get("pad", 0.08))
    grid = IMG_SIZE // PATCH
    aoi_mask = aoi_patch_mask(poly, crop_box, frame_wh, grid) if poly else np.ones((grid, grid), bool)
    if not aoi_mask.any():
        aoi_mask[:] = True
    logger.info("Video %s: %.1f s @ %.2f fps, %dx%d | crop %s | model %s, mode %s",
                args.video.name, duration, vfps, *frame_wh, crop_box, args.model, mode)

    trace = Trace.load(args.trace, vfps) if args.trace else None
    span = (args.num_frames - 1) / args.model_fps
    windows = [Window(i, t, int(round(t * vfps)))
               for i, t in enumerate(plan_windows(duration, args.model_fps,
                                                  args.num_frames, args.stride_s))]
    if args.max_windows:
        windows = windows[: args.max_windows]
    if not windows:
        raise SystemExit(f"clip shorter than one {span:.1f} s window")

    for w in windows:
        if trace is not None:
            lvl = trace.max_level_between(w.t_end - span, w.t_end)
            w.calm = lvl == 0
        else:
            w.calm = w.t_end <= args.calib_s
    calm = [w for w in windows if w.calm]
    if mode == "ridge" and not args.ridge_from:
        half = len(calm) // 2
        for w in calm[:half]:
            w.role = "fit"
        for w in calm[half:]:
            w.role = "calib"
    else:
        for w in calm:
            w.role = "calib"
    logger.info("%d windows (%d calm: %d fit, %d calib)", len(windows), len(calm),
                sum(w.role == "fit" for w in windows), sum(w.role == "calib" for w in windows))

    if mode == "framediff":
        scorer = None
    else:
        encoder, predictor = load_vjepa(args.model, args.num_frames, device, args.hub_repo,
                                        args.weights, args.cache_dir)
        scorer = Scorer(encoder, predictor, mode, args.num_frames, target_slices, device, dtype)

    def to_dev(clip):
        return torch.from_numpy(clip).to(device)

    by_idx = {w.idx: w for w in windows}
    if mode == "ridge" and args.ridge_from:
        scorer.W = torch.load(args.ridge_from, map_location=device)
        logger.info("Ridge forecaster loaded from %s", args.ridge_from)
    elif mode == "ridge":
        fit_ids = {w.idx for w in windows if w.role == "fit"}
        if not fit_ids:
            raise SystemExit("ridge mode found no calm windows to fit on — use --mode "
                             "persistence, or a clip/trace with a calm stretch")
        for _widx, _t, _f, clip, _last in iter_windows(args.video, vfps, crop_box, args.model_fps,
                                                       args.num_frames, args.stride_s, fit_ids):
            scorer.accumulate(to_dev(clip))
        scorer.fit(args.ridge_lambda)
        torch.save(scorer.W, out / "ridge.pt")
        logger.info("Ridge forecaster fitted on %d windows → %s", len(fit_ids), out / "ridge.pt")

    timings = []
    for widx, t_end, fidx, clip, _last in iter_windows(args.video, vfps, crop_box, args.model_fps,
                                                          args.num_frames, args.stride_s):
        if widx not in by_idx:
            break
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m = framediff_map(clip, args.num_frames, target_slices) if scorer is None \
            else scorer.score(to_dev(clip))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
        w = by_idx[widx]
        w.t_end, w.frame_end = t_end, fidx
        w.smap = m.mean(0).astype(np.float32)
        w.scores = {"mean": float(w.smap.mean()), "p95": float(np.percentile(w.smap, 95)),
                    "aoi": float(w.smap[aoi_mask].mean())}
        if widx % 20 == 0:
            logger.info("window %d/%d t=%.1fs %s=%.4f (%.0f ms)", widx + 1, len(windows), t_end,
                        args.score, w.scores[args.score], 1000 * timings[-1])
    windows = [w for w in windows if w.scores]

    calib = [w.scores[args.score] for w in windows if w.role == "calib"]
    tau = conformal_threshold(calib, args.alpha)
    if not math.isfinite(tau):
        logger.warning("Only %d calibration windows — too few for α=%.3f; no threshold. "
                       "Lower --alpha or use a longer calm stretch.", len(calib), args.alpha)

    times = [w.t_end for w in windows]
    scores = [w.scores[args.score] for w in windows]
    # Fit windows are masked to zero so in-sample errors can neither raise nor
    # break up a surprise run (they are calm by construction).
    masked = [0.0 if w.role == "fit" else w.scores[args.score] for w in windows]
    runs = exceedance_runs(times, masked, tau) if math.isfinite(tau) else []
    ev_level = onsets(trace.t_s, trace.level, lambda v: v >= 2) if trace is not None else []
    ev_rho = onsets(trace.t_s, trace.rho1, lambda v: v < 0) if trace is not None else []

    def lead_report(evs):
        leads = [lead_time(te, runs, args.stride_s, args.max_lead_s) for te in evs]
        hits = [x for x in leads if x is not None]
        return {"onsets_s": evs, "lead_s": leads,
                "detected": len(hits), "n": len(evs),
                "median_lead_s": float(np.median(hits)) if hits else None}

    hours = len(windows) * args.stride_s / 3600.0
    eps = episodes(trace.t_s, trace.level, lambda v: v >= 2) if trace is not None else []
    n_fa = false_alarms(runs, eps, args.max_lead_s, args.post_s) if trace is not None else None

    # Threshold-free view. "concurrent": does the score separate windows that
    # contain WARNING+ from those that don't? "anticipation_Hs": among windows
    # that are entirely calm, does the score predict WARNING+ within the next H s?
    # The second is the real early-warning question.
    # Windows the ridge forecaster was fitted on are in-sample, so they are left
    # out of every metric below (and were never used for τ).
    window_auroc = None
    if trace is not None:
        sc = np.array(scores)
        ev_idx = [i for i, w in enumerate(windows) if w.role != "fit"]
        hot = [(trace.max_level_between(windows[i].t_end - span, windows[i].t_end) or 0) >= 2
               for i in ev_idx]
        window_auroc = {"concurrent": auroc(sc[ev_idx], hot)}
        calm_idx = [i for i, w in enumerate(windows) if w.calm and w.role != "fit"]
        for h in (2.0, 4.0):
            ahead = [(trace.max_level_between(windows[i].t_end + 1e-6,
                                              windows[i].t_end + h) or 0) >= 2 for i in calm_idx]
            window_auroc[f"anticipation_{h:g}s"] = auroc(sc[calm_idx], ahead)
            window_auroc[f"anticipation_{h:g}s_positives"] = int(sum(ahead))
    summary = {
        "video": str(args.video), "trace": str(args.trace) if args.trace else None,
        "model": None if mode == "framediff" else args.model, "mode": mode, "score": args.score,
        "window": {"num_frames": args.num_frames, "model_fps": args.model_fps,
                   "span_s": span, "target_s": target_slices * TUBELET / args.model_fps,
                   "stride_s": args.stride_s},
        "crop_box": crop_box,
        "ridge_from": str(args.ridge_from) if args.ridge_from else None,
        "n_fit_windows_excluded": sum(w.role == "fit" for w in windows),
        "calibration": {"alpha": args.alpha, "n_calib": len(calib), "tau": tau,
                        "basis": "trace level 0" if trace is not None else f"first {args.calib_s} s"},
        "events": {"warning_plus": lead_report(ev_level), "rho1_negative": lead_report(ev_rho)},
        "warning_episodes": len(eps),
        "window_auroc": window_auroc,
        "surprise_runs": len(runs),
        "false_alarms": n_fa,
        "false_alarms_per_hour": n_fa / hours if n_fa is not None and hours > 0 else None,
        "timing_ms": {"median": 1000 * float(np.median(timings)),
                      "p95": 1000 * float(np.percentile(timings, 95)),
                      "device": torch.cuda.get_device_name() if device.startswith("cuda") else device},
    }

    with open(out / "surprise.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_end_s", "frame_end", "role", "calm", "s_mean", "s_p95", "s_aoi",
                     "above_tau", "rho1", "level"])
        for w in windows:
            rho1, lvl = trace.at(w.t_end) if trace is not None else ("", "")
            wr.writerow([f"{w.t_end:.3f}", w.frame_end, w.role, int(w.calm),
                         f"{w.scores['mean']:.6f}", f"{w.scores['p95']:.6f}",
                         f"{w.scores['aoi']:.6f}", int(w.scores[args.score] > tau),
                         rho1, lvl])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out / "maps.npy", np.stack([w.smap for w in windows]).astype(np.float16))
    write_plot(out / "plot.png", windows, args.score, tau, trace, ev_level, ev_rho,
               f"{args.video.name} · " + ("pixel frame-difference baseline" if mode == "framediff" else f"V-JEPA 2.1 {args.model} · {mode}") + f" · τ at α={args.alpha}")
    if args.overlay:
        write_overlay(out / "overlay.mp4", args.video, windows, crop_box, args.score, tau,
                      1.0 / args.stride_s)

    logger.info("τ=%.5f from %d calm windows; %d surprise runs", tau, len(calib), len(runs))
    if window_auroc:
        logger.info("AUROC %s", {k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in window_auroc.items()})
    for name, rep in summary["events"].items():
        if rep["n"]:
            logger.info("%s: %d/%d events had surprise raised, median lead %s s",
                        name, rep["detected"], rep["n"], rep["median_lead_s"])
    logger.info("Timing: median %.0f ms / window on %s → %s",
                summary["timing_ms"]["median"], summary["timing_ms"]["device"], out)


if __name__ == "__main__":
    main()
