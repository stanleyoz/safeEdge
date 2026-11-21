"""
Self-labelling for conflict forecasting: turn a replayed clip's per-frame trace
into "a pedestrian-vehicle conflict happens HERE within the next h seconds"
targets on the V-JEPA patch grid, plus the constant-velocity forecast a learned
head has to beat.

SafeEdge labels its own footage. Replay any recording through the unchanged
edge loop with --trace (python -m edge.main --source file --trace ...), then:

  python tools/conflict_labels.py --trace X.trace.jsonl --fps 7.7 \\
      --windows-from research/vjepa-surprise/results/vitg_predictor --out labels/X

A conflict is a pedestrian-vehicle pair, on the ground plane, closer than
--d-conflict and converging faster than --closing-gate or with the vehicle
moving faster than --v-move (the same shape as the STL EMERGENCY rule). Speeds
come from net displacement over --vel-window-s, not frame-to-frame differences:
on the July clip, parked-car tracks accumulate 50-350 m of jitter "path" for
2-4 m of net displacement, and frame-difference speeds would label parked cars
as moving. Steps faster than --max-speed are track-ID switches (bimodal on that
clip: real steps stay under 0.3 m/frame, switches jump 1-16 m) and break the
track instead of counting as motion. Even so, with the July homography a
driving car reads about 0.5 m/s and parked-car jitter reaches the same, so
vehicle_moving is evidence for review, not ground truth: --review-video writes
a sheet of three frames per event so a person can confirm labels in seconds. --require-vehicle-motion keeps only conflicts with a moving vehicle
(the genuine conflicts on that clip; the rest are pedestrians beside parked
cars).

Each conflict frame marks the image region spanning both boxes (the "collision
space") on the 24 x 24 grid of the cropped model input, so targets line up with
V-JEPA patch features. For every window end t and horizon h:

  target[h, t]  union of conflict regions over frames in (t, t + h]
  onset[h, t]   a conflict starts in (t, t + h] with none in the last 0.5 s:
                the anticipation case, as opposed to an ongoing conflict
  cv_pred[h, t] where constant-velocity extrapolation of the tracks at t
                predicts a conflict within h (the phi3-style baseline; it can
                only forecast objects already detected)

Labels inherit the trace's geometry: with the July homography, distances are
compressed (see research/vjepa-surprise/README.md), so recalibrate before
labelling at scale.

Outputs in --out: labels.npz, events.json, summary.json and, with --plot, a
timeline plot.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.vjepa_surprise import (  # noqa: E402
    IMG_SIZE, PATCH, VEHICLE_LABELS, boxes_to_patch_mask, plan_windows,
)

GRID = IMG_SIZE // PATCH


# ── Tracks ──────────────────────────────────────────────────────────────────

@dataclass
class Obs:
    xz: np.ndarray          # ground plane, metres
    box: np.ndarray         # full-frame pixels x0 y0 x1 y1


@dataclass
class Track:
    kind: str               # "ped" | "veh"
    obs: dict = field(default_factory=dict)    # frame -> Obs

    def at_or_before(self, f: int, max_back: int) -> tuple[int, Obs] | None:
        for g in range(f, f - max_back - 1, -1):
            if g in self.obs:
                return g, self.obs[g]
        return None

    def velocity(self, f: int, k: int, fps: float, max_speed: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Ground (m/s) and box (px/s) velocity from net displacement over the
        last k frames, using only the unbroken part of the track: a step faster
        than max_speed is an ID switch (the tracker handing an id to another
        car), not motion, and ends the segment. None when the clean segment is
        shorter than k/2 frames: unknown, rather than wrong."""
        if f not in self.obs:
            return None
        g, cur = f, self.obs[f]
        for back in range(f - 1, f - k - 1, -1):
            o = self.obs.get(back)
            if o is None:
                continue
            if np.linalg.norm(self.obs[g].xz - o.xz) > max_speed * (g - back) / fps:
                break
            g = back
        if f - g < max(1, k // 2):
            return None
        o, dt = self.obs[g], (f - g) / fps
        return (cur.xz - o.xz) / dt, (cur.box - o.box) / dt


def load_tracks(trace_path: Path, all_persons: bool) -> tuple[list[dict], dict[int, Track]]:
    """Per-frame records and per-id tracks. Pedestrians outside the AOI are
    dropped unless all_persons (STL ignores them too); objects without a
    ground position are skipped."""
    recs = [json.loads(line) for line in open(trace_path) if line.strip()]
    if any(r.get("objects") is None for r in recs):
        raise SystemExit("trace has no per-object data: re-run edge.main --trace with this repo")
    tracks: dict[int, Track] = {}
    for r in recs:
        for o in r["objects"]:
            if o["xz"] is None:
                continue
            if o["label"] == "person":
                if not (o["in_aoi"] or all_persons):
                    continue
                kind = "ped"
            elif o["label"] in VEHICLE_LABELS:
                kind = "veh"
            else:
                continue
            t = tracks.setdefault(o["id"], Track(kind))
            t.obs[r["frame"]] = Obs(np.asarray(o["xz"], float), np.asarray(o["box"], float))
    return recs, tracks


# ── Conflicts ───────────────────────────────────────────────────────────────

@dataclass
class Params:
    fps: float
    d_conflict: float = 1.8
    closing_gate: float = 0.5
    v_move: float = 0.5
    vel_window_s: float = 1.0
    require_vehicle_motion: bool = False
    max_speed: float = 6.0      # m/s; faster steps are ID switches

    @property
    def k(self) -> int:
        return max(1, int(round(self.vel_window_s * self.fps)))


def hull(a: np.ndarray, b: np.ndarray) -> list[float]:
    """Smallest box containing both: the image region between and including
    the pedestrian and the vehicle."""
    return [float(min(a[0], b[0])), float(min(a[1], b[1])),
            float(max(a[2], b[2])), float(max(a[3], b[3]))]


def is_conflict(d: float, closing: float | None, v_speed: float | None, p: Params) -> bool:
    if d >= p.d_conflict:
        return False
    moving = v_speed is not None and v_speed > p.v_move
    if p.require_vehicle_motion:
        return moving
    return moving or (closing is not None and closing > p.closing_gate)


def frame_conflicts(f: int, peds, vehs, tracks: dict[int, Track], p: Params) -> list[dict]:
    """Conflicting pairs at frame f, each with its region and evidence."""
    out = []
    for pid in peds:
        po = tracks[pid].obs[f]
        for vid in vehs:
            vo = tracks[vid].obs[f]
            d = float(np.linalg.norm(po.xz - vo.xz))
            if d >= p.d_conflict:
                continue
            pv, vv = tracks[pid].velocity(f, p.k, p.fps, p.max_speed), tracks[vid].velocity(f, p.k, p.fps, p.max_speed)
            v_speed = float(np.linalg.norm(vv[0])) if vv else None
            closing = None
            if pv and vv:
                rel, rel_v = po.xz - vo.xz, pv[0] - vv[0]
                closing = float(-(rel @ rel_v) / max(np.linalg.norm(rel), 1e-6))
            if is_conflict(d, closing, v_speed, p):
                out.append({"ped": pid, "veh": vid, "d": d, "closing": closing,
                            "v_speed": v_speed, "region": hull(po.box, vo.box)})
    return out


def present(f: int, tracks: dict[int, Track], kind: str) -> list[int]:
    return [i for i, t in tracks.items() if t.kind == kind and f in t.obs]


def cv_forecast(f: int, h: float, tracks: dict[int, Track], p: Params,
                step_s: float = 0.25) -> list[list[float]]:
    """Regions where extrapolating every detected track at constant velocity
    from frame f gives a conflict within h seconds: the boxes moved by their
    pixel velocity to the first conflicting time, hulled. Objects without a
    velocity yet (just appeared) are held still."""
    regions = []
    peds, vehs = present(f, tracks, "ped"), present(f, tracks, "veh")
    zero = (np.zeros(2), np.zeros(4))
    for pid in peds:
        po, (pv, pbv) = tracks[pid].obs[f], tracks[pid].velocity(f, p.k, p.fps, p.max_speed) or zero
        for vid in vehs:
            vo, (vv, vbv) = tracks[vid].obs[f], tracks[vid].velocity(f, p.k, p.fps, p.max_speed) or zero
            v_speed = float(np.linalg.norm(vv))
            for tau in np.arange(step_s, h + 1e-9, step_s):
                rel = (po.xz + pv * tau) - (vo.xz + vv * tau)
                d = float(np.linalg.norm(rel))
                closing = float(-(rel @ (pv - vv)) / max(d, 1e-6))
                if is_conflict(d, closing, v_speed, p):
                    regions.append(hull(po.box + pbv * tau, vo.box + vbv * tau))
                    break
    return regions


def group_events(conf_frames: dict[int, list[dict]], fps: float, merge_gap_s: float,
                 v_move: float = 0.5) -> list[dict]:
    """Consecutive conflict frames (gaps up to merge_gap_s) form one event."""
    events, cur = [], None
    for f in sorted(conf_frames):
        if cur is None or (f - cur["end_frame"]) / fps > merge_gap_s:
            cur = {"onset_frame": f, "end_frame": f, "pairs": set(), "min_d": math.inf,
                   "max_closing": None, "max_v_speed": None, "region": None}
            events.append(cur)
        cur["end_frame"] = f
        for c in conf_frames[f]:
            cur["pairs"].add((c["ped"], c["veh"]))
            cur["min_d"] = min(cur["min_d"], c["d"])
            for key, src in (("max_closing", "closing"), ("max_v_speed", "v_speed")):
                if c[src] is not None:
                    cur[key] = c[src] if cur[key] is None else max(cur[key], c[src])
            r = c["region"]
            cur["region"] = r if cur["region"] is None else hull(np.asarray(cur["region"]), np.asarray(r))
    for e in events:
        e["onset_s"], e["end_s"] = e["onset_frame"] / fps, e["end_frame"] / fps
        e["pairs"] = sorted([list(x) for x in e["pairs"]])
        e["vehicle_moving"] = e["max_v_speed"] is not None and e["max_v_speed"] > v_move
    return events


# ── Windows and targets ─────────────────────────────────────────────────────

def window_ends(n_frames: int, fps: float, windows_from: Path | None, span_s: float,
                stride_s: float) -> np.ndarray:
    if windows_from is not None:
        rows = list(csv.DictReader(open(windows_from / "surprise.csv")))
        return np.array([float(r["t_end_s"]) for r in rows])
    num_frames = int(round(span_s * 4.0)) + 1       # plan_windows' 4 fps model clock
    return np.array(plan_windows(n_frames / fps, 4.0, num_frames, stride_s))


def build_targets(t_end: np.ndarray, horizons, conf_frames, tracks, n_frames: int,
                  p: Params, crop_box, ongoing_s: float = 0.5, with_cv: bool = True):
    H, N = len(horizons), len(t_end)
    target = np.zeros((H, N, GRID, GRID), np.uint8)
    cv = np.zeros((H, N, GRID, GRID), np.uint8)
    onset = np.zeros((H, N), bool)
    ongoing = np.zeros(N, bool)
    frame_end = np.minimum(np.floor(t_end * p.fps + 1e-6).astype(int), n_frames - 1)
    conf_sorted = np.array(sorted(conf_frames), dtype=int)
    for i, f in enumerate(frame_end):
        back = int(round(ongoing_s * p.fps))
        ongoing[i] = bool(((conf_sorted > f - back) & (conf_sorted <= f)).any())
        for j, h in enumerate(horizons):
            ahead = conf_sorted[(conf_sorted > f) & (conf_sorted <= f + int(round(h * p.fps)))]
            if len(ahead):
                regions = [c["region"] for g in ahead for c in conf_frames[int(g)]]
                target[j, i] = boxes_to_patch_mask(regions, crop_box, GRID, pad=0.0)
                onset[j, i] = not ongoing[i]
            if with_cv:
                regions = cv_forecast(int(f), h, tracks, p)
                if regions:
                    cv[j, i] = boxes_to_patch_mask(regions, crop_box, GRID, pad=0.0)
    return frame_end, target, onset, ongoing, cv


def cv_lead_times(events, t_end, cv, horizons, crop_box, max_lead_s: float) -> list[float | None]:
    """Per event: how long before onset the constant-velocity forecast (any
    horizon) first marked an overlapping region and kept marking it up to
    onset. None if it never did."""
    any_h = cv.max(axis=0)                           # [N, G, G]
    leads = []
    for e in events:
        region = boxes_to_patch_mask([e["region"]], crop_box, GRID, pad=0.0)
        idx = [i for i, t in enumerate(t_end) if e["onset_s"] - max_lead_s <= t <= e["onset_s"]]
        hit = [bool((any_h[i] & region).any()) for i in idx]
        lead = None
        for i, h in zip(reversed(idx), reversed(hit)):
            if not h:
                break
            lead = e["onset_s"] - t_end[i]
        leads.append(None if lead is None else round(float(lead), 2))
    return leads


def write_plot(path: Path, events, t_end, onset, cv, horizons, n_frames, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 2.8))
    for e in events:
        ax.axvspan(e["onset_s"], e["end_s"] + 1 / fps, ymin=0.7, ymax=0.95,
                   color="#d50000" if e["vehicle_moving"] else "#ff6d00")
    for j, h in enumerate(horizons):
        y = 0.5 - j * 0.15
        ax.scatter(t_end[onset[j]], np.full(onset[j].sum(), y), s=6, c="#7b1fa2", label=f"onset within {h:g} s" if j == 0 else None)
        ax.scatter(t_end[cv[j].reshape(len(t_end), -1).any(1)], np.full(cv[j].reshape(len(t_end), -1).any(1).sum(), y - 0.06),
                   s=6, c="#0288d1", label="constant-velocity forecast" if j == 0 else None)
        ax.text(-2, y - 0.03, f"{h:g} s", ha="right", va="center", fontsize=8)
    ax.set_xlim(0, n_frames / fps); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("clip time (s)")
    ax.set_title("Conflict events (red: vehicle moving, orange: vehicle parked) and window labels")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def write_review(path: Path, video: Path, events, recs, fps: float, crop_box, max_events: int = 60):
    """One row per event: 1.5 s before onset, onset, end. The pair's boxes and
    the event region are drawn; a person confirms or rejects each label (and
    its vehicle_moving flag) at a glance. Stays local: it shows people."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    x0, y0, x1, y1 = crop_box
    rows = []
    for n, e in enumerate(events[:max_events]):
        ids = {i for pr in e["pairs"] for i in pr}
        tiles = []
        for tag, t in (("-1.5s", e["onset_s"] - 1.5), ("onset", e["onset_s"]), ("end", e["end_s"])):
            f = int(np.clip(round(t * fps), 0, len(recs) - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if not ok:
                im = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
            else:
                for o in recs[f]["objects"]:
                    if o["id"] in ids:
                        b = [int(v) for v in o["box"]]
                        col = (0, 0, 255) if o["label"] == "person" else (255, 160, 0)
                        cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), col, 2)
                        cv2.putText(im, f'{o["label"][0]}{o["id"]}', (b[0], b[1] - 5), 0, 0.6, col, 2)
                r = [int(v) for v in e["region"]]
                cv2.rectangle(im, (r[0], r[1]), (r[2], r[3]), (0, 0, 200), 1)
                im = im[y0:y1, x0:x1]
            cv2.putText(im, f'#{n} {tag} {t:.1f}s', (8, 24), 0, 0.7, (0, 255, 255), 2)
            tiles.append(im)
        row = np.hstack(tiles)
        flag = "vehicle MOVING?" if e["vehicle_moving"] else "vehicle parked?"
        cv2.putText(row, f'min_d {e["min_d"]:.2f} m  {flag}', (8, row.shape[0] - 10), 0, 0.7, (0, 255, 255), 2)
        rows.append(row)
    cap.release()
    if rows:
        sheet = np.vstack(rows)
        scale = min(1.0, 1800 / sheet.shape[1])
        cv2.imwrite(str(path), cv2.resize(sheet, None, fx=scale, fy=scale), [cv2.IMWRITE_JPEG_QUALITY, 70])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--fps", type=float, required=True, help="true capture fps of the clip")
    ap.add_argument("--crop-box", type=int, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                    help="model-input crop (default: from --windows-from's summary.json)")
    ap.add_argument("--windows-from", type=Path, default=None,
                    help="vjepa_surprise.py run dir: reuse its window end times and crop box")
    ap.add_argument("--span-s", type=float, default=3.75)
    ap.add_argument("--stride-s", type=float, default=0.5)
    ap.add_argument("--horizons", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--d-conflict", type=float, default=1.8, help="m; STL proximity_emergency")
    ap.add_argument("--closing-gate", type=float, default=0.5, help="m/s; STL closing_gate")
    ap.add_argument("--v-move", type=float, default=0.5, help="m/s; STL v_veh_floor")
    ap.add_argument("--vel-window-s", type=float, default=1.0)
    ap.add_argument("--max-speed", type=float, default=6.0,
                    help="m/s; a track step faster than this is an ID switch and breaks the track")
    ap.add_argument("--require-vehicle-motion", action="store_true")
    ap.add_argument("--all-persons", action="store_true", help="include pedestrians outside the AOI")
    ap.add_argument("--merge-gap-s", type=float, default=1.0)
    ap.add_argument("--max-lead-s", type=float, default=8.0)
    ap.add_argument("--no-cv", action="store_true", help="skip the constant-velocity baseline")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--review-video", type=Path, default=None,
                    help="write review.jpg: three frames per event from this clip, for a human check")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    crop_box = args.crop_box
    if crop_box is None:
        if args.windows_from is None:
            raise SystemExit("pass --crop-box or --windows-from")
        crop_box = json.loads((args.windows_from / "summary.json").read_text())["crop_box"]
    p = Params(args.fps, args.d_conflict, args.closing_gate, args.v_move, args.vel_window_s,
               args.require_vehicle_motion, args.max_speed)

    recs, tracks = load_tracks(args.trace, args.all_persons)
    n = len(recs)
    conf_frames = {}
    for r in recs:
        f = r["frame"]
        c = frame_conflicts(f, present(f, tracks, "ped"), present(f, tracks, "veh"), tracks, p)
        if c:
            conf_frames[f] = c
    events = group_events(conf_frames, args.fps, args.merge_gap_s, args.v_move)

    t_end = window_ends(n, args.fps, args.windows_from, args.span_s, args.stride_s)
    frame_end, target, onset, ongoing, cv = build_targets(
        t_end, args.horizons, conf_frames, tracks, n, p, crop_box, with_cv=not args.no_cv)
    leads = cv_lead_times(events, t_end, cv, args.horizons, crop_box, args.max_lead_s) \
        if not args.no_cv else [None] * len(events)
    for e, lead in zip(events, leads):
        e["cv_lead_s"] = lead

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "labels.npz", t_end_s=t_end, frame_end=frame_end,
                        horizons=np.asarray(args.horizons), target=target, onset=onset,
                        ongoing=ongoing, cv_pred=cv, crop_box=np.asarray(crop_box))
    (args.out / "events.json").write_text(json.dumps(events, indent=1))
    summary = {
        "trace": args.trace.name, "fps": args.fps, "n_frames": n, "crop_box": list(crop_box),
        "grid": GRID, "params": vars(p) | {"k_frames": p.k}, "horizons": args.horizons,
        "windows": {"n": len(t_end), "source": str(args.windows_from) if args.windows_from else "planned",
                    "ongoing": int(ongoing.sum()),
                    "onset_within": {f"{h:g}s": int(onset[j].sum()) for j, h in enumerate(args.horizons)}},
        "events": {"n": len(events), "vehicle_moving": sum(e["vehicle_moving"] for e in events),
                   "conflict_frames": len(conf_frames),
                   "cv_anticipated": sum(e["cv_lead_s"] is not None and e["cv_lead_s"] > 0 for e in events)},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    if args.plot:
        write_plot(args.out / "labels.png", events, t_end, onset, cv, args.horizons, n, args.fps)
    if args.review_video:
        write_review(args.out / "review.jpg", args.review_video, events, recs, args.fps, crop_box)
    print(json.dumps(summary["events"] | summary["windows"], indent=1))
    for e in events:
        print(f'  {e["onset_s"]:6.1f}-{e["end_s"]:6.1f}s  min_d {e["min_d"]:.2f} m  '
              f'closing {e["max_closing"] if e["max_closing"] is None else round(e["max_closing"], 2)}  '
              f'veh {e["max_v_speed"] if e["max_v_speed"] is None else round(e["max_v_speed"], 2)} m/s'
              f'{"  MOVING" if e["vehicle_moving"] else ""}  cv lead {e["cv_lead_s"]}')


if __name__ == "__main__":
    main()
