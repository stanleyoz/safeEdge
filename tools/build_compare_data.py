"""
Pack one replayed clip into a single JSON for the STL-vs-V-JEPA dashboard
(dashboard/static/compare.html).

Inputs are things the repo already produces, so nothing here re-runs a model:

  --trace   per-frame JSONL from `python -m edge.main --source file --trace ...`
            (signals, rho1-rho5, raw STL level, tracked boxes)
  runs      one or more tools/vjepa_surprise.py output directories
            (surprise.csv, summary.json, maps.npy)

The STL channel is the raw level pushed through the edge loop's own
InterventionEngine with the holds used on the day, so the page shows exactly
the WARNING/EMERGENCY the live dashboard showed. Surprise maps are stored as
uint8, coloured relative to each patch's calm baseline as in write_overlay.

The video itself is not embedded: the page plays it from a local file. Its
container fps (often not the true capture fps) is recorded so the page can map
video time to trace frames.

  python tools/build_compare_data.py \\
      --trace research/vjepa-surprise/results/traces/segment_0455_to_end.trace.jsonl \\
      --fps 7.7 --video data/segment_0455_to_end.mp4 --downgrade-hold 23 \\
      --out research/vjepa-surprise/results/compare/segment_0455_to_end.json \\
      research/vjepa-surprise/results/{vitg_predictor,vitb_ridge_crossseason,framediff}
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.safety.intervention import InterventionEngine  # noqa: E402

LABELS = {"person": 0, "car": 1}   # anything else -> 2
RUN_NAMES = {("giant", "predictor"): "V-JEPA 2.1 ViT-g predictor",
             ("gigantic", "predictor"): "V-JEPA 2.1 ViT-G predictor",
             ("base", "ridge"): "V-JEPA 2.1 ViT-B + ridge",
             ("large", "ridge"): "V-JEPA 2.1 ViT-L + ridge",
             ("base", "persistence"): "V-JEPA 2.1 ViT-B persistence"}


def r3(v):
    return None if v is None else round(float(v), 3)


def debounce(recs: list[dict], upgrade_hold: int, downgrade_hold: int):
    """Raw per-frame STL level -> the level the dashboard showed, plus its
    events, using the edge loop's InterventionEngine unchanged."""
    events = []
    eng = InterventionEngine(upgrade_hold=upgrade_hold, downgrade_hold=downgrade_hold)
    out = []
    for r in recs:
        sig = SimpleNamespace(d_min=r["d_min"], v_veh_max=r["v_veh_max"], d_pred=r["d_pred"])
        state = SimpleNamespace(intervention_level=r["level"], rho_min=r["rho_min"],
                                rho1=r["rho1"], rho2=r["rho2"], rho3=r["rho3"],
                                rho4=r["rho4"], rho5=r["rho5"], signals=sig)
        ev = eng.process(state)
        if ev is not None:
            events.append({"frame": r["frame"], "level": ev.new_level, "message": ev.message})
        out.append(eng._current_level)
    return out, events


def pack_run(run: Path) -> dict:
    summ = json.loads((run / "summary.json").read_text())
    rows = list(csv.DictReader(open(run / "surprise.csv")))
    key = "s_" + summ["score"] if "s_" + summ["score"] in rows[0] else "s_p95"
    tau = summ["calibration"]["tau"]
    out = {
        "id": run.name,
        "name": RUN_NAMES.get((summ["model"], summ["mode"]),
                              "Pixel frame difference" if summ["mode"] == "framediff"
                              else f'{summ["model"]} {summ["mode"]}'),
        "mode": summ["mode"], "score": key, "tau": tau,
        "alpha": summ["calibration"]["alpha"],
        "span_s": summ["window"]["span_s"], "target_s": summ["window"]["target_s"],
        "crop_box": summ["crop_box"],
        "auroc": summ.get("window_auroc", {}),
        "ms_per_window": summ.get("timing_ms", {}).get("median"),
        "device": summ.get("timing_ms", {}).get("device"),
        "t_end_s": [round(float(r["t_end_s"]), 3) for r in rows],
        "s": [round(float(r[key]), 5) for r in rows],
        "role": [r["role"] for r in rows],
    }
    maps_path = run / "maps.npy"
    if maps_path.exists():
        maps = np.load(maps_path).astype(np.float32)
        calib = [i for i, r in enumerate(rows) if r["role"] == "calib"] or list(range(len(rows)))
        base = np.median(maps[calib], axis=0)
        floor = float(np.median([out["s"][i] for i in calib]))
        rise = np.maximum(maps - base, 0)
        scale = (tau - floor) if tau > floor else float(np.percentile(rise, 99)) or 1.0
        q = (np.clip(rise / (2 * scale), 0, 1) * 255).astype(np.uint8)
        out["maps"] = {"shape": list(q.shape), "b64": base64.b64encode(q.tobytes()).decode()}
    return out


def video_meta(path: Path | None, width: int, height: int, container_fps: float | None):
    if path is not None:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        meta = {"file": path.name, "container_fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        cap.release()
        return meta
    if container_fps is None:
        raise SystemExit("pass --video, or --container-fps (with --width/--height)")
    return {"file": None, "container_fps": container_fps, "width": width, "height": height}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", type=Path, help="vjepa_surprise.py output dirs")
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--fps", type=float, required=True, help="true capture fps of the clip")
    ap.add_argument("--video", type=Path, default=None, help="read container fps and size from it")
    ap.add_argument("--container-fps", type=float, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--upgrade-hold", type=int, default=3)
    ap.add_argument("--downgrade-hold", type=int, default=23,
                    help="frames; the July demo ran with SAFEEDGE_DOWNGRADE_HOLD=23")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.trace) if line.strip()]
    level, events = debounce(recs, args.upgrade_hold, args.downgrade_hold)

    def objects(r):
        return [[o["id"], LABELS.get(o["label"], 2), int(o["in_aoi"]),
                 *[round(v) for v in o["box"]]] for o in r.get("objects") or []]

    data = {
        "clip": {"fps": args.fps, "n_frames": len(recs), "trace": args.trace.name,
                 **video_meta(args.video, args.width, args.height, args.container_fps)},
        "stl": {"upgrade_hold": args.upgrade_hold, "downgrade_hold": args.downgrade_hold,
                "events": events},
        "frames": {
            "level_raw": [r["level"] for r in recs],
            "level": level,
            **{k: [r3(r.get(k)) for r in recs]
               for k in ("d_min", "v_veh_max", "v_closing", "d_pred",
                         "rho1", "rho2", "rho3", "rho4", "rho5")},
            "objects": [objects(r) for r in recs],
        },
        "runs": [pack_run(r) for r in args.runs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"{args.out}  {args.out.stat().st_size / 1e6:.1f} MB  "
          f"{len(recs)} frames, {len(events)} STL events, {len(data['runs'])} runs")


if __name__ == "__main__":
    main()
