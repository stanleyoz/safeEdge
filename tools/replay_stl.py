"""
Re-run the safety layer of the edge loop on a recorded trace, without the
detector: signal extraction, trajectory prediction, STL and the intervention
hysteresis, exactly as edge/main.py calls them, fed with the tracked boxes the
trace recorded. A clip that takes minutes to replay through YOLO re-runs here
in seconds, so homographies, signal settings and STL parameters can be swept
and compared on identical detections.

  python tools/replay_stl.py --trace X.trace.jsonl --fps 7.7 \\
      --homography config/homography.npy --stl config/stl_specs.yaml \\
      --out X.restl.trace.jsonl

--check compares the result with the trace's own levels and distances (same
code and settings must reproduce them; boxes are stored to 0.1 px, so tiny
distance differences are expected). The output is a trace in the same format,
so every downstream tool (labels, comparison dashboard, analysis) reads it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402

from edge.detection.signal_extractor import RawDetection, SignalExtractor, extractor_options  # noqa: E402
from edge.safety.intervention import InterventionEngine  # noqa: E402
from edge.safety.stl_monitor import SignalFrame, STLMonitor  # noqa: E402
from edge.safety.trajectory import predict_min_distance  # noqa: E402


def replay(recs: list[dict], H: np.ndarray, fps: float, stl_path: Path,
           upgrade_hold: int = 3, downgrade_hold: int | None = None,
           extractor_kwargs: dict | None = None) -> list[dict]:
    """New trace records from the recorded detections."""
    down = downgrade_hold if downgrade_hold is not None else max(15, int(round(fps * 3)))
    if extractor_kwargs is None:                      # same as the edge loop: the spec's signals section
        extractor_kwargs = extractor_options(yaml.safe_load(open(stl_path)))
    ex = SignalExtractor(H, fps=fps, **extractor_kwargs)
    stl = STLMonitor(stl_path)
    eng = InterventionEngine(upgrade_hold=upgrade_hold, downgrade_hold=down)
    out = []
    f = lambda v: None if v is None else float(v)
    for r in recs:
        dets = [RawDetection(o["id"], o["label"], np.asarray(o["box"], float), 1.0)
                for o in r["objects"] if o["in_aoi"]]
        b = ex.extract(dets)
        d_pred = predict_min_distance(b.pedestrians, b.vehicles, horizon_s=4.0)
        s = stl.update(SignalFrame(t=r["frame"], d_min=b.d_min, v_veh_max=b.v_veh_max, d_pred=d_pred,
                                   alert_active=eng.alert_active, v_closing=b.v_closing))
        eng.process(s)
        ground = {o.track_id: o for o in (*b.pedestrians, *b.vehicles)}
        objects = []
        for o in r["objects"]:
            g = ground.get(o["id"])
            objects.append({**o, "xz": None if g is None else [round(float(g.xyz[0]), 3), round(float(g.xyz[2]), 3)],
                            "speed": None if g is None else round(float(np.linalg.norm(g.vel)), 3)})
        out.append({"frame": r["frame"], "t_s": r["frame"] / fps,
                    "d_min": f(b.d_min), "v_veh_max": f(b.v_veh_max), "v_closing": f(b.v_closing),
                    "d_pred": f(d_pred), "rho1": f(s.rho1), "rho2": f(s.rho2), "rho3": f(s.rho3),
                    "rho4": f(s.rho4), "rho5": f(s.rho5), "rho_min": f(s.rho_min),
                    "level": int(s.intervention_level), "shown": eng._current_level, "objects": objects})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--fps", type=float, required=True, help="the fps the edge loop ran with (--fps)")
    ap.add_argument("--homography", type=Path, required=True)
    ap.add_argument("--stl", type=Path, default=Path("config/stl_specs.yaml"))
    ap.add_argument("--downgrade-hold", type=int, default=None, help="default: the edge loop's max(15, 3 s)")
    ap.add_argument("--check", action="store_true", help="compare with the trace's own levels")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.trace) if line.strip()]
    new = replay(recs, np.load(args.homography), args.fps, args.stl, downgrade_hold=args.downgrade_hold)
    if args.check:
        lv_old = np.array([r["level"] for r in recs]); lv_new = np.array([r["level"] for r in new])
        d_old = np.array([r["d_min"] for r in recs]); d_new = np.array([r["d_min"] for r in new])
        print(f"raw level identical in {np.mean(lv_old == lv_new) * 100:.2f}% of {len(recs)} frames; "
              f"d_min max |diff| {np.abs(d_old - d_new).max():.4f} m")
    if args.out:
        with open(args.out, "w") as fh:
            for r in new:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
