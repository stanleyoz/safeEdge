"""
Regenerate the results table from committed run outputs — no video, no weights.

For each run directory (surprise.csv + maps.npy + summary.json) and the
geometric trace it was scored against, reports:

  concurrent AUROC      score vs "window contains WARNING+"
  anticipation AUROC    among fully calm windows, score vs "WARNING+ within
                        the next H s" (H = 2, 4), with a block-bootstrap 95% CI
                        (20 s blocks, 2000 resamples) because positives
                        cluster in a handful of episodes
  far-field rows        the same AUROCs with the score recomputed over the top
                        k rows of the 24 x 24 surprise map only (exploratory:
                        the cut-off was chosen after looking at the data)
  pair / actors         the same AUROCs with the score pooled over the image
                        regions of pedestrian-vehicle pairs, or of pedestrians +
                        moving vehicles (needs a trace with per-object boxes;
                        same functions as tools/vjepa_surprise.py --score pair)

Windows the ridge forecaster was fitted on (role == "fit") are in-sample and
are excluded everywhere, as in tools/vjepa_surprise.py.

  python research/vjepa-surprise/scripts/analyze_runs.py \\
      --trace research/vjepa-surprise/results/traces/segment_0455_to_end.trace.jsonl \\
      --fps 7.7 research/vjepa-surprise/results/{framediff,vitb_*,vitg_predictor}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.vjepa_surprise import (  # noqa: E402
    IMG_SIZE, PATCH, TUBELET, auroc, window_region_scores,
)


def load_trace(path: Path, fps: float):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    objects = [r.get("objects") for r in recs]
    return (np.array([r["frame"] for r in recs]) / fps,
            np.array([r["level"] for r in recs]),
            objects if all(o is not None for o in objects) else None)


def max_level(tt, lv, t0, t1) -> int:
    sel = (tt > t0) & (tt <= t1)
    return int(lv[sel].max()) if sel.any() else 0


def block_ci(scores, labels, times, block_s=20.0, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    blocks = np.floor(times / block_s).astype(int)
    ub = np.unique(blocks)
    members = {b: np.where(blocks == b)[0] for b in ub}
    out = []
    for _ in range(n_boot):
        sel = np.concatenate([members[b] for b in rng.choice(ub, len(ub))])
        a = auroc(scores[sel], labels[sel])
        if a is not None:
            out.append(a)
    return tuple(np.percentile(out, [2.5, 97.5]))


def analyse(run: Path, tt, lv, objects, horizons, far_rows, pair_radius_m, moving_mps):
    summ = json.loads((run / "summary.json").read_text())
    span = summ["window"]["span_s"]
    rows = list(csv.DictReader(open(run / "surprise.csv")))
    t = np.array([float(r["t_end_s"]) for r in rows])
    keep = np.array([r["role"] != "fit" for r in rows])
    calm = keep & np.array([r["calm"] == "1" for r in rows])
    hot = np.array([max_level(tt, lv, x - span, x) >= 2 for x in t])
    ahead = {h: np.array([max_level(tt, lv, x, x + h) >= 2 for x in t]) for h in horizons}

    def score_block(s):
        res = {"concurrent": auroc(s[keep], hot[keep])}
        for h in horizons:
            idx = np.where(calm)[0]
            res[f"antic_{h:g}s"] = auroc(s[idx], ahead[h][idx])
            res[f"antic_{h:g}s_ci"] = block_ci(s[idx], ahead[h][idx], t[idx])
            res[f"antic_{h:g}s_pos"] = int(ahead[h][idx].sum())
        return res

    out = {"run": run.name, "mode": summ["mode"], "model": summ["model"],
           "ms_per_window": summ["timing_ms"]["median"],
           "warning_plus_detected": f'{summ["events"]["warning_plus"]["detected"]}/'
                                    f'{summ["events"]["warning_plus"]["n"]}',
           "false_alarm_runs": summ["false_alarms"],
           "full": score_block(np.array([float(r["s_p95"]) for r in rows]))}
    maps_path = run / "maps.npy"
    if maps_path.exists() and far_rows:
        maps = np.load(maps_path).astype(np.float32)
        s_far = np.percentile(maps[:, :far_rows, :].reshape(len(maps), -1), 95, axis=1)
        out[f"far_rows_0_{far_rows - 1}"] = score_block(s_far)
    if maps_path.exists() and objects is not None:
        w = summ["window"]
        target_span = round(w["target_s"] * w["model_fps"] / TUBELET) * TUBELET / w["model_fps"]
        grid = IMG_SIZE // PATCH
        reg = []
        for x, m in zip(t, maps):
            sel = np.where((tt > x - target_span) & (tt <= x))[0]
            reg.append(window_region_scores(m, [objects[i] for i in sel], tuple(summ["crop_box"]),
                                            grid, pair_radius_m, moving_mps))
        for kind in ("pair", "actors"):
            out[kind] = score_block(np.array([r[kind] for r in reg]))
            out[kind]["windows_with_region"] = int(sum(r[f"{kind}_patches"] > 0 for r in reg))
    return out


def fmt(res, key, h):
    a, (lo, hi) = res[key][f"antic_{h:g}s"], res[key][f"antic_{h:g}s_ci"]
    return f"{a:.3f} [{lo:.2f}, {hi:.2f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--fps", type=float, required=True, help="true capture fps of the clip")
    ap.add_argument("--far-rows", type=int, default=8)
    ap.add_argument("--pair-radius-m", type=float, default=5.0)
    ap.add_argument("--moving-mps", type=float, default=0.5)
    ap.add_argument("--json", type=Path, default=None, help="also write all numbers here")
    args = ap.parse_args()

    tt, lv, objects = load_trace(args.trace, args.fps)
    results = [analyse(r, tt, lv, objects, (2.0, 4.0), args.far_rows,
                       args.pair_radius_m, args.moving_mps) for r in args.runs]

    far = f"far_rows_0_{args.far_rows - 1}"
    print("| Run | Concurrent AUROC | Anticipation 2 s [95% CI] | Anticipation 4 s [95% CI] "
          f"| Far-field ({far}) antic. 2 s | WARNING+ raised | False-alarm runs | ms/window |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        ff = f'{r[far]["antic_2s"]:.3f}' if far in r else "-"
        print(f'| {r["run"]} | {r["full"]["concurrent"]:.3f} | {fmt(r, "full", 2)} | '
              f'{fmt(r, "full", 4)} | {ff} | {r["warning_plus_detected"]} | '
              f'{r["false_alarm_runs"]} | {r["ms_per_window"]:.0f} |')
    if objects is not None:
        print()
        print("| Run | Pooling | Concurrent AUROC | Anticipation 2 s [95% CI] "
              "| Anticipation 4 s [95% CI] | Windows with region |")
        print("| --- | --- | --- | --- | --- | --- |")
        for r in results:
            for key, name in (("full", "frame p95"), ("pair", "pair-local"), ("actors", "actors")):
                if key not in r:
                    continue
                nwin = r[key].get("windows_with_region", "all")
                print(f'| {r["run"]} | {name} | {r[key]["concurrent"]:.3f} | '
                      f'{fmt(r, key, 2)} | {fmt(r, key, 4)} | {nwin} |')
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
