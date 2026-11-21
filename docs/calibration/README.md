# Camera calibration from the scaled site map

The ground-plane homography turns image positions into metres. Every distance,
speed and STL robustness value depends on it, and so will the conflict labels
and the synthetic training clips. The July homography was fitted from assumed
bay sizes and does not describe a physical camera:

| Check | July homography | Expected |
| --- | --- | --- |
| Side-on parked car length | 2.8 m median (2.5-3.4) | 4.2-4.9 m |
| Kerb (row 505) to path near edge (row 690), centre column | 2.3 m | lawn plus path, several metres; read it off the map |
| Horizon row | 132 (in the trees) | where the car park vanishes, near row 400 |
| Two focal-length estimates | disagree by 87 % | agree within a few % |

`july_homography_grid.jpg` shows it: a 1 m ground grid drawn by the July
homography, skewed against the kerb and running up into the trees.

## Procedure

1. **Use the empty-car-park frame** `frame_2026-11-21_empty.png`: the median
   of 31 raw frames across one minute of the live camera (21 Nov 2026), with
   no vehicles or people. The camera has not moved since July (the standpipe
   and path edges sit within about 1 px of the July frame), so one
   calibration serves the July clips and the live feed. The lamp post was
   replaced since July (white globe, now a black pole); its base is a check
   point rather than an anchor. `frame_july_empty.png` is the July view.
2. **Open `docs/tools/map-calibrator.html`** in a browser (directly from disk;
   nothing is uploaded) and load the frame and the scaled site map.
3. **Set the scale**: click two map points a known distance apart (a scale
   bar, or two features with a dimension on the plan) and type the distance.
4. **Add points**: click a ground-level landmark in the frame, then the same
   spot on the map. `point_sheet.jpg` marks seven on the current frame (the
   car park has no painted bay lines, so kerbs and post bases carry the fit):

   | # | Landmark | Pixel (approx.) |
   | --- | --- | --- |
   | 1 | Lamp post base | 824, 513 |
   | 2 | Left kerb-island nose | 598, 500 |
   | 3 | Right kerb-island nose | 987, 503 |
   | 4 | Standpipe centre post base | 518, 604 |
   | 5 | Median island, near-left end | 283, 488 |
   | 6 | Median island, far end | 505, 434 |
   | 7 | Small lamp post base on the median island | 420, 450 |

   Add as many more as the map shows: bay-line ends at the kerb, other post
   bases, kerb corners, drain covers. Ground level only (the base of a post,
   never its top or a car body), spread near and far, left and right. Aim for
   eight or more.
5. **Add lines** for long straight edges whose ends are not visible: A, the
   path's far edge (row about 617); B, its near edge (row about 689); C, the
   right diagonal kerb; D, the far kerb on the left. Click two map points
   along the edge, then two or more frame points anywhere along the same edge,
   and Finish line. Lines add constraint where points are scarce: the
   foreground (A, B) and the far sides (C, D).
6. **Enter the camera mount height** if it is known; it is the strongest
   independent check.
7. Watch the preview: the 1 m grid should sit flat on the lawn and asphalt,
   and the horizon line should fall where the car park vanishes. Export
   `calib.json`.
8. **Run the check** with the physical camera fit (`--camera`: focal length,
   orientation and position, 7 parameters). Kerb landmarks lie along a few
   lines, and a free 8-parameter homography bends to absorb bad pairs; on
   synthetic two-line landmarks with 1.5 px click noise the camera fit maps
   the ground 3-7 times more accurately. `--height-prior-sd` adds the mount
   height as a soft constraint.

   ```bash
   python tools/check_homography.py calib.json --camera --height-prior-sd 0.2 \
       --frame frame.png \
       --trace research/vjepa-surprise/results/traces/segment_0455_to_end.trace.jsonl \
       --old config/homography.npy --out calib_check/
   ```

   Map scale: measure the scale bar to the pixel. On a Google Maps
   screenshot the 5 m bar's end ticks were 142 px apart (0.0352 m/px), 4 %
   off the value set by eye; every distance inherits a scale error. Satellite
   views also hide the ground under tree canopy: drop landmarks placed there.

## Accepting a calibration

| Check | Accept when |
| --- | --- |
| RMS over points and lines | below about 0.3 m |
| Error per landmark, in pixels | below about 5 px (a careful click is good to 2-3 px); judge far points in pixels, since one pixel there spans most of a metre. A larger value marks a mis-click, a mismatched pair or a map mistake: fix or drop that landmark |
| Leave-one-out error per landmark | close to its own error; a landmark that only fits when it is in the fit is suspect |
| Focal consistency | the two focal-length estimates agree within about 5 % |
| Camera height | plausible for the mount, and within about 0.3 m of the known height if given |
| Horizon row | where the ground plane vanishes in the frame |
| Side-on car length | about 4.2-4.9 m median (only valid for the July trace if the camera has not moved) |
| `overlay.png` | grid square to the kerb and path; each red tick (landmark to fitted position) short |
| `topdown.png` | parked cars and bays right-angled and to scale |

The script writes `calib_check/homography.npy` as a candidate and never
touches `config/`. Once accepted, copy it to `config/homography.npy` on the
workstation and the Jetson, re-replay the July clip, and regenerate the trace,
labels and comparison data from it.

The camera model it reports (focal length, mount height, tilt) is also what
the synthetic pipeline uses to place and scale pedestrians and vehicles on
background plates.

## Result: 21 Nov 2026 calibration

`calib_2026-11-21.json` (8 points, 3 lines, Google Maps satellite map at
0.0352 m/px), camera fit with a soft 1.2 m height prior. Outputs in
`check_2026-11-21/`.

| Check | July homography | 21 Nov calibration |
| --- | --- | --- |
| Side-on parked car length | 2.8 m | 4.42 m |
| Horizon row | 132 | 402 (car park vanishes near 400) |
| Focal length / field of view | inconsistent (87 %) | 760 px / 80 degrees, consistent |
| Kerb to path near edge, centre column | 2.3 m | 7.3 m |
| Image reprojection error | n/a | 5.5 px RMS (satellite-map precision) |
| Mount height | 2.9 m implied | 1.53 m (1.2-1.5 m all fit: ground distances barely change) |

Dropped: P8 and P9, placed under tree canopy on the satellite view (24-30 px
off); the first line 3, whose two map points were 0.3 m apart. Remaining
landmark errors of 4-14 px are within what a satellite screenshot allows.
The map image itself is a Google Maps screenshot and is not in the repo.
