"""Measure whether the real game's fall is constant speed or accelerating, and compare with the sim's GRAVITY.

`GRAVITY = 2800.0` has been there since the move to pymunk (2026-08-01), and there is no record anywhere
of comparing it with the real game. What looks right by eye is **the time the fall takes**,
not the speed profile. The same 0.60 seconds is 833px/s throughout at constant speed,
and 1673px/s at landing when accelerating, a 2x difference.

What it does: pick up the y of the falling fruit with timestamps, and

    constant   y = y0 + v*t
    accelerating   y = y0 + v0*t + 0.5*g*t^2

fit both and compare the residuals. If acceleration wins, g is the real game's gravity as is.

**Empty the board and drop just one fruit.** With fruits on the board the falling one gets mixed up.
One fall all the way down the board is the best data.

Detection (`detect_fruits`) takes 33ms per frame and cannot keep up with the capture rate, so
while capturing it only warps and stores (0.1ms), and detection runs after capturing.

Usage:
  python scripts/measure_fall_speed.py            # capture 3 seconds and fit
  python scripts/measure_fall_speed.py --seconds 5
  python scripts/measure_fall_speed.py --fps 120 --csv artifacts/fall.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT
from src.game import capture as capture_mod
from src.sim.sim_physics import GRAVITY
from src.vision.board import _warp, localize
from src.vision.classify import fruit_radius
from src.vision.colors import FRUIT_NAMES
from src.vision.fruits import detect as detect_fruits
from src.vision.state import Fruit

# The band at the top considered the start of the fall (normalized px). Only fruits appearing here are tracked.
START_Y = 130.0
# A fruit sticking out of the top edge appears with a clipped circle, so neither center nor radius is reliable. Only points
# deeper than the radius are used. The clipping effect is proportional to the radius, so mixing them shifts the initial speed
# by fruit size, and gravity, which trades off against the initial speed, also wobbles with size.
FULLY_INSIDE_MARGIN = 2.0
# The minimum fall distance taken as a trajectory. On a short arc the curvature is buried in noise and cannot be told from constant speed
# (measured: 13 points over y 11->124 gave a residual ratio of 2.15, undecidable).
MIN_SPAN = 220.0
MIN_POINTS = 10
# Downward movement tracking allows per frame. The max speed on the real game plus margin.
MAX_SPEED = 2400.0
STEP_SLACK = 15.0
# Only allow as much upward overshoot per frame as detection flicker causes.
UP_SLACK = 4.0
# Even if detection drops out, this many consecutive frames are bridged by extrapolation.
MAX_MISSES = 5
# Landing check. Looked at only after the fall is under way (after falling LAND_AFTER px). At the start of the fall
# the accelerating model does not move even 1px per frame, so checking there cuts it off immediately.
LAND_AFTER = 100.0
# Landed when the speed drops below this fraction of the max so far. With an absolute value (2px/frame)
# it was buried in detection noise, and points bouncing on the floor stayed in the trajectory and broke the fit.
LAND_FRACTION = 0.35
# Time window for measuring speed. Measuring over a 1-frame difference makes the noise effect depend on the capture rate
# (at 120Hz with 4px noise it wobbles 480px/s and was misjudged as landing mid-fall). Fixing it in seconds
# keeps the noise effect the same even when the rate changes.
SPEED_WINDOW = 0.04

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0, help="seconds to capture")
    parser.add_argument(
        "--fps",
        type=int,
        default=120,
        help="frame rate requested from dxcam (fitting uses actual timestamps, so a cap is fine)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/fall.csv"),
        help="where the measured (t, y) are appended. Always kept by default",
    )
    return parser.parse_args()


def _grab_burst(seconds: float, fps: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Localize the board once and fix the corners, then just warp and store."""
    capture_mod.CAPTURE_FPS = fps

    frame = capture_mod.capture()
    while frame is None:
        frame = capture_mod.capture()
    result = localize(frame, None)
    if not result.found or result.corners is None:
        raise SystemExit("board not found. Bring the game screen to the front before running")

    corners = result.corners
    times: list[float] = []
    boards: list[np.ndarray] = []
    # For checking identity with the previous frame. dxcam can return the same frame.
    previous: np.ndarray | None = None

    print(f"capturing {seconds:.1f} seconds. Drop one fruit now.")
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        frame = capture_mod.capture()
        if frame is None:
            continue
        now = time.perf_counter() - start
        board = _warp(frame, corners)
        # Storing without deduplication makes duplicate frames pull the fit toward constant speed.
        # A coarsely thinned comparison does not work: with `board[::16, ::16]` (32x25) a grape of radius 26
        # moving a few px shows no difference, real new frames were thrown away and it fell to 19Hz.
        # Checking every pixel takes only about 0.05ms and does not affect the capture rate.
        if previous is not None and np.array_equal(board, previous):
            continue
        previous = board
        times.append(now)
        boards.append(board)

    print(f"  {len(boards)} frames (about {len(boards) / seconds:.0f} Hz)")
    return np.asarray(times), boards


def _detect(times: np.ndarray, boards: list[np.ndarray]) -> list[tuple[float, list[Fruit]]]:
    """Run detection over the captured boards in order. It cannot keep up while capturing, so it runs here in bulk."""
    return [(float(t), list(detect_fruits(board))) for t, board in zip(times, boards)]


def _trim_landing(ts: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Discard points after landing.

    Including a fruit resting on the floor makes the flat stretch drag the fit and the acceleration disappears
    (in synthetic data, just 3 points left at the end made the residual jump to 16.8px). Cut at the first point where the speed
    falls below LAND_FRACTION of the max so far.
    """
    if len(ys) < 3:
        return ts, ys

    def speed_at(i: int) -> float | None:
        """The speed at index i. Measured looking back SPEED_WINDOW seconds."""
        for j in range(i - 1, -1, -1):
            if ts[i] - ts[j] >= SPEED_WINDOW:
                return (ys[i] - ys[j]) / (ts[i] - ts[j])
        return None

    peak = 0.0
    for i in range(1, len(ys)):
        speed = speed_at(i)
        if speed is None:
            continue
        if ys[i] - ys[0] > LAND_AFTER and peak > 0.0 and speed < LAND_FRACTION * peak:
            return ts[:i], ys[:i]
        peak = max(peak, speed)
    return ts, ys


def _track_from(
    frames: list[tuple[float, list[Fruit]]], start: int, fruit: Fruit
) -> tuple[list[float], list[float]]:
    """Track one fruit by continuity. Returns from the start of the fall to landing.

    'The topmost fruit every frame' cannot track it. Right after dropping, the next fruit appears at the top of the board,
    so the moment that one becomes topmost y jumps back and the trajectory breaks (measured:
    cut off at y=124). Among the same type, pick the one closest to the position extrapolated
    from the previous velocity.

    Stop once it lands. Including the time resting on the floor makes the flat stretch drag the fit
    and the acceleration disappears.
    """
    ts = [frames[start][0]]
    ys = [fruit.y]
    misses = 0
    for t, fruits in frames[start + 1 :]:
        dt = t - ts[-1]
        if dt <= 0.0:
            continue
        # Extrapolate with the velocity from the last 2 points. With only 1 point, start from rest.
        v = (ys[-1] - ys[-2]) / (ts[-1] - ts[-2]) if len(ys) >= 2 else 0.0
        predicted = ys[-1] + v * dt
        reach = STEP_SLACK + MAX_SPEED * dt
        best: Fruit | None = None
        for candidate in fruits:
            if candidate.type != fruit.type:
                continue
            if candidate.y < ys[-1] - UP_SLACK or candidate.y > ys[-1] + reach:
                continue
            if candidate.y < candidate.radius + FULLY_INSIDE_MARGIN:
                continue
            if best is None or abs(candidate.y - predicted) < abs(best.y - predicted):
                best = candidate
        if best is None:
            # If lost for just 1 frame, bridge it by extrapolation. If it continues, end tracking.
            misses += 1
            if misses > MAX_MISSES:
                break
            continue
        misses = 0
        ts.append(t)
        ys.append(best.y)
    return _trim_landing(ts, ys)


def _trajectory(
    frames: list[tuple[float, list[Fruit]]],
) -> tuple[np.ndarray, np.ndarray, Fruit]:
    """Track every fruit that appears at the top edge, and return the trajectory that fell farthest.

    Also returns which fruit was tracked. Needed later to see whether gravity changes with fruit size.
    """
    best: tuple[list[float], list[float], Fruit] | None = None
    for i, (_t, fruits) in enumerate(frames):
        for fruit in fruits:
            if fruit.y > START_Y:
                continue
            if fruit.y < fruit.radius + FULLY_INSIDE_MARGIN:
                continue
            ts, ys = _track_from(frames, i, fruit)
            if best is None or (ys[-1] - ys[0]) > (best[1][-1] - best[1][0]):
                best = (ts, ys, fruit)

    if best is None:
        raise SystemExit(
            f"no fruit starting to fall at the top edge (y < {START_Y:.0f})."
            "Drop after capture starts. Empty the board"
        )
    ts, ys, fruit = best
    span = ys[-1] - ys[0]
    if len(ys) < MIN_POINTS or span < MIN_SPAN:
        raise SystemExit(
            f"trajectory too short ({len(ys)} points / {span:.0f}px fallen)."
            f"needs {MIN_POINTS} points and {MIN_SPAN:.0f}px."
            "Empty the board and capture again one fall all the way from top to bottom"
        )
    # Shift the start of the fall to t=0, so the intercept means the same on the constant and accelerating sides.
    return np.asarray(ts) - ts[0], np.asarray(ys), fruit


def _fit(t: np.ndarray, y: np.ndarray) -> None:
    """Fit constant speed and acceleration and compare.

    A quadratic contains the linear model, so its residual is always smaller. 'Which is smaller' does not
    decide it; **how much smaller** does. On synthetic data (30/60/120Hz,
    detection noise 0.5/2.0px), true acceleration gave residual ratios of 18-102x and recovered gravity 2739-2818,
    and true constant speed gave a ratio of 1.0x and gravity near 0.
    """
    linear = np.polyfit(t, y, 1)
    quad = np.polyfit(t, y, 2)
    rms_linear = float(np.sqrt(np.mean((np.polyval(linear, t) - y) ** 2)))
    rms_quad = float(np.sqrt(np.mean((np.polyval(quad, t) - y) ** 2)))
    ratio = rms_linear / max(rms_quad, 1e-9)
    g = float(quad[0]) * 2.0

    print()
    print(f"{len(t)} points / {t[-1]:.3f} s / y {y[0]:.1f} -> {y[-1]:.1f} px")
    print(f"  constant fit  y = {linear[1]:.1f} + {linear[0]:.1f}*t")
    print(f"      residual RMS {rms_linear:6.2f} px")
    print(f"  accel fit  y = {quad[2]:.1f} + {quad[1]:.1f}*t + {quad[0]:.1f}*t^2")
    print(f"      residual RMS {rms_quad:6.2f} px, gravity {g:.0f}")

    # Mean speed of the first and second halves. For a fall from rest, free fall gives 3.0x.
    # But the real game releases above the board's top edge, so it already has speed when it becomes visible
    # (measured 557px/s). The ratio shrinks by that much, so it is not used as the basis for the verdict.
    half = len(t) // 2
    v_first = (y[half] - y[0]) / max(t[half] - t[0], 1e-9)
    v_second = (y[-1] - y[half]) / max(t[-1] - t[half], 1e-9)
    speed_ratio = v_second / max(v_first, 1e-9)

    # Decide by whether the quadratic coefficient can be told from 0. If a in y = y0 + v0*t + a*t^2 is
    # significantly positive, it accelerates. The test holds even when the initial speed v0 is not 0, and unlike
    # looking at the ratio it is unaffected by 'released above the top edge'.
    design = np.vstack([t * t, t, np.ones_like(t)]).T
    covariance = np.linalg.inv(design.T @ design) * rms_quad**2
    se_a = float(np.sqrt(covariance[0, 0]))
    se_v0 = float(np.sqrt(covariance[1, 1]))
    t_value = float(quad[0]) / max(se_a, 1e-9)

    print()
    print(f"  residual ratio (const/accel)   {ratio:6.2f}x")
    print(f"  speed first→second half      {v_first:.0f} -> {v_second:.0f} px/s = {speed_ratio:.2f}x")
    print(f"  initial speed v0              {float(quad[1]):.0f} ± {se_v0:.0f} px/s"
          f"   (if not 0, released above the board's top edge)")
    print(f"  t value of quadratic coef       {t_value:6.1f}   (|t| > 4 distinguishes it from 0 = accelerating)")
    print()

    # Check whether the fit holds up physically. The second real measurement gave gravity 2733,
    # a plausible value, but the same fit's initial speed was -983px/s (the fruit flying upward)
    # with a speed ratio of 11.4x. Free fall caps the first/second half ratio at 3.0, so
    # both are evidence of a broken trajectory. Looking only at how plausible the numbers are
    # picks up false support.
    v0 = float(quad[1])
    problems: list[str] = []
    if v0 < -60.0:
        problems.append(f"initial speed is {v0:.0f}px/s (the dropped fruit is flying upward)")
    if speed_ratio > 3.5:
        problems.append(f"first/second half ratio is {speed_ratio:.2f}x (3.0 is the cap even for free fall)")
    if rms_quad > 12.0:
        problems.append(f"residual {rms_quad:.1f}px even with the accelerating fit (detection is erratic)")

    if problems:
        print("  => undecidable. The trajectory is broken:")
        for problem in problems:
            print(f"       - {problem}")
        print("     empty the board and capture again one fall all the way from the top edge to the floor")
        print("     (dumping (t, y) with --csv and looking directly is faster)")
        return

    if t_value > 4.0:
        print(f"  => accelerating. Real game gravity is {g:.0f} ± {2 * se_a:.0f}")
        print(f"     difference from the sim's GRAVITY = {GRAVITY:.0f} is "
              f"{abs(g - GRAVITY) / max(2 * se_a, 1e-9):.1f}σ")
    else:
        print(f"  => falling at constant speed. Speed is {linear[0]:.0f} px/s")
        print("     the sim is an accelerating model, so this would mean removing GRAVITY for a terminal velocity")

    # Compare on the same footing as the sim. Computing the fall time from rest sqrt(2*span/G) makes the sim look slower
    # by as much as the real game has initial speed, giving the opposite conclusion. Compute how many seconds the sim takes
    # to fall the same distance from the same initial speed.
    span = y[-1] - y[0]
    sim_seconds = (-v0 + np.sqrt(v0 * v0 + 2 * GRAVITY * span)) / GRAVITY
    print()
    print(f"  reference: falling {span:.0f}px took {t[-1]:.3f} s measured.")
    print(f"        the sim from the same initial speed {v0:.0f}px/s takes {sim_seconds:.3f} s"
          f" ({t[-1] / max(sim_seconds, 1e-9):.2f}x as fast)")


def _append_run(path: Path, t: np.ndarray, y: np.ndarray, fruit: Fruit) -> int:
    """Append the measured trajectory with the run number and fruit type.

    It used to overwrite, so measuring 3 runs kept only the last one. Without the type,
    whether gravity wobbles with fruit size could not be checked either.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    run = 1
    if path.exists():
        previous = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if previous.size:
            run = int(previous[:, 0].max()) + 1
    else:
        path.write_text("run,type,radius,t,y\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        for ti, yi in zip(t, y):
            # repr writes numpy type names, so convert to float. Digits are not rounded.
            handle.write(
                f"{run},{fruit.type},{fruit.radius!r},{float(ti)!r},{float(yi)!r}\n"
            )
    return run


def _shared_gravity(
    runs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    """Fit with one gravity shared across all runs.

    One run at a time, initial speed and gravity trade off on short arcs and it does not settle
    (5 runs scattered over 1313-1488). Sharing only gravity while leaving initial speed and intercept free per run
    averages out that trade-off and settles much better.

    With gravity fixed, y - g*t^2/2 is linear in t, so for each scanned gravity
    just solve per-run linear least squares and sum the residuals.
    """
    grid = np.linspace(600.0, 3000.0, 1201)
    total = np.empty_like(grid)
    for k, g in enumerate(grid):
        summed = 0.0
        points = 0
        for t, y in runs:
            basis = np.vstack([t, np.ones_like(t)]).T
            target = y - 0.5 * g * t * t
            coef, *_ = np.linalg.lstsq(basis, target, rcond=None)
            summed += float(((basis @ coef - target) ** 2).sum())
            points += len(t)
        total[k] = summed / points
    best = int(np.argmin(total))
    # Not twice the minimum residual… but the width where the per-point variance grows by one standard error.
    dof = sum(len(t) for t, _ in runs) - 2 * len(runs) - 1
    threshold = total[best] * (1.0 + 2.0 / max(dof, 1))
    inside = grid[total <= threshold]
    return float(grid[best]), float((inside.max() - inside.min()) / 2.0)


def _summarize(path: Path) -> None:
    """List the stored runs and see whether gravity agrees and whether it changes with size."""
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    runs = sorted({int(r) for r in rows[:, 0]})
    if len(runs) < 2:
        return
    print()
    print(f"=== {len(runs)} runs stored in {path} ===")
    print(f"{'run':>4} {'fruit':>10} {'radius':>6} {'pts':>4} {'fall px':>7} {'v0':>7} {'gravity':>13}")
    per_type: dict[int, list[float]] = {}
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for run in runs:
        sel = rows[rows[:, 0] == run]
        fruit_type = int(sel[0, 1])
        radius = float(sel[0, 2])
        t, y = sel[:, 3], sel[:, 4]
        if len(t) < MIN_POINTS:
            print(f"{run:4d} {'':>10} {radius:6.1f} {len(t):4d}   (not enough points)")
            continue
        quad = np.polyfit(t, y, 2)
        residual = float(np.sqrt(np.mean((np.polyval(quad, t) - y) ** 2)))
        design = np.vstack([t * t, t, np.ones_like(t)]).T
        se_a = float(np.sqrt(np.linalg.inv(design.T @ design)[0, 0])) * residual
        gravity = float(quad[0]) * 2.0
        per_type.setdefault(fruit_type, []).append(gravity)
        series.append((t, y))
        print(f"{run:4d} {FRUIT_NAMES[fruit_type]:>10} {radius:6.1f} {len(t):4d}"
              f" {y[-1] - y[0]:7.0f} {float(quad[1]):7.0f} {gravity:6.0f} ± {2 * se_a:4.0f}")

    if len(per_type) > 1:
        print()
        print("  by fruit size:")
        for fruit_type in sorted(per_type):
            values = per_type[fruit_type]
            spread = f" (range {max(values) - min(values):.0f})" if len(values) > 1 else ""
            print(f"    {FRUIT_NAMES[fruit_type]:>10} radius {fruit_radius(fruit_type):5.1f}"
                  f"  n={len(values)}  gravity mean {np.mean(values):.0f}{spread}")
        # Only cherry-orange can be drawn, with radii of just 14.2-38.5. With the same release
        # height, the systematic difference from the fall until fully inside the board is about 7,
        # buried in the per-run scatter of 71-97 (confirmed with 200 synthetic runs). The differences
        # here are mostly noise, so do not read too much into them.
        print("    (the systematic difference over this range is about 7, buried in the per-run scatter of 71-97,")
        print("     so differences here are mostly noise. More runs cannot separate them)")

    if len(series) >= 2:
        gravity, error = _shared_gravity(series)
        print()
        print(f"  fitting with gravity shared over all {len(series)} runs: {gravity:.0f} ± {error:.0f}")
        print(f"    ratio to the sim's GRAVITY = {GRAVITY:.0f}: {GRAVITY / gravity:.2f}x")


def main() -> None:
    args = _parse_args()
    times, boards = _grab_burst(args.seconds, args.fps)
    t, y, fruit = _trajectory(_detect(times, boards))
    path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    run = _append_run(path, t, y, fruit)
    print(f"  -> {path} (run {run}, {FRUIT_NAMES[fruit.type]})")
    _fit(t, y)
    _summarize(path)


if __name__ == "__main__":
    main()
