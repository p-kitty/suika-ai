"""実機の落下が等速か加速かを測り、sim の GRAVITY と突き合わせる。

`GRAVITY = 2800.0` は pymunk 化 (2026-08-01) のときに入ったきりで、実機と
突き合わせた記録がどこにも無い。目で見て合っているのは落下に**かかる時間**
であって、速度のプロファイルではない。同じ 0.60 秒でも、等速なら終始
833px/s、加速なら着地 1673px/s と 2 倍違う。

やること: 落下中の実の y を時刻つきで拾い、

    等速   y = y0 + v*t
    加速   y = y0 + v0*t + 0.5*g*t^2

の両方を当てて残差を比べる。加速が勝てば g がそのまま実機の重力になる。

**盤を空にして 1 個だけ落とすこと。** 盤に実があると落下中の実を取り違える。
盤いっぱいまで落ちる 1 本が、いちばん条件の良いデータになる。

検出 (`detect_fruits`) は 1 フレーム 33ms かかって撮影レートに間に合わないので、
撮る間は warp だけして貯め (0.1ms)、検出は撮り終えてから回す。

用法:
  python scripts/measure_fall_speed.py            # 3 秒撮って当てる
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

# 落下の始まりと見なす上端の帯 (正規化 px)。ここに現れた実だけを追い始める。
START_Y = 130.0
# 上端からはみ出た実は円が欠けて写るので、中心も半径も当てにならない。半径より
# 深く入った点だけ使う。欠けの影響は半径に比例するので、混ぜると実の大きさで
# 初速がずれ、初速と取り合いになっている重力まで大きさ依存でぶれる。
FULLY_INSIDE_MARGIN = 2.0
# 軌跡として採る最小の落下距離。短い弧は曲がりがノイズに埋もれて等速と区別できない
# (実測: y 11->124 の 13 点は残差比 2.15 で判定不能だった)。
MIN_SPAN = 220.0
MIN_POINTS = 10
# 追跡が 1 フレームで許す下方向の移動。実機で出る最大速度に余裕を足したもの。
MAX_SPEED = 2400.0
STEP_SLACK = 15.0
# 検出のちらつきで 1 フレーム上振れするぶんだけは許す。
UP_SLACK = 4.0
# 検出がこぼれても、これだけの連続フレームは外挿で跨ぐ。
MAX_MISSES = 5
# 着地の判定。落下が乗ってから (LAND_AFTER px 落ちてから) 見る。落ち始めは
# 加速モデルだと 1 フレームで 1px も動かないので、そこに当てると即打ち切りになる。
LAND_AFTER = 100.0
# 速さがそれまでの最大のこの割合を割ったら着地。絶対値 (2px/フレーム) で見ると
# 検出の雑音に埋もれ、床で跳ねた点が軌跡に残って当てはめを壊した。
LAND_FRACTION = 0.35
# 速さを測る時間窓。1 フレーム差で測ると、雑音の効き方が撮影レートで変わる
# (120Hz・雑音 4px だと 480px/s ぶれ、落下の途中で着地と誤判定した)。秒で
# 決め打てば、レートが変わっても雑音の効きは同じになる。
SPEED_WINDOW = 0.04

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0, help="撮影する秒数")
    parser.add_argument(
        "--fps",
        type=int,
        default=120,
        help="dxcam に要求するフレームレート (実際に出た時刻で当てるので上限でよい)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/fall.csv"),
        help="測った (t, y) を追記する先。既定で必ず残す",
    )
    return parser.parse_args()


def _grab_burst(seconds: float, fps: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """盤を 1 度だけ localize して隅を固定し、あとは warp だけして貯める。"""
    capture_mod.CAPTURE_FPS = fps

    frame = capture_mod.capture()
    while frame is None:
        frame = capture_mod.capture()
    result = localize(frame, None)
    if not result.found or result.corners is None:
        raise SystemExit("盤が見つからない。ゲーム画面を前面に出してから実行する")

    corners = result.corners
    times: list[float] = []
    boards: list[np.ndarray] = []
    # 直前フレームとの同一判定用。dxcam は同じフレームを返すことがある。
    previous: np.ndarray | None = None

    print(f"{seconds:.1f} 秒撮る。いま 1 個落として。")
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        frame = capture_mod.capture()
        if frame is None:
            continue
        now = time.perf_counter() - start
        board = _warp(frame, corners)
        # 間引かずに貯めると、重複フレームが等速側へ寄せた当てはめを作る。
        # 粗く間引いた比較では駄目で、`board[::16, ::16]` (32x25) だと半径 26 の
        # グレープが数 px 動いても差が出ず、本物の新規フレームを捨てて 19Hz まで
        # 落ちた。全画素で見ても 0.05ms 程度で、撮影レートには効かない。
        if previous is not None and np.array_equal(board, previous):
            continue
        previous = board
        times.append(now)
        boards.append(board)

    print(f"  {len(boards)} フレーム ({len(boards) / seconds:.0f} Hz 相当)")
    return np.asarray(times), boards


def _detect(times: np.ndarray, boards: list[np.ndarray]) -> list[tuple[float, list[Fruit]]]:
    """撮り終えた盤を順に検出する。撮影中は間に合わないのでここでまとめて回す。"""
    return [(float(t), list(detect_fruits(board))) for t, board in zip(times, boards)]


def _trim_landing(ts: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """着地してからの点を捨てる。

    床で止まった実まで含めると、平らな区間が当てはめを引きずって加速が消える
    (合成で、末尾に 3 点残るだけで残差が 16.8px まで荒れた)。速さがそれまでの
    最大の LAND_FRACTION を割った最初の点で切る。
    """
    if len(ys) < 3:
        return ts, ys

    def speed_at(i: int) -> float | None:
        """i 番目の速さ。SPEED_WINDOW 秒ぶん遡って測る。"""
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
    """1 個の実を連続性で追う。落ち始めから着地までを返す。

    「毎フレームいちばん上の実」では追えない。落とした直後に次の実が盤の上端へ
    現れるので、そちらが最上位になった瞬間に y が戻って軌跡が切れる (実測で
    y=124 で打ち切られた)。同じ type の中から、直前の速度で外挿した位置に
    いちばん近いものを選ぶ。

    着地したら止める。床で静止したぶんまで含めると、平らな区間が当てはめを
    引きずって加速が消える。
    """
    ts = [frames[start][0]]
    ys = [fruit.y]
    misses = 0
    for t, fruits in frames[start + 1 :]:
        dt = t - ts[-1]
        if dt <= 0.0:
            continue
        # 直前 2 点から速度を測って外挿する。1 点しかなければ静止から始める。
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
            # 1 フレーム見失っただけなら外挿で跨ぐ。続くようなら追跡を終える。
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
    """上端に現れた実を片端から追って、いちばん長く落ちた軌跡を返す。

    どの実を追ったかも返す。重力が実の大きさで変わるかを後から見るのに要る。
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
            f"上端 (y < {START_Y:.0f}) に落ち始めの実が見つからない。"
            "撮影が始まってから落とす。盤は空にする"
        )
    ts, ys, fruit = best
    span = ys[-1] - ys[0]
    if len(ys) < MIN_POINTS or span < MIN_SPAN:
        raise SystemExit(
            f"軌跡が短すぎる ({len(ys)} 点 / {span:.0f}px 落下)。"
            f"{MIN_POINTS} 点かつ {MIN_SPAN:.0f}px 要る。"
            "盤を空にして、上から下まで落ちきる 1 本を撮り直す"
        )
    # 落ち始めを t=0 に寄せる。等速側と加速側で切片の意味を揃えるため。
    return np.asarray(ts) - ts[0], np.asarray(ys), fruit


def _fit(t: np.ndarray, y: np.ndarray) -> None:
    """等速と加速を当てて見比べる。

    2 次は 1 次を含むので残差は必ず 2 次のほうが小さい。「どちらが小さいか」では
    決まらないので、**どれだけ小さいか**で決める。合成データ (30/60/120Hz、
    検出ノイズ 0.5/2.0px) では、真に加速なら残差比 18〜102 倍・重力 2739〜2818 を
    復元し、真に等速なら比 1.0 倍・重力ほぼ 0 になった。
    """
    linear = np.polyfit(t, y, 1)
    quad = np.polyfit(t, y, 2)
    rms_linear = float(np.sqrt(np.mean((np.polyval(linear, t) - y) ** 2)))
    rms_quad = float(np.sqrt(np.mean((np.polyval(quad, t) - y) ** 2)))
    ratio = rms_linear / max(rms_quad, 1e-9)
    g = float(quad[0]) * 2.0

    print()
    print(f"点 {len(t)} 個 / {t[-1]:.3f} 秒 / y {y[0]:.1f} -> {y[-1]:.1f} px")
    print(f"  等速で当てる y = {linear[1]:.1f} + {linear[0]:.1f}*t")
    print(f"      残差 RMS {rms_linear:6.2f} px")
    print(f"  加速で当てる y = {quad[2]:.1f} + {quad[1]:.1f}*t + {quad[0]:.1f}*t^2")
    print(f"      残差 RMS {rms_quad:6.2f} px、重力 {g:.0f}")

    # 前半と後半の平均速度。落下が静止から始まるなら自由落下で 3.0 倍になる。
    # ただし実機は盤の上端より上で放すので、見え始めた時点で既に速度を持つ
    # (実測 557px/s)。そのぶん比は縮むので、判定の根拠には使わない。
    half = len(t) // 2
    v_first = (y[half] - y[0]) / max(t[half] - t[0], 1e-9)
    v_second = (y[-1] - y[half]) / max(t[-1] - t[half], 1e-9)
    speed_ratio = v_second / max(v_first, 1e-9)

    # 2 次の係数が 0 と区別できるかで決める。y = y0 + v0*t + a*t^2 の a が
    # 有意に正なら加速。初速 v0 が 0 でなくても成り立つ判定で、比を見る
    # やり方と違って「上端より上で放される」ぶんに影響されない。
    design = np.vstack([t * t, t, np.ones_like(t)]).T
    covariance = np.linalg.inv(design.T @ design) * rms_quad**2
    se_a = float(np.sqrt(covariance[0, 0]))
    se_v0 = float(np.sqrt(covariance[1, 1]))
    t_value = float(quad[0]) / max(se_a, 1e-9)

    print()
    print(f"  残差比 (等速/加速)   {ratio:6.2f} 倍")
    print(f"  前半→後半の速さ      {v_first:.0f} -> {v_second:.0f} px/s = {speed_ratio:.2f} 倍")
    print(f"  初速 v0              {float(quad[1]):.0f} ± {se_v0:.0f} px/s"
          f"   (0 でなければ盤の上端より上で放されている)")
    print(f"  2 次係数の t 値       {t_value:6.1f}   (|t| > 4 なら 0 と区別できる = 加速)")
    print()

    # 当てはめが物理として成り立っているかを見る。実測 2 本目は重力 2733 と
    # もっともらしい値を出したが、同じ当てはめの初速が -983px/s (実が上へ飛ぶ)
    # で速度比が 11.4 倍だった。自由落下の前半後半比は 3.0 が上限なので、
    # どちらも軌跡が壊れている証拠になる。数字のもっともらしさだけ見ると
    # 偽の裏付けを拾う。
    v0 = float(quad[1])
    problems: list[str] = []
    if v0 < -60.0:
        problems.append(f"初速が {v0:.0f}px/s (落とした実が上へ飛んでいる)")
    if speed_ratio > 3.5:
        problems.append(f"前半後半比が {speed_ratio:.2f} 倍 (自由落下でも 3.0 が上限)")
    if rms_quad > 12.0:
        problems.append(f"加速で当てても残差 {rms_quad:.1f}px (検出が暴れている)")

    if problems:
        print("  => 判定できない。軌跡が壊れている:")
        for problem in problems:
            print(f"       - {problem}")
        print("     盤を空にして、上端から床まで落ちきる 1 本を撮り直す")
        print("     (--csv に出して (t, y) を直接見るのが早い)")
        return

    if t_value > 4.0:
        print(f"  => 加速している。実機の重力は {g:.0f} ± {2 * se_a:.0f}")
        print(f"     sim の GRAVITY = {GRAVITY:.0f} との差は "
              f"{abs(g - GRAVITY) / max(2 * se_a, 1e-9):.1f}σ")
    else:
        print(f"  => 等速で落ちている。速さは {linear[0]:.0f} px/s")
        print("     sim は加速モデルなので、GRAVITY を消して終端速度に替える話になる")

    # sim と同じ土俵で比べる。静止からの落下時間 sqrt(2*span/G) を出すと、実機が
    # 初速を持っているぶんだけ sim が遅いように見えて逆の結論になる。同じ初速から
    # 同じ距離を落ちるのに sim が何秒かかるかを出す。
    span = y[-1] - y[0]
    sim_seconds = (-v0 + np.sqrt(v0 * v0 + 2 * GRAVITY * span)) / GRAVITY
    print()
    print(f"  参考: {span:.0f}px 落ちるのに実測 {t[-1]:.3f} 秒。")
    print(f"        sim は同じ初速 {v0:.0f}px/s からなら {sim_seconds:.3f} 秒"
          f" ({t[-1] / max(sim_seconds, 1e-9):.2f} 倍の速さ)")


def _append_run(path: Path, t: np.ndarray, y: np.ndarray, fruit: Fruit) -> int:
    """測った軌跡を run 番号と実の種類つきで追記する。

    上書きにしていたので、3 本測っても最後の 1 本しか残らなかった。種類を
    残さなかったので、重力が実の大きさでぶれているかも確かめられなかった。
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
            # repr は numpy の型名まで書いてしまうので float に落とす。桁は丸めない。
            handle.write(
                f"{run},{fruit.type},{fruit.radius!r},{float(ti)!r},{float(yi)!r}\n"
            )
    return run


def _shared_gravity(
    runs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    """全 run で重力を 1 つに共有して当てる。

    1 本ずつだと、短い弧では初速と重力が取り合いになって決まらない
    (5 本で 1313〜1488 とばらついた)。重力だけ共有し、初速と切片は run ごとに
    自由にすると、その取り合いが平均化されて決まりが良くなる。

    重力を固定すれば y - g*t^2/2 は t の 1 次式なので、走査した各重力について
    run ごとの線形最小二乗を解いて残差を足すだけでよい。
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
    # 残差が最小の 2 倍… ではなく、点あたり分散が 1 標準誤差ぶん増える幅を取る。
    dof = sum(len(t) for t, _ in runs) - 2 * len(runs) - 1
    threshold = total[best] * (1.0 + 2.0 / max(dof, 1))
    inside = grid[total <= threshold]
    return float(grid[best]), float((inside.max() - inside.min()) / 2.0)


def _summarize(path: Path) -> None:
    """貯まった run を並べて、重力が揃っているか・大きさで変わるかを見る。"""
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    runs = sorted({int(r) for r in rows[:, 0]})
    if len(runs) < 2:
        return
    print()
    print(f"=== {path} に貯まった {len(runs)} 本 ===")
    print(f"{'run':>4} {'実':>10} {'半径':>6} {'点':>4} {'落下px':>7} {'初速':>7} {'重力':>13}")
    per_type: dict[int, list[float]] = {}
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for run in runs:
        sel = rows[rows[:, 0] == run]
        fruit_type = int(sel[0, 1])
        radius = float(sel[0, 2])
        t, y = sel[:, 3], sel[:, 4]
        if len(t) < MIN_POINTS:
            print(f"{run:4d} {'':>10} {radius:6.1f} {len(t):4d}   (点が足りない)")
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
        print("  実の大きさ別:")
        for fruit_type in sorted(per_type):
            values = per_type[fruit_type]
            spread = f" (幅 {max(values) - min(values):.0f})" if len(values) > 1 else ""
            print(f"    {FRUIT_NAMES[fruit_type]:>10} 半径 {fruit_radius(fruit_type):5.1f}"
                  f"  n={len(values)}  重力 平均 {np.mean(values):.0f}{spread}")
        # ツモれるのは cherry〜orange だけで、半径は 14.2〜38.5 しかない。放す
        # 高さが同じなら、盤に入りきるまでの落下差から来る系統差は 7 程度で、
        # 1 本あたりの散らばり 71〜97 に埋もれる (合成 200 回で確認)。ここの
        # 差はほぼノイズなので、読み過ぎないこと。
        print("    (この幅で出る系統差は 7 程度。1 本の散らばり 71〜97 に埋もれるので、")
        print("     ここの差はほぼノイズ。本数を増やしても分離できない)")

    if len(series) >= 2:
        gravity, error = _shared_gravity(series)
        print()
        print(f"  全 {len(series)} 本で重力を共有して当てると: {gravity:.0f} ± {error:.0f}")
        print(f"    sim の GRAVITY = {GRAVITY:.0f} との比 {GRAVITY / gravity:.2f} 倍")


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
