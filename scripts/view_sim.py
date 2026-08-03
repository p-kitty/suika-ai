"""画面なし sim の着地を円で見る。

用法:
  python scripts/view_sim.py
  python scripts/view_sim.py --seed 7

操作:
  マウス   落とす列
  クリック その列に落とす (左パネルで物理アニメ)
  g        auto トグル (bootstrap の最善列で連続落下)
  f        アニメ無効化トグル (ON で毎手の落下アニメを飛ばし、結果だけ表示)
  r        リセット
  Esc      終了 (アニメ中はスキップ)

左: いまの盤 + held の接点 preview / 落下アニメ。右: 落としたあとの結果 (アニメ中も最終盤)。
ヘッダ右に NEXT の円。ツモは cherry〜orange を毎回ランダム (seed 指定時は再現)。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.draw import FRUIT_BGR, mode_badge, put_text
from src.observe import clamp_drop_x
from src.policy import choose_x, drop_scores
from src.reward import GAME_OVER_Y
from src.sim_env import SimEnv
from src.sim_physics import DT, iter_simulate_drop, land_y
from src.vision.classify import fruit_radius
from src.vision.colors import FRUIT_NAMES
from src.vision.normalized import NORMALIZED_HEIGHT, NORMALIZED_WIDTH
from src.vision.state import Fruit

PAD = 16
GAP = 20
HEADER = 72
FOOTER = 56
NEXT_PREVIEW_R = 18
# 物理 1 ステップあたりの表示。1 なら実時間、2 なら 2x。
ANIM_STRIDE = 1
ANIM_WAIT_MS = max(1, int(round(1000.0 * DT / ANIM_STRIDE)))
WINDOW = "suika-ai sim"
# main で画面サイズに合わせて入れる。
SCALE = 1.5


def _fit_scale(max_scale: float = 2.0) -> float:
    """ウィンドウ全体が画面に収まる SCALE。1080p だと 2 倍だと下が切れる。"""
    screen_w, screen_h = 1920, 1080
    try:
        import ctypes

        screen_w = int(ctypes.windll.user32.GetSystemMetrics(0))
        screen_h = int(ctypes.windll.user32.GetSystemMetrics(1))
    except Exception:
        pass
    # タイトルバー・タスクバーぶんを空ける。
    max_w = max(640, screen_w - 48)
    max_h = max(480, screen_h - 96)
    scale_h = (max_h - PAD * 2 - HEADER - FOOTER) / NORMALIZED_HEIGHT
    scale_w = (max_w - PAD * 2 - GAP) / (NORMALIZED_WIDTH * 2)
    return float(max(0.8, min(max_scale, scale_h, scale_w)))


def main() -> None:
    global SCALE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="ツモ列の乱数シード (省略時は毎回ランダム)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="描画倍率 (省略時は画面に収まるよう自動)",
    )
    args = parser.parse_args()
    SCALE = args.scale if args.scale is not None else _fit_scale()

    env = SimEnv(seed=args.seed)
    obs = env.reset()
    aim_x = NORMALIZED_WIDTH / 2
    total_score = 0.0
    last_info = "ok"
    auto_play = False
    fast_forward = False
    message = "mouse: aim/drop  g: auto  f: fast  r: reset  Esc: quit"

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        nonlocal aim_x
        panel_w = int(round(NORMALIZED_WIDTH * SCALE))
        if PAD <= x < PAD + panel_w:
            aim_x = (x - PAD) / SCALE
        elif PAD + panel_w + GAP <= x < PAD + 2 * panel_w + GAP:
            aim_x = (x - PAD - panel_w - GAP) / SCALE

        if event == cv2.EVENT_LBUTTONDOWN and obs.held_type is not None:
            _drop()

    def _toggle_auto() -> None:
        nonlocal auto_play, message
        auto_play = not auto_play
        message = f"auto={'ON' if auto_play else 'off'}"

    def _toggle_fast() -> None:
        nonlocal fast_forward, message
        fast_forward = not fast_forward
        message = f"fast={'ON' if fast_forward else 'off'}"

    def _drop(x: float | None = None) -> None:
        nonlocal obs, total_score, last_info, message, aim_x, auto_play
        if obs.held_type is None or not obs.ready:
            message = "not ready"
            return
        target = clamp_drop_x(aim_x if x is None else x, obs.held_type)
        aim_x = target
        # 見せるだけ。盤の更新・採点・勝敗は SimEnv に任せる (物理は決定的
        # なので、アニメで見た最終形と env.step の結果は一致する)。
        _play_drop_anim(
            before=list(obs.fruits),
            held_type=obs.held_type,
            drop_x=target,
            next_type=obs.next_type,
            total_score=total_score,
            info=last_info,
            auto_play=auto_play,
            on_toggle_auto=_toggle_auto,
            fast_forward=fast_forward,
        )
        step = env.step(target)
        obs = step.observation
        total_score += step.score
        last_info = step.info
        message = (
            f"drop x={target:.0f}  score+{step.score:.0f}  "
            f"merges={step.merges}  {last_info}"
        )
        if step.done:
            if auto_play:
                auto_play = False
                message += "  (auto off — r to reset)"
            else:
                message += "  (done — r to reset)"

    def _reset() -> None:
        nonlocal obs, total_score, last_info, message, aim_x
        obs = env.reset()
        total_score = 0.0
        last_info = "ok"
        aim_x = NORMALIZED_WIDTH / 2
        message = "reset"

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        held = obs.held_type
        if held is not None:
            aim = clamp_drop_x(aim_x, held)
        else:
            aim = aim_x

        after: list[Fruit] = list(obs.fruits)
        merges = 0
        land: tuple[float, float] | None = None
        score = penalties = 0.0
        if held is not None:
            held_r = fruit_radius(held)
            land = (aim, land_y(obs.fruits, aim, held_r))
            score, penalties, _eval, after, merges = drop_scores(
                obs.fruits, held, aim, next_type=obs.next_type
            )

        frame = _render(
            before=list(obs.fruits),
            after=after,
            aim_x=aim,
            land=land,
            held_type=held,
            next_type=obs.next_type,
            merges=merges,
            score=score,
            penalties=penalties,
            total_score=total_score,
            info=last_info,
            message=message,
            auto_play=auto_play,
            fast_forward=fast_forward,
        )
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        if key == ord("r"):
            _reset()
        elif key == ord("g"):
            _toggle_auto()
        elif key == ord("f"):
            _toggle_fast()

        done = last_info in ("dead", "win")
        if auto_play and not done and obs.held_type is not None and obs.ready:
            x = choose_x(obs)
            aim_x = x
            _drop(x)

    cv2.destroyAllWindows()


def _play_drop_anim(
    *,
    before: list[Fruit],
    held_type: int,
    drop_x: float,
    next_type: int | None,
    total_score: float,
    info: str,
    auto_play: bool = False,
    on_toggle_auto: Callable[[], None] | None = None,
    fast_forward: bool = False,
) -> None:
    """左パネルに落下物理を再生する。右は最終 AFTER DROP を固定表示。

    fast_forward が ON のときはアニメ自体を回さず即 return する
    (drop_scores が既に物理を 1 回解いているので、二重に解かない)。
    """
    # 結果の score / penalties は最初から固定表示 (アニメ中も 0 にしない)。
    result_score, result_penalties, _eval, final_after, _final_merges = drop_scores(
        before, held_type, drop_x, next_type=next_type
    )
    if fast_forward:
        return
    skip = False
    frame_i = 0
    auto_on = auto_play
    for after, merges, _merge_types in iter_simulate_drop(before, held_type, drop_x):
        show = (not skip) and (frame_i % ANIM_STRIDE == 0)
        frame_i += 1
        if not show:
            continue
        canvas = _render(
            before=after,
            after=final_after,
            aim_x=drop_x,
            land=None,
            held_type=None,
            next_type=next_type,
            merges=merges,
            score=result_score,
            penalties=result_penalties,
            total_score=total_score,
            info=info,
            message=f"animating… merges={merges}  (Esc skip / g auto)",
            left_title="LIVE",
            right_title="AFTER DROP",
            auto_play=auto_on,
        )
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(ANIM_WAIT_MS) & 0xFF
        if key == 27:
            skip = True
        elif key == ord("g") and on_toggle_auto is not None:
            on_toggle_auto()
            auto_on = not auto_on


def _render(
    *,
    before: list[Fruit],
    after: list[Fruit],
    aim_x: float,
    land: tuple[float, float] | None,
    held_type: int | None,
    next_type: int | None,
    merges: int,
    score: float,
    penalties: float,
    total_score: float,
    info: str,
    message: str,
    left_title: str = "NOW",
    right_title: str = "AFTER DROP",
    auto_play: bool = False,
    fast_forward: bool = False,
) -> np.ndarray:
    panel_w = int(round(NORMALIZED_WIDTH * SCALE))
    panel_h = int(round(NORMALIZED_HEIGHT * SCALE))
    width = PAD * 2 + panel_w * 2 + GAP
    height = PAD * 2 + panel_h + HEADER + FOOTER
    canvas = np.full((height, width, 3), 36, dtype=np.uint8)

    left = _board_panel(before, aim_x, land, held_type, title=left_title)
    right = _board_panel(after, aim_x, None, None, title=right_title)
    y0 = PAD + HEADER
    canvas[y0 : y0 + panel_h, PAD : PAD + panel_w] = left
    x1 = PAD + panel_w + GAP
    canvas[y0 : y0 + panel_h, x1 : x1 + panel_w] = right

    held_name = FRUIT_NAMES[held_type] if held_type is not None else "—"
    next_name = FRUIT_NAMES[next_type] if next_type is not None else "—"
    put_text(
        canvas,
        f"held={held_name}  next={next_name}  aim x={aim_x:.0f}  scale={SCALE:.2f}",
        (PAD, 28),
        (220, 220, 220),
        scale=0.6,
    )
    put_text(
        canvas,
        f"preview  score={score:.0f}  penalties={penalties:.1f}  merges={merges}",
        (PAD, 52),
        (180, 255, 180),
        scale=0.55,
    )
    _draw_next_preview(canvas, next_type, width)
    mode_badge(canvas, auto_play, fast_forward)
    put_text(
        canvas,
        f"episode  score={total_score:.0f}  info={info}",
        (PAD, height - 28),
        (200, 200, 255),
        scale=0.55,
    )
    put_text(canvas, message, (PAD, height - 8), (180, 180, 180), scale=0.5, thickness=1)
    return canvas


def _board_panel(
    fruits: list[Fruit],
    aim_x: float,
    land: tuple[float, float] | None,
    held_type: int | None,
    *,
    title: str,
) -> np.ndarray:
    panel_w = int(round(NORMALIZED_WIDTH * SCALE))
    panel_h = int(round(NORMALIZED_HEIGHT * SCALE))
    img = np.full((panel_h, panel_w, 3), (45, 55, 70), dtype=np.uint8)
    # 床・枠
    cv2.rectangle(img, (0, 0), (panel_w - 1, panel_h - 1), (90, 100, 120), 2)
    danger_y = int(round(GAME_OVER_Y * SCALE))
    cv2.line(img, (0, danger_y), (panel_w - 1, danger_y), (40, 40, 160), 1)

    ax = int(round(aim_x * SCALE))
    cv2.line(img, (ax, 0), (ax, panel_h - 1), (0, 255, 255), 1)

    for fruit in fruits:
        _draw_fruit(img, fruit)

    if land is not None and held_type is not None:
        lx, ly = land
        ghost = Fruit(
            type=held_type,
            x=lx,
            y=ly,
            radius=fruit_radius(held_type),
            confidence=100.0,
        )
        _draw_fruit(img, ghost)

    put_text(img, title, (8, 22), (230, 230, 230), scale=0.55)
    put_text(img, f"n={len(fruits)}", (8, 44), (180, 180, 180), scale=0.45, thickness=1)
    return img


def _draw_next_preview(
    canvas: np.ndarray, next_type: int | None, width: int
) -> None:
    """ヘッダ右端付近に NEXT の円。右端は AUTO/LIVE バッジ用に空ける。"""
    badge_reserve = 110
    cx = width - PAD - NEXT_PREVIEW_R - 8 - badge_reserve
    cy = HEADER // 2 + PAD // 2
    put_text(canvas, "NEXT", (cx - 28, cy - NEXT_PREVIEW_R - 6), (200, 200, 200), scale=0.45)
    if next_type is None:
        cv2.circle(canvas, (cx, cy), NEXT_PREVIEW_R, (90, 90, 90), 2)
        return
    bgr = FRUIT_BGR[next_type]
    overlay = canvas.copy()
    cv2.circle(overlay, (cx, cy), NEXT_PREVIEW_R, bgr, -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
    cv2.circle(canvas, (cx, cy), NEXT_PREVIEW_R, bgr, 2)
    label = FRUIT_NAMES[next_type][:3]
    put_text(
        canvas,
        label,
        (cx - NEXT_PREVIEW_R, cy + NEXT_PREVIEW_R + 14),
        (240, 240, 240),
        scale=0.4,
        thickness=1,
    )


def _draw_fruit(img: np.ndarray, fruit: Fruit) -> None:
    cx = int(round(fruit.x * SCALE))
    cy = int(round(fruit.y * SCALE))
    r = max(2, int(round(fruit.radius * SCALE)))
    bgr = FRUIT_BGR[fruit.type]
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), r, bgr, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.circle(img, (cx, cy), r, bgr, 2)
    cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1)
    label = FRUIT_NAMES[fruit.type][:3]
    put_text(
        img,
        label,
        (cx - r, max(14, cy - r - 4)),
        (240, 240, 240),
        scale=0.35,
        thickness=1,
    )


if __name__ == "__main__":
    main()
