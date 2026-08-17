import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.training.agent import LinearPolicy
from src.game.capture import CAPTURE_FPS, capture
from src.util.config import load
from src.viz.debug_dump import dump
from src.game.env import Env
from src.game.hotkeys import EdgeKey
from src.observe import Observation
from src.policy import choose_x
from src.viz.preview import PreviewState, render_preview
from src.game.window import maximize_window

WINDOW_TITLE = "Suika"
MESSAGE_SECONDS = 3.0
DEFAULT_CKPT = Path(__file__).resolve().parent / "artifacts" / "policy_sim.npz"

DUMP_KEY = ord("s")
QUIT_KEY = 27
# フォーカスに関係なく効かせる (VRC 前面でも)。Space はジャンプと被るので使わない。
VK_G = 0x47
VK_P = 0x50
VK_L = 0x4C

# 待ち中プレビュー。映像だけ回し、古い検出円は載せない。
PUMP_HZ = float(CAPTURE_FPS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suika 画面エージェント")
    parser.add_argument(
        "--policy",
        choices=("bootstrap", "learned"),
        default=None,
        help="落とす列の方策。省略時は npz があれば learned、なければ bootstrap",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CKPT,
        help="learned 用の重み (npz)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env = Env()
    # choose_x の候補評価 (simulate_drop) をプロセス並列化する。プロセス起動コストが
    # 大きいので、手ごとではなくプログラム起動時に 1 回だけ作って使い回す。
    pool = ProcessPoolExecutor()
    message = ""
    message_until = 0.0
    next_auto_dump = 0.0
    next_pump = 0.0
    aim_x: float | None = None
    auto_play = False
    obs = Observation(
        ready=False,
        blocked=False,
        fruits=(),
        held_type=None,
        held_x=None,
        next_type=None,
    )

    learned_policy: LinearPolicy | None = None
    if args.checkpoint.is_file():
        learned_policy = LinearPolicy()
        learned_policy.load(args.checkpoint)
        print(f"loaded {args.checkpoint}")

    if args.policy is None:
        policy_name = "learned" if learned_policy is not None else "bootstrap"
    else:
        policy_name = args.policy
    if policy_name == "learned" and learned_policy is None:
        raise SystemExit(f"learned 用の重みが無い: {args.checkpoint}")

    def choose(obs_in: Observation) -> float:
        if policy_name == "learned":
            assert learned_policy is not None
            _, x, _ = learned_policy.act(obs_in, greedy=True)
            return x
        return choose_x(obs_in, pool=pool)

    maximize_window(WINDOW_TITLE)
    g_key = EdgeKey(VK_G)
    p_key = EdgeKey(VK_P)
    l_key = EdgeKey(VK_L)
    frame: np.ndarray | None = None
    print(f"policy={policy_name}")

    def preview_state(now: float) -> PreviewState:
        return PreviewState(
            aim_x=aim_x,
            auto_play=auto_play,
            policy_name=policy_name,
            message=message,
            message_until=message_until,
            now=now,
        )

    def poll_g_toggle() -> bool:
        """G の立ち上がりで auto をトグル。押されたら True。"""
        nonlocal auto_play, message, message_until
        if not g_key.poll():
            return False
        auto_play = not auto_play
        message = f"auto={'ON' if auto_play else 'off'}"
        message_until = time.monotonic() + MESSAGE_SECONDS
        print(message)
        return True

    def poll_policy_toggle() -> None:
        """L で bootstrap / learned を切替。"""
        nonlocal policy_name, message, message_until
        if not l_key.poll():
            return
        if learned_policy is None:
            message = "learned ckpt missing"
            message_until = time.monotonic() + MESSAGE_SECONDS
            print(message)
            return
        policy_name = "bootstrap" if policy_name == "learned" else "learned"
        message = f"policy={policy_name}"
        message_until = time.monotonic() + MESSAGE_SECONDS
        print(message)

    def poll_step_key() -> bool:
        """P の立ち上がりで 1 手。待ち中は状態だけ更新する。"""
        return p_key.poll()

    def pump_ui() -> None:
        """settle / aim 待ち中も映像を回す。狙い線だけは残す。"""
        nonlocal frame, next_pump
        # 待ち中の押しっぱなし／再押しを、戻ったあとに誤発火させない。
        poll_step_key()
        # G / L も待ち中に取りこぼすと、押した瞬間が step/settle の間に埋もれて
        # 切り替わらないまま次のトグルまで判定がずれる。手動 P 単発の待ちは
        # should_abort を経由せずここだけを通るので、G の検出もここに置く。
        poll_g_toggle()
        poll_policy_toggle()
        now = time.monotonic()
        if now < next_pump:
            cv2.waitKey(1)
            return
        next_pump = now + 1.0 / PUMP_HZ
        fresh = capture()
        if fresh is not None:
            frame = fresh
        if frame is None:
            return
        render_preview(
            frame,
            board=env.board,
            obs=obs,
            state=preview_state(now),
            debug_board=False,
            window_title=WINDOW_TITLE,
        )

    def show(now: float) -> None:
        if frame is None:
            return
        render_preview(
            frame,
            board=env.board,
            obs=obs,
            state=preview_state(now),
            debug_board=True,
            window_title=WINDOW_TITLE,
        )

    def should_abort() -> bool:
        """自動中の待ちループ用。G で off にしたら打ち切る。"""
        pump_ui()
        return not auto_play

    while True:
        fresh = capture()
        if fresh is not None:
            frame = fresh
        if frame is None:
            # 最初のフレームが来るまでキーだけ見る。
            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                break
            continue

        key = cv2.waitKey(1) & 0xFF
        now = time.monotonic()

        # G / P / L は VRChat 前面でも効くようグローバル検出。押しっぱなしで連打しない。
        # step / settle の待ち中も should_abort / pump_ui 経由で同じ検出を回す。
        poll_g_toggle()
        poll_policy_toggle()
        step_pressed = poll_step_key()

        # 新しいキャプチャが来たときだけ検出。映像とオーバーレイを同じ周期にする。
        if fresh is not None or step_pressed or key == DUMP_KEY:
            obs = env.observe(frame)
        board = env.board

        if obs.ready and obs.held_x is not None and aim_x is None:
            aim_x = obs.held_x

        interval = load().get("debug_dump_interval_sec", 0)
        auto_dump = bool(interval) and board is not None and board.found and now >= next_auto_dump

        if key == DUMP_KEY or auto_dump:
            next_auto_dump = now + max(interval, 1)
            if board is not None:
                message = dump(frame, board)
                message_until = now + MESSAGE_SECONDS
                print(message)

        # P = 1 手 (global)。g で連続自動中なら ready のたびに落とす。
        # 待ちに入る前に AUTO/LIVE を描画して見せる。
        from_auto = auto_play and not step_pressed
        if step_pressed or (auto_play and obs.ready and not obs.blocked):
            message = "settling..."
            message_until = now + MESSAGE_SECONDS
            show(now)

            def abort() -> bool:
                # 待ち中もプレビューを回す。auto なら G で打ち切れる。
                if from_auto:
                    return should_abort()
                pump_ui()
                return False

            if not obs.ready:
                message = "step: not ready"
                print(message)
            else:
                def on_aim(target: float) -> None:
                    """列が決まった瞬間に線を出して、aim 中ずっと見えるようにする。"""
                    nonlocal aim_x, message, message_until, frame, obs, board
                    aim_x = target
                    message = f"aiming x={target:.0f}"
                    message_until = time.monotonic() + MESSAGE_SECONDS
                    if frame is None:
                        return
                    frame, obs, board = _refresh(env, frame, obs)
                    show(time.monotonic())

                # 静止確認→同じ観測で列決め→狙い。動いている盤を読まない。
                result = env.step(abort=abort, choose=choose, on_aim=on_aim)
                if from_auto and not auto_play:
                    message = "auto=off"
                    frame, obs, board = _refresh(env, frame, obs)
                elif result.info == "not settled":
                    message = "step: not settled"
                    print(message)
                    frame, obs, board = _refresh(env, frame, obs)
                else:
                    target = result.target_x
                    # 狙い線は落とした列。held は次の実の位置なので上書きしない。
                    if target is not None:
                        aim_x = target
                    message = (
                        f"{policy_name} x={target:.0f} -> {result.info}"
                        if target is not None
                        else f"{policy_name} -> {result.info}"
                    )
                    obs = result.observation
                    frame, obs, board = _refresh(env, frame, obs)
                    print(message)
                    if from_auto and not auto_play:
                        message = f"{message} (stop)"
                    elif result.done:
                        auto_play = False
                        message = f"{message} (stop)"
            message_until = time.monotonic() + MESSAGE_SECONDS

        show(now)

        if key == QUIT_KEY:
            break

    cv2.destroyAllWindows()
    pool.shutdown()


def _refresh(env: Env, frame: np.ndarray, fallback: Observation):
    """操作後に最新フレームで観測し直す。取れなければ直前のものを使う。"""
    fresh = capture()
    if fresh is None:
        return frame, fallback, env.board
    obs = env.observe(fresh)
    return fresh, obs, env.board


if __name__ == "__main__":
    main()
