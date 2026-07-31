import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.agent import LinearPolicy
from src.capture import CAPTURE_FPS, capture
from src.config import load
from src.debug_dump import dump
from src.env import Env
from src.hotkeys import EdgeKey
from src.observe import Observation
from src.policy import choose_x
from src.preview import PreviewState, render_preview
from src.window import maximize_window

WINDOW_TITLE = "Suika"
MESSAGE_SECONDS = 3.0
DEFAULT_CKPT = Path(__file__).resolve().parent / "artifacts" / "policy_sim.npz"

DUMP_KEY = ord("s")
QUIT_KEY = 27
# Work regardless of focus (even with VRC in front). Space is not used since it collides with jump.
VK_G = 0x47
VK_P = 0x50
VK_L = 0x4C

# Preview while waiting. Only the video runs; stale detection circles are not drawn.
PUMP_HZ = float(CAPTURE_FPS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suika screen agent")
    parser.add_argument(
        "--policy",
        choices=("bootstrap", "learned"),
        default=None,
        help="policy for the drop column. When omitted, learned if an npz exists, otherwise bootstrap",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CKPT,
        help="weights for learned (npz)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env = Env()
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
        raise SystemExit(f"no weights for learned: {args.checkpoint}")

    def choose(obs_in: Observation) -> float:
        if policy_name == "learned":
            assert learned_policy is not None
            _, x, _ = learned_policy.act(obs_in, greedy=True)
            return x
        return choose_x(obs_in)

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
        """Toggle auto on the rising edge of G. True when pressed."""
        nonlocal auto_play, message, message_until
        if not g_key.poll():
            return False
        auto_play = not auto_play
        message = f"auto={'ON' if auto_play else 'off'}"
        message_until = time.monotonic() + MESSAGE_SECONDS
        print(message)
        return True

    def poll_policy_toggle() -> None:
        """Switch bootstrap / learned with L."""
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
        """One move on the rising edge of P. While waiting only the state is updated."""
        return p_key.poll()

    def pump_ui() -> None:
        """Keep the video running while waiting for settle / aim. Only the aim line stays."""
        nonlocal frame, next_pump
        # Do not misfire on a key held or pressed again while waiting after returning.
        poll_step_key()
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
        render_preview(
            frame,
            board=env.board,
            obs=obs,
            state=preview_state(now),
            debug_board=True,
            window_title=WINDOW_TITLE,
        )

    def should_abort() -> bool:
        """For wait loops during auto. Abort when switched off with G."""
        poll_g_toggle()
        pump_ui()
        return not auto_play

    while True:
        fresh = capture()
        if fresh is not None:
            frame = fresh
        if frame is None:
            # Only watch keys until the first frame arrives.
            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                break
            continue

        key = cv2.waitKey(1) & 0xFF
        now = time.monotonic()

        # G / P / L are detected globally so they work even with VRChat in front. Holding does not repeat.
        # While waiting in step / settle, the same detection runs via should_abort / pump_ui.
        poll_g_toggle()
        poll_policy_toggle()
        step_pressed = poll_step_key()

        # Detect only when a new capture arrives. Keeps video and overlay on the same cycle.
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

        # P = one move (global). During continuous auto with g, drop on every ready.
        # Draw AUTO/LIVE before entering the wait so it shows.
        from_auto = auto_play and not step_pressed
        if step_pressed or (auto_play and obs.ready and not obs.blocked):
            message = "settling..."
            message_until = now + MESSAGE_SECONDS
            show(now)

            def abort() -> bool:
                # Keep the preview running while waiting. On auto, G can abort it.
                if from_auto:
                    return should_abort()
                pump_ui()
                return False

            if not obs.ready:
                message = "step: not ready"
                print(message)
            else:
                def on_aim(target: float) -> None:
                    """Show the line the moment the column is decided, so it stays visible throughout aim."""
                    nonlocal aim_x, message, message_until, frame, obs, board
                    aim_x = target
                    message = f"aiming x={target:.0f}"
                    message_until = time.monotonic() + MESSAGE_SECONDS
                    frame, obs, board = _refresh(env, frame, obs)
                    show(time.monotonic())

                # Confirm settle → decide the column on the same observation → aim. Do not read a moving board.
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
                    # The aim line is the dropped column. held is the next fruit's position, so do not overwrite it.
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


def _refresh(env: Env, frame: np.ndarray, fallback: Observation):
    """Re-observe on the latest frame after the action. Use the previous one if none is available."""
    fresh = capture()
    if fresh is None:
        return frame, fallback, env.board
    obs = env.observe(fresh)
    return fresh, obs, env.board


if __name__ == "__main__":
    main()
