"""学習に渡す観測。検出の生データから余分を落とす。"""

from dataclasses import dataclass

from .vision.board import BoardResult
from .vision.normalized import NORMALIZED_WIDTH
from .vision.state import Fruit


@dataclass(frozen=True)
class Observation:
    """1 フレームの盤面の読み。

    座標は正規化した盤面 (幅 NORMALIZED_WIDTH)。held の x はそのまま落とす列。
    fruits は表示・方策用 (Tracker 平滑化後)。raw_fruits は静止判定用の生検出。
    """

    ready: bool
    blocked: bool
    fruits: tuple[Fruit, ...]
    held_type: int | None
    held_x: float | None
    next_type: int | None
    raw_fruits: tuple[Fruit, ...] = ()

    @property
    def motion_fruits(self) -> tuple[Fruit, ...]:
        """静止判定用。生検出があればそちら、なければ fruits。"""
        return self.raw_fruits if self.raw_fruits else self.fruits


def from_board(
    result: BoardResult,
    *,
    raw_fruits: list[Fruit] | tuple[Fruit, ...] | None = None,
) -> Observation:
    """検出結果を学習用の観測に直す。

    ready は「次の一手を決められる」状態。盤面が見えていて、ダイアログが無く、
    落下待ちフルーツの種類と列が両方読めたときだけ真になる。
    """
    if not result.found or result.blocked or result.fruits is None:
        return Observation(
            ready=False,
            blocked=result.blocked,
            fruits=(),
            held_type=None,
            held_x=None,
            next_type=None,
            raw_fruits=(),
        )

    held = result.held_fruit
    held_type = held.fruit.type if held is not None and held.fruit is not None else None
    held_x = held.x if held is not None else None

    next_fruit = result.next_fruit
    next_type = (
        next_fruit.fruit.type
        if next_fruit is not None and next_fruit.fruit is not None
        else None
    )

    fruits = tuple(result.fruits)
    raw = tuple(raw_fruits) if raw_fruits is not None else fruits

    return Observation(
        ready=held_type is not None and held_x is not None,
        blocked=False,
        fruits=fruits,
        held_type=held_type,
        held_x=held_x,
        next_type=next_type,
        raw_fruits=raw,
    )


def clamp_drop_x(x: float, fruit_type: int | None = None) -> float:
    """フルーツが壁にめり込まない範囲に落とす列を収める。"""
    from .vision.classify import fruit_radius

    radius = 0.0 if fruit_type is None else fruit_radius(fruit_type)
    return float(min(max(x, radius), NORMALIZED_WIDTH - radius))
