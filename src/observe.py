"""The observation passed to learning. Drops the excess from raw detection data."""

from dataclasses import dataclass

from .vision.board import BoardResult
from .vision.colors import FRUIT_NAMES
from .vision.normalized import NORMALIZED_WIDTH
from .vision.state import Fruit


@dataclass(frozen=True)
class Observation:
    """The reading of the board for one frame.

    Coordinates are the normalized board (width NORMALIZED_WIDTH). held's x is the column dropped as is.
    """

    ready: bool
    blocked: bool
    fruits: tuple[Fruit, ...]
    held_type: int | None
    held_x: float | None
    next_type: int | None

    @property
    def held_name(self) -> str | None:
        return None if self.held_type is None else FRUIT_NAMES[self.held_type]

    @property
    def next_name(self) -> str | None:
        return None if self.next_type is None else FRUIT_NAMES[self.next_type]


def from_board(result: BoardResult) -> Observation:
    """Turn a detection result into an observation for learning.

    ready is the state where 'the next move can be decided'. True only when the board is visible, there is no dialog,
    and both the type and column of the waiting fruit could be read.
    """
    if not result.found or result.blocked or result.fruits is None:
        return Observation(
            ready=False,
            blocked=result.blocked,
            fruits=(),
            held_type=None,
            held_x=None,
            next_type=None,
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

    return Observation(
        ready=held_type is not None and held_x is not None,
        blocked=False,
        fruits=tuple(result.fruits),
        held_type=held_type,
        held_x=held_x,
        next_type=next_type,
    )


def clamp_drop_x(x: float, fruit_type: int | None = None) -> float:
    """Clamp the drop column to the range where the fruit does not sink into the walls."""
    from .vision.classify import fruit_radius_ratios

    radius = 0.0
    if fruit_type is not None:
        radius = fruit_radius_ratios()[fruit_type] * NORMALIZED_WIDTH

    return float(min(max(x, radius), NORMALIZED_WIDTH - radius))
