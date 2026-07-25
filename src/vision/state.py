from dataclasses import dataclass

from .colors import FRUIT_NAMES


@dataclass
class Fruit:
    type: int
    x: float
    y: float
    radius: float
    confidence: float

    @property
    def name(self) -> str:
        return FRUIT_NAMES[self.type]
