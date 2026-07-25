from dataclasses import dataclass

from .colors import FRUIT_NAMES_JA


@dataclass
class Fruit:
    type: int
    x: float
    y: float
    radius: float
    confidence: float

    @property
    def name(self) -> str:
        return FRUIT_NAMES_JA[self.type]
