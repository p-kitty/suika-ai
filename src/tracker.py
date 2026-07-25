from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .vision.state import Fruit

# 検出が数フレーム欠けても消さない。静止したフルーツの点滅を防ぐ。
MAX_MISSING = 5
# この回数以上見えたトラックだけを確定として返す。
MIN_HITS = 2
# 同一フルーツとみなす距離 (半径比)。
MATCH_DISTANCE_RATIO = 0.7
# 種類は直近フレームの多数決で決める。
VOTE_WINDOW = 9

POSITION_ALPHA = 0.5
RADIUS_ALPHA = 0.25


@dataclass
class _Track:
    x: float
    y: float
    radius: float
    confidence: float
    types: deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    hits: int = 0
    missing: int = 0

    @property
    def type(self) -> int:
        return Counter(self.types).most_common(1)[0][0]


class Tracker:
    """検出結果をフレーム間で対応付けて安定させる。

    視点の微動で検出が揺れても、位置と種類を平滑化・多数決することで
    表示が点滅しないようにする。
    """

    def __init__(self) -> None:
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        self._tracks = []

    def update(self, fruits: list[Fruit]) -> list[Fruit]:
        unmatched = list(fruits)

        for track in self._tracks:
            match = self._pop_nearest(track, unmatched)
            if match is None:
                track.missing += 1
                continue

            track.missing = 0
            track.hits += 1
            track.x += (match.x - track.x) * POSITION_ALPHA
            track.y += (match.y - track.y) * POSITION_ALPHA
            track.radius += (match.radius - track.radius) * RADIUS_ALPHA
            track.confidence = match.confidence
            track.types.append(match.type)

        for fruit in unmatched:
            track = _Track(
                x=fruit.x,
                y=fruit.y,
                radius=fruit.radius,
                confidence=fruit.confidence,
                hits=1,
            )
            track.types.append(fruit.type)
            self._tracks.append(track)

        self._tracks = [track for track in self._tracks if track.missing <= MAX_MISSING]

        return [
            Fruit(
                type=track.type,
                x=track.x,
                y=track.y,
                radius=track.radius,
                confidence=track.confidence,
            )
            for track in self._tracks
            if track.hits >= MIN_HITS
        ]

    @staticmethod
    def _pop_nearest(track: _Track, fruits: list[Fruit]) -> Fruit | None:
        threshold = max(track.radius, 4.0) * MATCH_DISTANCE_RATIO
        best_index = None
        best_distance = threshold

        for index, fruit in enumerate(fruits):
            distance = float(np.hypot(fruit.x - track.x, fruit.y - track.y))
            if distance < best_distance:
                best_distance = distance
                best_index = index

        if best_index is None:
            return None

        return fruits.pop(best_index)
