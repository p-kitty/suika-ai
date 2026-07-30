"""落下待ちフルーツ検出。盤面の縁を越えた実に潰されないこと。"""

import cv2
import numpy as np

from src.vision.held import (
    BAND_HEIGHT,
    DROP_HEIGHT,
    _band_mask,
    _find_blob,
    _height_error,
)


def _blank_band() -> np.ndarray:
    # 夜空っぽい暗い帯。
    return np.full((BAND_HEIGHT, 400, 3), 20, dtype=np.uint8)


def _paint_fruit(band: np.ndarray, x: int, y: int, radius: int, bgr: tuple[int, int, int]) -> None:
    cv2.circle(band, (x, y), radius, bgr, -1)


def test_find_blob_at_drop_height() -> None:
    band = _blank_band()
    # イチゴ色。落下点の高さ (帯 y ≈ BAND_HEIGHT - DROP_HEIGHT)。
    cy = int(BAND_HEIGHT - DROP_HEIGHT)
    _paint_fruit(band, 200, cy, 16, (40, 40, 200))
    blob = _find_blob(_band_mask(band))
    assert blob is not None
    x, y, radius = blob
    assert abs(x - 200) < 8
    assert _height_error(y) <= 15
    assert 10 < radius < 28


def test_overhanging_fruit_does_not_hide_held() -> None:
    # NOTES: パイナップルが縁を越えると held が消えて進行停止していた。
    band = _blank_band()
    cy = int(BAND_HEIGHT - DROP_HEIGHT)
    _paint_fruit(band, 200, cy, 16, (40, 40, 200))
    # 帯の下端・別列に大きい実 (縁からのはみ出し)。実プレイでも列はずれやすい。
    _paint_fruit(band, 320, BAND_HEIGHT - 8, 70, (0, 200, 255))

    blob = _find_blob(_band_mask(band))
    assert blob is not None
    x, y, _radius = blob
    assert abs(x - 200) < 12
    assert _height_error(y) <= 15
