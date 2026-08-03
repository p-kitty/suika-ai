"""Run saved images through detection in bulk and put the results in a form checkable by eye.

    python scripts/check_detection.py                    # screenshots and debug
    python scripts/check_detection.py debug              # specify a location

*_board.png is treated as an already warped board, anything else as a full screen.
Writes images with detections and masks side by side to debug/check/, and also lists suspicious
detections. Run this every time thresholds are touched and look at the difference from last time.
"""

import sys
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT

from src.draw import put_text
from src.imagefile import read, write
from src.vision.board import localize
from src.vision.classify import fruit_radius_ratios
from src.vision.fruits import detect, fruit_mask
from src.vision.held import HeldResult
from src.vision.normalized import NORMALIZED_WIDTH
from src.vision.state import Fruit

DEFAULT_SOURCES = ("screenshots", "debug")
OUTPUT_DIR = ROOT / "debug" / "check"

# Classification always picks the nearest stage, so radii stay around the expected value.
# If it is still off this much, suspect the blob segmentation.
SUSPECT_RADIUS_RATIO = 1.25
SUSPECT_CONFIDENCE = 50.0


def main(argv: list[str]) -> int:
    names = argv or list(DEFAULT_SOURCES)
    paths = sorted({path for name in names for path in _collect(ROOT / name)})

    if not paths:
        print(f"no images found: {', '.join(names)}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_fruits = 0
    total_suspects = 0
    for path in paths:
        fruits, suspects = _check(path)
        total_fruits += fruits
        total_suspects += suspects

    print(f"\n{len(paths)} images / {total_fruits} detected / {total_suspects} suspicious")
    print(f"written to: {OUTPUT_DIR}")
    return 0


def _collect(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        return []

    # Masks are binary, so running detection on them is meaningless. Files starting with _
    # are treated as temporary files placed by hand and skipped.
    return [
        path
        for path in source.glob("*.png")
        if OUTPUT_DIR not in path.parents
        and not path.name.startswith("_")
        and not path.name.endswith("_mask.png")
    ]


def _check(path: Path) -> tuple[int, int]:
    image = read(path)
    if image is None:
        print(f"{path.name}: unreadable")
        return 0, 0

    board, fruits, held, state = _run(image, path)

    if state != "ok":
        print(f"{path.name}: {state}")
        return 0, 0

    suspects = _suspects(fruits)
    print(
        f"{path.name}: {len(fruits)} fruits"
        + (f" / waiting {held}" if held else "")
        + (f" / {len(suspects)} suspicious" if suspects else "")
    )
    for fruit, reason in suspects:
        print(f"    {fruit.name:11s} ({fruit.x:3.0f},{fruit.y:3.0f}) {reason}")

    # Detections and masks are always compared, so combine them into one image per scene.
    mask = cv2.cvtColor(fruit_mask(board), cv2.COLOR_GRAY2BGR)
    write(OUTPUT_DIR / f"{path.stem}.png", cv2.hconcat([_annotate(board, fruits), mask]))

    return len(fruits), len(suspects)


def _run(image, path: Path) -> tuple:
    # A warped board does not show the waiting fruit, so that is not looked at.
    if path.name.endswith("_board.png"):
        return image, detect(image), "", "ok"

    result = localize(image)
    if not result.found or result.normalized is None:
        return None, [], "", "board not found"
    if result.blocked:
        return None, [], "", "covered by a dialog"

    return result.normalized, result.fruits or [], _held_label(result.held_fruit), "ok"


def _held_label(held: HeldResult | None) -> str:
    if held is None or held.radius is None:
        return "none"

    name = held.fruit.name if held.fruit is not None else "unclassified"

    return f"{name} x={held.x:.0f} r={held.radius:.0f} ({-(held.y or 0):.0f} above the top edge)"


def _suspects(fruits: list[Fruit]) -> list[tuple[Fruit, str]]:
    ratios = fruit_radius_ratios()
    found = []

    for fruit in fruits:
        expected = ratios[fruit.type] * NORMALIZED_WIDTH
        deviation = max(fruit.radius / expected, expected / fruit.radius)

        if fruit.confidence < SUSPECT_CONFIDENCE:
            found.append((fruit, f"confidence {fruit.confidence:.0f}%"))
        elif deviation > SUSPECT_RADIUS_RATIO:
            found.append((fruit, f"radius {fruit.radius:.0f}px (expected {expected:.0f}px)"))

    return found


def _annotate(board, fruits: list[Fruit]):
    output = board.copy()

    for fruit in fruits:
        center = (int(fruit.x), int(fruit.y))
        color = (0, 255, 0) if fruit.confidence >= SUSPECT_CONFIDENCE else (0, 165, 255)

        cv2.circle(output, center, max(2, int(fruit.radius)), color, 2)
        cv2.circle(output, center, 2, color, -1)

        label = f"{fruit.name} {fruit.confidence:.0f}"
        origin = (center[0] - int(fruit.radius), max(10, center[1] - int(fruit.radius) - 4))
        put_text(output, label, origin, color, scale=0.35, thickness=1)

    put_text(output, f"{len(fruits)}", (6, 18), (255, 255, 255), scale=0.6)

    return output


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
