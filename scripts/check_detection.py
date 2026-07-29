"""保存済みの画像をまとめて検出にかけ、結果を目で確かめられる形にする。

    python scripts/check_detection.py                    # screenshots と debug
    python scripts/check_detection.py debug              # 場所を指定する

*_board.png は warp 済みの盤面として、それ以外は画面全体として扱う。
検出結果とマスクを並べた画像を debug/check/ に書き出し、あわせて怪しい
検出を挙げる。しきい値をいじるたびにこれを回して、前回との差を見る。
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.draw import put_text
from src.imagefile import read, write
from src.vision.board import NORMALIZED_WIDTH, localize
from src.vision.classify import fruit_radius_ratios
from src.vision.fruits import detect, fruit_mask
from src.vision.held import HeldResult
from src.vision.state import Fruit

DEFAULT_SOURCES = ("screenshots", "debug")
OUTPUT_DIR = ROOT / "debug" / "check"

# 分類は必ず一番近い段階を選ぶので、半径は期待値の周りに収まる。
# それでもこれだけずれていたら、塊の切り出しを疑う。
SUSPECT_RADIUS_RATIO = 1.25
SUSPECT_CONFIDENCE = 50.0


def main(argv: list[str]) -> int:
    names = argv or list(DEFAULT_SOURCES)
    paths = sorted({path for name in names for path in _collect(ROOT / name)})

    if not paths:
        print(f"画像が見つからない: {', '.join(names)}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_fruits = 0
    total_suspects = 0
    for path in paths:
        fruits, suspects = _check(path)
        total_fruits += fruits
        total_suspects += suspects

    print(f"\n{len(paths)} 枚 / 検出 {total_fruits} 個 / 疑わしい {total_suspects} 個")
    print(f"書き出し先: {OUTPUT_DIR}")
    return 0


def _collect(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        return []

    # マスクは二値なので検出にかけても意味がない。先頭が _ のものは
    # 手で置いた一時ファイルとみなして飛ばす。
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
        print(f"{path.name}: 読めない")
        return 0, 0

    board, fruits, held, state = _run(image, path)

    if state != "ok":
        print(f"{path.name}: {state}")
        return 0, 0

    suspects = _suspects(fruits)
    print(
        f"{path.name}: {len(fruits)} 個"
        + (f" / 落下待ち {held}" if held else "")
        + (f" / 疑わしい {len(suspects)} 個" if suspects else "")
    )
    for fruit, reason in suspects:
        print(f"    {fruit.name:11s} ({fruit.x:3.0f},{fruit.y:3.0f}) {reason}")

    # 検出とマスクは必ず見比べるので、1 シーン 1 枚にまとめる。
    mask = cv2.cvtColor(fruit_mask(board), cv2.COLOR_GRAY2BGR)
    write(OUTPUT_DIR / f"{path.stem}.png", cv2.hconcat([_annotate(board, fruits), mask]))

    return len(fruits), len(suspects)


def _run(image, path: Path) -> tuple:
    # warp 済みの盤面には落下待ちフルーツが写っていないので、そこは見ない。
    if path.name.endswith("_board.png"):
        return image, detect(image), "", "ok"

    result = localize(image)
    if not result.found or result.normalized is None:
        return None, [], "", "盤面が見つからない"
    if result.blocked:
        return None, [], "", "ダイアログで覆われている"

    return result.normalized, result.fruits or [], _held_label(result.held_fruit), "ok"


def _held_label(held: HeldResult | None) -> str:
    if held is None or held.radius is None:
        return "なし"

    name = held.fruit.name if held.fruit is not None else "分類できず"

    return f"{name} x={held.x:.0f} r={held.radius:.0f} (上辺から {-(held.y or 0):.0f})"


def _suspects(fruits: list[Fruit]) -> list[tuple[Fruit, str]]:
    ratios = fruit_radius_ratios()
    found = []

    for fruit in fruits:
        expected = ratios[fruit.type] * NORMALIZED_WIDTH
        deviation = max(fruit.radius / expected, expected / fruit.radius)

        if fruit.confidence < SUSPECT_CONFIDENCE:
            found.append((fruit, f"信頼度 {fruit.confidence:.0f}%"))
        elif deviation > SUSPECT_RADIUS_RATIO:
            found.append((fruit, f"半径 {fruit.radius:.0f}px (期待 {expected:.0f}px)"))

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
