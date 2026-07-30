"""Ground truth for 'where to place' in screenshots/doko*.png.

Look at the images for the board. The test runs localize → choose_x end to end.

- held / next … the human ground truth from looking at the image. Compared against detection results
- expect_x   … the allowed range of the drop column (normalized board, width 400). Not ambiguous even with multiple same types
- note       … why that column (one move only; the second move is not written)

How to add more:
  1. put screenshots/dokoN.png
  2. look at the image and write held/next and expect_x
  3. when it fails, separate detection mistakes from policy mistakes and fix them (do not loosen the ground truth)
"""

from __future__ import annotations

from typing import Any

DropCase = dict[str, Any]

EXPECTED_DROPS: dict[str, DropCase] = {
    "doko1.png": {
        "held": "strawberry",
        "next": "strawberry",
        "expect_x": (85, 145),
        "note": "toward the big side of the grape (x≈101). Avoid directly above the center of a different type",
    },
    "doko2.png": {
        "held": "dekopon",
        "next": "orange",
        "expect_x": (25.5, 30),
        "note": "push the orange on the left, roll it right and make a peach",
    },
    "doko3.png": {
        "held": "orange",
        "next": "orange",
        "expect_x": (205, 225),
        "note": "place right of the apple, and make a pear with the next orange",
    },
    "doko4.png": {
        "held": "grape",
        "next": "strawberry",
        "expect_x": (130, 200),
        "note": "do not slide to the left edge; toward between the apple and pear",
    },
    "doko5.png": {
        "held": "orange",
        "next": "orange",
        "expect_x": (350, 367),
        "note": "on the peach",
    },
    "doko6.png": {
        "held": "orange",
        "next": "grape",
        "expect_x": (240, 265),
        "note": "merge with the existing orange (x≈236), rolling right to make a pineapple",
    },
    "doko7.png": {
        "held": "grape",
        "next": "grape",
        "expect_x": (190, 220),
        "note": "grow the dekopon (x≈216)",
    },
    "doko8.png": {
        "held": "orange",
        "next": "cherry",
        "expect_x": (130, 150),
        "note": "right of the apple (x≈120). Do not put it on the melon side",
    },
    "doko9.png": {
        "held": "dekopon",
        "next": "grape",
        "expect_x": (175, 374.5),
        "note": "right of the orange (x≈173) to the upper right of the melon",
    },
    "doko10.png": {
        "held": "grape",
        "next": "strawberry",
        "expect_x": (22.7, 25),
        "note": "put it at the left edge, and make a dekopon with the next strawberry",
    },
    "doko11.png": {
        "held": "grape",
        "next": "cherry",
        "expect_x": (148, 150),
        "note": "place between the strawberry (x≈152) and the dekopon. Roll the next cherry to make an orange",
    },
    "doko12.png": {
        "held": "dekopon",
        "next": "orange",
        "expect_x": (250, 270),
        "note": "merge with the existing dekopon (x≈274), but lean left to join the orange on the left",
    },
    #TODO: doko13.png
}

# Positions the policy cannot solve yet. strict xfail. Remove once fixed.
# When detection itself is broken, do not put it here; leave the test red.
KNOWN_DROP_FAILURES: dict[str, str] = {
    "doko4.png": "gets pulled into the left gap and slides to the left edge",
    "doko5.png": "chooses right of the apple column (toward the grape)",
    "doko7.png": "chooses left of the dekopon column (toward the apple)",
    "doko10.png": "pushes toward the melon's right shoulder",
    "doko11.png": "stacks on the dekopon on a tall pile",
    "doko12.png": "the dekopon merge leans right, not pushed enough toward the orange on the left",
}
