"""Ground truth for the images in screenshots/.

Transcribed by looking at the board. Coordinates are fruit centers on the warped board (400x500),
estimated by eye, so a few px off. The matching side allows a tolerance of the radius,
so this much offset is not a problem.

The board corners are aligned to the inside-of-the-wall basis (WALL_INSET_X/Y_RATIO in board.py).

Ordered top to bottom. To add more, put an image in screenshots, run
scripts/check_detection.py, and write it while looking at the pictures in debug/check/.
"""

# An image where a dialog covers the board, so it cannot be read at all.
BLOCKED = ("7.png",)

EXPECTED = {
    "1.png": [
        ("orange", 36, 229),
        ("orange", 325, 230),
        ("grape", 249, 275),
        ("grape", 372, 276),
        ("dekopon", 199, 290),
        ("strawberry", 153, 317),
        ("apple", 308, 330),
        ("peach", 68, 344),
        ("grape", 365, 368),
        ("pineapple", 205, 418),
        ("apple", 329, 445),
        ("orange", 40, 463),
    ],
    "2.png": [
        ("cherry", 384, 90),
        ("orange", 359, 141),
        ("orange", 176, 223),
        ("orange", 37, 230),
        ("melon", 302, 276),
        ("watermelon", 111, 378),
        ("dekopon", 327, 472),
        ("grape", 249, 474),
        ("strawberry", 376, 484),
        ("cherry", 12, 490),
    ],
    "3.png": [
        ("apple", 151, 42),
        ("strawberry", 35, 66),
        ("peach", 329, 82),
        ("cherry", 12, 91),
        ("pear", 212, 140),
        ("pineapple", 79, 162),
        ("cherry", 382, 201),
        ("melon", 302, 276),
        ("watermelon", 110, 380),
        ("dekopon", 327, 471),
        ("grape", 268, 473),
        ("strawberry", 375, 483),
        ("cherry", 12, 490),
    ],
    "4.png": [
        ("strawberry", 326, 302),
        ("grape", 369, 306),
        ("dekopon", 230, 350),
        ("peach", 139, 401),
        ("pineapple", 319, 418),
        ("orange", 38, 436),
        ("dekopon", 220, 466),
        ("cherry", 12, 490),
    ],
    "5.png": [
        ("orange", 266, 289),
        ("orange", 360, 298),
        ("dekopon", 29, 306),
        ("apple", 190, 343),
        ("pineapple", 304, 415),
        ("pineapple", 78, 418),
        ("dekopon", 205, 468),
        ("cherry", 13, 488),
        ("cherry", 386, 492),
    ],
    "6.png": [
        ("cherry", 383, 71),
        ("strawberry", 320, 103),
        ("dekopon", 368, 111),
        ("peach", 68, 135),
        ("dekopon", 181, 195),
        ("melon", 302, 238),
        ("watermelon", 110, 342),
        ("dekopon", 368, 471),
        ("grape", 50, 478),
        ("cherry", 11, 492),
    ],
    # Tests that fruits near the bottom of the board are recognized even when looking slightly upward.
    "8.png": [
        ("grape", 266, 477),
        ("cherry", 141, 490),
    ],
    # Same as above.
    "9.png": [
        ("orange", 224, 468),
        ("dekopon", 31, 476),
        ("dekopon", 371, 476),
    ],
    # A nearly full board. Fruits at the top edge are cut by the edge band.
    "10.png": [
        ("strawberry", 272, -36),
        ("dekopon", 241, 19),
        ("cherry", 368, 20),
        ("apple", 311, 32),
        ("peach", 113, 40),
        ("strawberry", 20, 48),
        ("strawberry", 262, 57),
        ("strawberry", 374, 60),
        ("apple", 196, 65),
        ("dekopon", 36, 102),
        ("pineapple", 313, 147),
        ("watermelon", 110, 231),
        ("dekopon", 252, 250),
        ("strawberry", 372, 269),
        ("grape", 370, 319),
        ("cherry", 381, 382),
        ("cherry", 154, 384),
        ("melon", 262, 396),
        ("grape", 64, 399),
        ("strawberry", 18, 416),
        ("strawberry", 375, 426),
        ("apple", 125, 449),
        ("dekopon", 48, 465),
        ("grape", 366, 478),
        ("cherry", 11, 491),
    ],
    "11.png": [
        ("cherry", 11, 369),
        ("strawberry", 52, 372),
        ("orange", 226, 389),
        ("dekopon", 96, 393),
        ("grape", 24, 414),
        ("peach", 328, 425),
        ("apple", 157, 450),
        ("orange", 71, 460),
        ("cherry", 13, 492),
    ],
}

# The next fruit to fall, held by the cloud. Its type and drop column on the normalized board
# (0-400). 7.png is covered by a dialog and not read.
EXPECTED_HELD = {
    "1.png": ("grape", 204),
    "2.png": ("strawberry", 229),
    "3.png": ("dekopon", 247),
    "4.png": ("cherry", 99),
    "5.png": ("cherry", 382),
    "6.png": ("orange", 53),
    "8.png": ("cherry", 193),
    "9.png": ("strawberry", 258),
    "10.png": ("orange", 217),
    # A view of the board from the left at an angle.
    "11.png": ("strawberry", 36),
}

# The contents of the next bubble. The fruit that comes after the waiting one.
# The bubble appears away from the board toward the edge of the screen, so it goes wrong easily when the view swings.
# 11.png is an angled view; back when it was measured against the board width it was misread as orange.
EXPECTED_NEXT = {
    "1.png": "grape",
    "2.png": "orange",
    "3.png": "strawberry",
    "4.png": "grape",
    "5.png": "orange",
    "6.png": "cherry",
    "8.png": "cherry",
    "9.png": "cherry",
    "10.png": "cherry",
    "11.png": "dekopon",
}

# Misreads not yet fixed. Remove the mark once fixed.
KNOWN_FAILURES = {
    # When the reddish fruits at the top (peach/dekopon/apple x2/strawberry/cherry) touch, the mask
    # fuses into one blob. The distance transform of a fused blob has no inner peaks.
    # The strawberry sticking out above the frame (272,-36) drops out of the mask with the edge band and never becomes a candidate.
    # The peach's radius also becomes the blob's, so it is read as an apple.
    # Cutting the mask by visible contours would separate them, but the stripes of watermelon and the net of melon
    # get cut too and big fruits shatter, so the segmentation needs rebuilding.
    "10.png": "similar colors fuse, and a strawberry sticking out of the frame drops with the edge band",
    # The board is nearly full, and inside the walls (after correcting to the true wall basis) hardly any background color
    # remains. _fit_background's seeds are all fruit, so
    # the fit breaks down and falls back to _saturation_mask, where similar colors fuse or drop out.
    # The fit itself needs rebuilding, such as taking seeds from the whole board rather than only the band by the walls.
    "3.png": "the board is nearly full, the background fit breaks and it falls back to a saturation-only mask",
}
