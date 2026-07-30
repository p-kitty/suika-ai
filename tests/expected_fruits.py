"""Ground truth for the images in screenshots/.

Transcribed by looking at the board. Coordinates are fruit centers on the warped board (400x500),
estimated by eye, so a few px off. The matching side allows a tolerance of the radius,
so this much offset is not a problem.

Ordered top to bottom. To add more, put an image in screenshots, run
scripts/check_detection.py, and write it while looking at the pictures in debug/check/.
"""

# An image where a dialog covers the board, so it cannot be read at all.
BLOCKED = ("7.png",)

EXPECTED = {
    "1.png": [
        ("orange", 63, 231),
        ("orange", 304, 232),
        ("grape", 241, 272),
        ("grape", 344, 273),
        ("dekopon", 199, 286),
        ("strawberry", 161, 310),
        ("apple", 290, 321),
        ("peach", 90, 334),
        ("grape", 338, 355),
        ("pineapple", 204, 400),
        ("apple", 308, 424),
        ("orange", 66, 440),
    ],
    "2.png": [
        ("cherry", 354, 107),
        ("orange", 333, 153),
        ("orange", 180, 226),
        ("orange", 64, 232),
        ("melon", 285, 273),
        ("watermelon", 126, 364),
        ("dekopon", 306, 448),
        ("grape", 241, 450),
        ("strawberry", 347, 459),
        ("cherry", 43, 464),
    ],
    "3.png": [
        ("apple", 159, 64),
        ("strawberry", 62, 86),
        ("peach", 308, 100),
        ("cherry", 43, 108),
        ("pear", 210, 152),
        ("pineapple", 99, 171),
        ("cherry", 352, 206),
        ("melon", 285, 273),
        ("watermelon", 125, 366),
        ("dekopon", 306, 447),
        ("grape", 257, 449),
        ("strawberry", 346, 458),
        ("cherry", 43, 464),
    ],
    "4.png": [
        ("strawberry", 305, 296),
        ("grape", 341, 300),
        ("dekopon", 225, 339),
        ("peach", 149, 385),
        ("pineapple", 299, 400),
        ("orange", 65, 416),
        ("dekopon", 217, 443),
        ("cherry", 43, 464),
    ],
    "5.png": [
        ("orange", 255, 285),
        ("orange", 334, 293),
        ("dekopon", 57, 300),
        ("apple", 192, 333),
        ("pineapple", 287, 397),
        ("pineapple", 98, 400),
        ("dekopon", 204, 445),
        ("cherry", 44, 463),
        ("cherry", 355, 466),
    ],
    "6.png": [
        ("cherry", 353, 90),
        ("strawberry", 300, 119),
        ("dekopon", 340, 126),
        ("peach", 90, 147),
        ("dekopon", 184, 201),
        ("melon", 285, 239),
        ("watermelon", 125, 332),
        ("dekopon", 340, 447),
        ("grape", 75, 454),
        ("cherry", 42, 466),
    ],
    # Tests that fruits near the bottom of the board are recognized even when looking slightly upward.
    "8.png": [
        ("grape", 255, 453),
        ("cherry", 151, 464),
    ],
    # Same as above.
    "9.png": [
        ("orange", 220, 445),
        ("dekopon", 59, 452),
        ("dekopon", 343, 452),
    ],
    # A nearly full board. Fruits at the top edge are cut by the edge band.
    "10.png": [
        ("strawberry", 260, -5),
        ("dekopon", 234, 44),
        ("cherry", 340, 45),
        ("apple", 293, 55),
        ("peach", 127, 62),
        ("strawberry", 50, 70),
        ("strawberry", 252, 78),
        ("strawberry", 345, 80),
        ("apple", 197, 85),
        ("dekopon", 63, 118),
        ("pineapple", 294, 158),
        ("watermelon", 125, 233),
        ("dekopon", 243, 250),
        ("strawberry", 344, 267),
        ("grape", 342, 312),
        ("cherry", 351, 368),
        ("cherry", 162, 370),
        ("melon", 252, 380),
        ("grape", 86, 383),
        ("strawberry", 48, 398),
        ("strawberry", 346, 407),
        ("apple", 137, 428),
        ("dekopon", 73, 442),
        ("grape", 339, 454),
        ("cherry", 42, 465),
    ],
    "11.png": [
        ("cherry", 42, 356),
        ("strawberry", 76, 359),
        ("orange", 222, 374),
        ("dekopon", 113, 378),
        ("grape", 53, 396),
        ("peach", 307, 406),
        ("apple", 164, 429),
        ("orange", 92, 438),
        ("cherry", 44, 466),
    ],
}

# The next fruit to fall, held by the cloud. Its type and drop column on the normalized board
# (0-400). 7.png is covered by a dialog and not read.
EXPECTED_HELD = {
    "1.png": ("grape", 203),
    "2.png": ("strawberry", 224),
    "3.png": ("dekopon", 239),
    "4.png": ("cherry", 116),
    "5.png": ("cherry", 352),
    "6.png": ("orange", 77),
    "8.png": ("cherry", 194),
    "9.png": ("strawberry", 248),
    "10.png": ("orange", 214),
    # A view of the board from the left at an angle.
    "11.png": ("strawberry", 63),
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
    # The strawberry sticking out above the frame (260,-5) drops out of the mask with the edge band and never becomes a candidate.
    # The peach's radius also becomes the blob's, so it is read as an apple.
    # Cutting the mask by visible contours would separate them, but the stripes of watermelon and the net of melon
    # get cut too and big fruits shatter, so the segmentation needs rebuilding.
    "10.png": "similar colors fuse, and a strawberry sticking out of the frame drops with the edge band",
}
