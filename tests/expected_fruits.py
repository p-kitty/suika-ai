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
    "8.png": [
        ("grape", 255, 453),
        ("cherry", 151, 464),
    ],
    "9.png": [
        ("orange", 220, 445),
        ("dekopon", 59, 452),
        ("dekopon", 343, 452),
    ],
    # A nearly full board. Fruits at the top edge are cut by the edge band.
    "10.png": [
        ("dekopon", 234, 44),
        ("cherry", 340, 45),
        ("apple", 293, 55),
        ("peach", 127, 62),
        ("strawberry", 50, 70),
        ("strawberry", 269, 79),
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
}

# Misreads not yet fixed. Remove the mark once fixed.
KNOWN_FAILURES = {
    # The 7 at the top (peach/dekopon/apple x2/strawberry x2/cherry) are all reddish, and
    # when they touch the mask fuses into one blob. The distance transform of a fused blob has no peaks
    # for the fruits inside, so 2 of the 4 missed never even become candidates.
    # The peach's radius also becomes the blob's, so it is read as an apple.
    # Cutting the mask by visible contours would separate them, but the stripes of watermelon and the net of melon
    # get cut too and big fruits shatter, so the segmentation needs rebuilding.
    "10.png": "when similar-colored fruits touch the mask fuses and inner peaks do not rise",
}
