FRUIT_NAMES = [
    "cherry",
    "strawberry",
    "grape",
    "dekopon",
    "orange",
    "apple",
    "pear",
    "peach",
    "pineapple",
    "melon",
    "watermelon",
]

# radius as a fraction of board width
FRUIT_RADIUS_RATIO = [
    (0.030, 0.048),
    (0.045, 0.060),
    (0.055, 0.075),
    (0.068, 0.088),
    (0.080, 0.102),
    (0.092, 0.115),
    (0.105, 0.130),
    (0.118, 0.145),
    (0.130, 0.160),
    (0.145, 0.178),
    (0.160, 0.210),
]

# OpenCV HSV: H 0-180, S/V 0-255
# Each entry is a list of (lower, upper) ranges; a fruit matches if any range fits.
FRUIT_HSV_RANGES = [
    # cherry - red
    [((0, 80, 80), (10, 255, 255)), ((170, 80, 80), (180, 255, 255))],
    # strawberry - red/green mix, slightly wider red
    [((0, 60, 60), (12, 255, 255)), ((170, 60, 60), (180, 255, 255))],
    # grape - purple
    [((125, 40, 40), (155, 255, 255))],
    # dekopon - orange
    [((8, 80, 80), (22, 255, 255))],
    # orange - orange
    [((10, 100, 100), (25, 255, 255))],
    # apple - deep red
    [((0, 100, 60), (10, 255, 255)), ((170, 100, 60), (180, 255, 255))],
    # pear - yellow-green
    [((25, 30, 80), (45, 255, 255))],
    # peach - pink
    [((145, 30, 120), (170, 180, 255)), ((0, 30, 120), (10, 180, 255))],
    # pineapple - yellow
    [((18, 80, 120), (35, 255, 255))],
    # melon - green
    [((35, 40, 60), (85, 255, 255))],
    # watermelon - green, usually largest
    [((35, 30, 40), (90, 255, 200))],
]
