"""screenshots/ に置いた画像の正解。

盤面を目で見て起こしたもの。座標は warp 後の盤面 (400x500) でのフルーツの
中心で、目分量なので数 px はずれている。突き合わせ側で半径ぶんの許容を
持たせてあるので、この程度のずれは問題にならない。

並びは上から下。増やすときは screenshots に画像を置いて
scripts/check_detection.py を回し、debug/check/ の絵を見ながら書く。
"""

# 盤面がダイアログで覆われていて、そもそも読めない画像。
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
    # 少し上を向いた状態でもボード下付近のフルーツが認識されることをテスト。
    "8.png": [
        ("grape", 255, 453),
        ("cherry", 151, 464),
    ],
    # 同上。
    "9.png": [
        ("orange", 220, 445),
        ("dekopon", 59, 452),
        ("dekopon", 343, 452),
    ],
    # 盤面がほぼ埋まった状態。上端のフルーツは縁の帯で切れている。
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

# 雲が持っている、次に落ちるフルーツ。種類と、正規化した盤面での落とす列
# (0〜400)。7.png はダイアログで覆われていて読まない。
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
    # 盤面を左から斜めに見た視点。
    "11.png": ("strawberry", 63),
}

# next の泡の中身。落下待ちのさらに次に来るフルーツ。
# 泡は盤面から離れて画面の端寄りに写るので、視点を振ったときに狂いやすい。
# 11.png は斜めから見た視点で、盤面幅を基準に測っていた頃は orange と誤った。
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

# まだ直っていない取り違え。直したらこの印を外す。
KNOWN_FAILURES = {
    # 上端の赤系 (peach/dekopon/apple x2/strawberry/cherry) が触れ合うとマスクが
    # 一つの塊に融合する。融合した塊の距離変換には内側のピークが立たない。
    # 枠上にはみ出したイチゴ (260,-5) は縁の帯でマスクから落ち、候補にも出ない。
    # peach の半径も塊のものになるため apple と読む。
    # マスクを見える輪郭で切れば分かれるが、watermelon の縞や melon の網目も
    # 一緒に切れて大きいフルーツが砕けるため、切り出しの作り直しが必要。
    "10.png": "同系色の融合と、枠外にはみ出したイチゴが縁の帯で落ちる",
}
