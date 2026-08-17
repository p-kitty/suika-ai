"""screenshots/ に置いた画像の正解。

盤面を目で見て起こしたもの。座標は warp 後の盤面 (400x500) でのフルーツの
中心で、目分量なので数 px はずれている。突き合わせ側で半径ぶんの許容を
持たせてあるので、この程度のずれは問題にならない。

盤面の四隅は壁の内側基準 (board.py の WALL_INSET_X/Y_RATIO) に揃えてある。

並びは上から下。増やすときは screenshots に画像を置いて
scripts/check_detection.py を回し、debug/check/ の絵を見ながら書く。
"""

# 盤面がダイアログで覆われていて、そもそも読めない画像。
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
    # 少し上を向いた状態でもボード下付近のフルーツが認識されることをテスト。
    "8.png": [
        ("grape", 266, 477),
        ("cherry", 141, 490),
    ],
    # 同上。
    "9.png": [
        ("orange", 224, 468),
        ("dekopon", 31, 476),
        ("dekopon", 371, 476),
    ],
    # 盤面がほぼ埋まった状態。上端のフルーツは縁の帯で切れている。
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

# 雲が持っている、次に落ちるフルーツ。種類と、正規化した盤面での落とす列
# (0〜400)。7.png はダイアログで覆われていて読まない。
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
    # 盤面を左から斜めに見た視点。
    "11.png": ("strawberry", 36),
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
    # 枠上にはみ出したイチゴ (272,-36) は縁の帯でマスクから落ち、候補にも出ない。
    # peach の半径も塊のものになるため apple と読む。
    # マスクを見える輪郭で切れば分かれるが、watermelon の縞や melon の網目も
    # 一緒に切れて大きいフルーツが砕けるため、切り出しの作り直しが必要。
    "10.png": "同系色の融合と、枠外にはみ出したイチゴが縁の帯で落ちる",
    # 盤面がほぼ満杯で、壁の内側 (本当の壁基準に直した後) にはもう地の色が
    # ほとんど残っていない。_fit_background の種がフルーツだらけになって
    # 当てはめが破綻し、_saturation_mask に落ちて同系色が融合・脱落する。
    # 種を壁際の帯に限らず盤面全体から取り直すなど、当てはめ自体の作り直しが必要。
    "3.png": "盤面がほぼ満杯で下地の当てはめが破綻し、彩度だけのマスクに落ちる",
}
