"""screenshots/doko*.png の「どこに置くか」正解。

盤面は画像を見る。テストは localize → choose_x の通し。

- held / next … 画像を見た人間の正解。検出結果と突き合わせる
- expect_x   … 落とす列の許容範囲 (正規化盤 幅 400)。同種が複数でも曖昧にしない
- note       … なぜその列か (1手だけ。2手目は書かない)

増やす手順:
  1. screenshots/dokoN.png を置く
  2. 画像を見て held/next と expect_x を書く
  3. 落ちたら検出ミスかポリシーミスかを切り分けて直す (正解を緩めない)
"""

from __future__ import annotations

from typing import Any

DropCase = dict[str, Any]

EXPECTED_DROPS: dict[str, DropCase] = {
    "doko1.png": {
        "held": "strawberry",
        "next": "strawberry",
        "expect_x": (85, 120),
        "note": "グレープ(x≈101)の上。隣の床が空いても上へ",
    },
    "doko2.png": {
        "held": "dekopon",
        "next": "orange",
        "expect_x": (25.5, 30),
        "note": "左寄りのオレンジを押して右に転がしピーチにする",
    },
    "doko3.png": {
        "held": "orange",
        "next": "orange",
        "expect_x": (205, 225),
        "note": "リンゴの右側に置き、次のオレンジで梨にする",
    },
    "doko4.png": {
        "held": "grape",
        "next": "strawberry",
        "expect_x": (130, 200),
        "note": "左端へ滑らず、リンゴ〜ナシの間寄り",
    },
    "doko5.png": {
        "held": "orange",
        "next": "orange",
        "expect_x": (350, 367),
        "note": "桃の上",
    },
    "doko6.png": {
        "held": "orange",
        "next": "grape",
        "expect_x": (240, 265),
        "note": "既存オレンジ(x≈236)と合成するが、右に転がるようにしてパインにする",
    },
    "doko7.png": {
        "held": "grape",
        "next": "grape",
        "expect_x": (190, 220),
        "note": "デコポン(x≈216)を育てる",
    },
    "doko8.png": {
        "held": "orange",
        "next": "cherry",
        "expect_x": (130, 150),
        "note": "リンゴ(x≈120)の右側。メロン側へ乗せない",
    },
    "doko9.png": {
        "held": "dekopon",
        "next": "grape",
        "expect_x": (175, 374.5),
        "note": "オレンジ(x≈173)の右側~メロンの右上",
    },
    "doko10.png": {
        "held": "grape",
        "next": "strawberry",
        "expect_x": (22.7, 25),
        "note": "左端にのせて、次のイチゴでデコポンにする",
    },
    "doko11.png": {
        "held": "grape",
        "next": "cherry",
        "expect_x": (148, 150),
        "note": "イチゴ(x≈152)とデコポンの間におく。次のチェリーをころがしてオレンジにする",
    },
    "doko12.png": {
        "held": "dekopon",
        "next": "orange",
        "expect_x": (250, 270),
        "note": "既存デコポン(x≈274)と合成するが、左のオレンジとくっつける為に左寄り",
    },
    #TODO: doko13.png
}

# まだポリシーが解けない局面。strict xfail。直したら外す。
# 検出自体が壊れているときはここに入れず、テストを赤のままにする。
KNOWN_DROP_FAILURES: dict[str, str] = {
    "doko3.png": "リンゴ真上を選び、右側 (梨への寄せ) にならない",
    "doko4.png": "左の隙間に引き込まれて左端へ滑る",
    "doko5.png": "リンゴ列より右 (グレープ寄り) を選ぶ",
    "doko7.png": "デコポン列より左 (apple 寄り) を選ぶ",
    "doko8.png": "リンゴ真上を選び、右側にならない",
    "doko9.png": "オレンジ真上を選び、右側〜メロン右上にならない",
    "doko10.png": "メロン右肩へ寄せてしまう",
    "doko11.png": "高い山のデコポン上に積む",
    "doko12.png": "デコポン合成が右寄りで、左オレンジへの寄せ不足",
}
