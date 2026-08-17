# AGENTS.md — この repo での作業のしかた

## ドキュメントの役割分担

3 つとも別の役をもつ。**同じことを 2 か所に書かない**。

- **AGENTS.md（この file）** … 作業の手順。コマンド、コード規約、変更の進め方、
  repo 固有の罠。「次に触る人が同じ手順を踏めるか」で書く。数字は書かない
- **[NOTES.md](NOTES.md)** … ゲームと方策のドメイン知識・実験ログ。測って分かったこと、
  失敗した試行、現行ペナルティ規則、未着手の課題。「なぜこの設計か」「何点だったか」
- **[README.md](README.md)** … 人間向けの入口（英語）。setup・実行方法・layout

迷ったときの振り分け:

| 書きたいこと | 行き先 |
|---|---|
| test / lint / script の叩き方 | AGENTS.md |
| score・n・CI などの測定結果と結論 | NOTES.md |
| ペナルティの重みと意味、方策の設計判断 | NOTES.md |
| 「A/B は `_apply_variant` に差す」のような手順 | AGENTS.md（出た数字は NOTES.md） |
| 効果がなかった試行、やらないと決めたこと | NOTES.md |
| module / script が増えた・名前が変わった | README.md の Layout（規約が要るなら AGENTS.md） |

AGENTS.md から詳細を書きたくなったら、書かずに NOTES.md の節へリンクする。

## 環境

- Windows / PowerShell。`.venv` は repo 直下にあり、`python` と `pytest` はそこを指す
- **`ruff` / `basedpyright` / `vulture` は venv に入っていない**。グローバルの
  `Python310\Scripts` にあるので、venv を有効にしたままでも素の名前で叩ける
- `main.py` は実物のゲーム画面と入力を掴む。エージェントは勝手に起動しない。
  動作確認は `scripts/` の sim 側で行う

## よく使うコマンド

```powershell
pytest                 # 約 160 件・8 秒弱。差分を入れたら毎回
basedpyright           # 型。既存エラーが残っていることがあるので、自分の差分由来かを確認する
ruff check --fix       # 未使用 import / 変数だけ（select = F401, F841）
vulture                # 未参照の関数・定数。誤検知あり、消す前に呼び出し元を確認
```

sim の評価・A/B・学習の実行例は README の Scripts と
[NOTES.md の学習まわり](NOTES.md#学習まわり)。長いものは数時間かかる。

## コード規約

- docstring とコメントは**日本語**、識別子と commit message は**英語**
- コメントは「何をしているか」ではなく**なぜその値・その形なのか**を書く
  （実測値、踏んだ罠、捨てた代替案）。`src/policy.py` の冒頭と定数群が見本
- `from __future__ import annotations` ＋ 型注釈。`typeCheckingMode = "basic"`
- module 内だけで使うものは `_` 前置。module をまたいで呼ぶものだけ public にする
- **`src/penalties.py` の重み・関数を `from .penalties import X` で束縛しない。**
  `src/policy.py` のように `pen.X` で参照する。`compare_policy.py` の A/B は
  module 属性を書き換えて差し替えるので、束縛すると A と B が同じ方策で走ってしまう
- `scripts/*.py` は先頭の `sys.path.insert` → `from scripts._bootstrap import ROOT`
  の形を踏襲する（path を足す関数は作らない）
- test は具体手順を固定しない。合成・危険回避・事故防止といった**方策の性質**を
  assert する（`tests/test_policy.py` の冒頭方針）。落下物理は `tests/test_sim_physics.py`

## 方策・学習を触るとき

- **数字を出す前に [測定のしかた](NOTES.md#測定のしかた何度も踏んでいる罠) を読む。**
  score のノイズは大きく、平均の増減だけでは何も言えない
- A/B は `scripts/compare_policy.py` の `_apply_variant` に変更を差して走らせる。
  恒久化するときは変種を戻し、**ON/OFF トグルはコードに残さない**。
  別 commit と比べたくなったら [git](#git) の worktree の項
- ペナルティ規則を足す・消す・重みを変えたら、NOTES.md の
  [現行ペナルティルール一覧](NOTES.md#現行ペナルティルール一覧) を同じ差分で直す
- **効果がなかった試行も NOTES.md に残す。** 内容・n・結論・revert したことまで書く。
  同じ道を二度歩かないための記録で、成功だけを残すと意味がない
- 純粋な refactor は play が変わらないことを確かめる
  （同一 seed で数百手の x / score / penalties / merge が一致するか）

## git

git の操作はこの節にまとめる。他の節に散らさない。

**branch**

- **`master` で作業しない。** 手を動かす前に branch を切る:
  `git switch -c <topic>`（例: `fix-wall-friction`, `docs-agents-md`）
- PR 先は `main`

**commit**

- branch にいる限り、**許可を取らずに commit してよい**。区切りごとに入れる。
  逆に `master` にいると気づいたら、commit せず先に branch を切る
- Conventional Commits（`feat:` `fix:` `refactor:` `docs:` `perf:`）、英語、
  要約は命令形の 1 行
- 本文に**なぜそうしたか**と**どう確かめたか**を書く。純粋な refactor なら
  play 不変をどう検証したかまで。直近の commit が見本
- commit 前に `pytest`（8 秒弱）。方策を変えたなら数字は
  [測定のしかた](NOTES.md#測定のしかた何度も踏んでいる罠) の作法で

**追跡するもの・しないもの**

- `screenshots/` は目視で起こした正解データなので追跡する
- `artifacts/` `debug/` は実行のたびに変わるので ignore

**別 commit と比べるとき**

- 生きた作業ツリーに `git stash` / `git checkout` を含む自動化を書かない。
  過去に未 commit の差分を stash に取り残す事故を起こした
  （`git fsck --unreachable` で復旧）
- `git worktree add` で隔離した別ツリーを使う
