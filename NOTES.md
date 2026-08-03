# 既知の問題・あとで直すこと

## 後回し

- **`10.png` ビジョン**: 同系色マスク融合＋枠外イチゴ。切り出し再設計が必要で、方策／held よりコスパが悪い
- **学習エピソード長**: `max_steps` を増やし、`episodes` を減らす（長い対局を少なめに）
- **序盤の寄せすぎ**: 大小順・近接で隣り合う実（例: リンゴとデコポン）の間に隙間が無さすぎる。オレンジを積むときに並び順を崩す手しか残らない。ある程度あけたい
- **床埋め後の大ツモ**: 床が埋まったあと、オレンジ／デコポンなど大きいツモを L（最大実）側ではなく小さい側へ置いて崩れる。大側・小側の置き分けが途中から効かなくなる

## 着手タイミング

- 動いている盤で x を決めない。先読みより静止待ちを優先 (`src/settle.py`)
- 瞬間速度だけでなく、静穏中の横ドリフトでも creep を待つ

## 方策 (bootstrap)

- `src/policy.py` は RL 前の薄い方策。合成・危険高さ・埋め込み・薄い大小順・転がり／弾かれの事故防止だけ
- 落下・衝突・合成の物理は pymunk (`src/sim_physics.py`。UT は `tests/test_sim_physics.py`)。`choose_x` も同じ `simulate_drop` を採点に使う
- 手の採点は `eval = score - penalties`。加点は本家点だけで、危険高さ・事故・埋め込みは減点で表す
- next 先読み: held の eval 上位 `HELD_TOP` 本だけを、刻み `NEXT_CANDIDATE_STEP` の候補で再評価（`NEXT_DISCOUNT` 掛け）。物理が重いので held より粗い
- 埋め込みが主減点。同種ペア待ちを、より大きい異種で直上・肩から塞ぐ手を強く引く
- 異種中央狙い (`FOREIGN_AIM`): 真下の実が異種かつ中心帯 (`FOREIGN_AIM_CENTER_FRAC`) なら `FOREIGN_AIM_PENALTY`。真下が同種なら OK。`merges` では切らない（異種真上から転がって合体する抜け道を塞ぐ）。同種過多 (`EXCESS_SAME` = 超過1個あたり20)。谷・肩への異種積みは禁じない
- 大きい実の谷育成は、谷に同種があるときか held/next が両方とも壁よりひとつ小さいときに限る。それ以外の隙間埋めは通常減点 (`GAP_JUNK` は廃止のまま)
- レイアウト: 大実どうしは近接。大側 (`sign`) で端に付いた L の外側かつ L 中心より下の角ポケットを強く減点 (`_big_layout_penalty`)。`wrong_side_roll` も同じ大側床の転がり事故用
- 持たないもの: 押し込み合成、復元押し、連鎖隙間空け、一段上への強制寄り、梯子発火のハードコード
- 具体手順の UT は増やさない。壊れたら事故防止か観測側を見る
- 探索コストはほぼ `simulate_drop` の呼び出し数。`HELD_TOP` / `NEXT_CANDIDATE_STEP` が実行時間を決める（旧 8/16 は 1 手 3.8 秒で収集が回らなかった。2/32 で 1.2 秒・score -3.4%）
- `CANDIDATE_STEP` は粗くしない。20 にすると危険な山の真上がグリッドに乗って `test_avoids_dangerous_tall_stack` が落ちる。速度は先読み側で稼ぐ
- `SLEEP_FRAMES` を削るのは無効。単発の `choose_x` は速くなるが、盤の収まりが変わって次手以降が重くなり、エピソード全体ではむしろ遅い（25 で実測）。物理の忠実度 (`SimEnv` と共有) も落ちる

## 学習まわり

- **score / penalties**: `score` は本家の合成点 (1〜65、減点なし)、`penalties` は事故・悪手の減点。`eval = score - penalties` は **bootstrap の手選び専用**（`choose_x`）。生徒の良し悪し・保存・ログは本家点 `score`（同点なら手数）。**RL の報酬も score**（密な減点は報酬にしない）
- `src/reward.py`: `merge_score(merge_types)` が本家と同じ合成点のみ (cherry→0 … watermelon 55、ダブル消去 65)。生存加点・死亡減点なし。エピソード終了は従来どおり (負けライン / ダブル消去)
- `src/encode.py`: 固定長観測ベクトル
- `src/sim_env.py`: 画面なし落下 sim (`sim_physics.simulate_drop`)。`SimStep` は本家点 `score` のみ（累積 eval は持たない）
- 評価: `python scripts/eval_policy.py` (`--policy bootstrap|learned`。`--workers` 既定=論理コア/2)
- 学習: `python scripts/train_sim.py` (収集 → オフライン BC。既定 max-steps=100 は打ち切りであり負けラインではない)。best は score → 手数 → match
- 教師収集は `ProcessPool` 並列 (既定 workers=論理コア/2。9700X なら 8。`--workers 1` で直列)
- `src/agent.py`: 離散列 32 ビン / hidden 128 の MLP (旧 20/64 の npz は再学習が必要)
- 実プレイ: `python main.py` (npz があれば既定 learned。`L` で bootstrap 切替、`--policy bootstrap`)

## 予定: RL (REINFORCE)

- いまはまだ早い。BC が浅いと素の REINFORCE は壊しやすい（過去に確認済み）
- 足す条件: `match` が高め（目安 60–70%+）かつ生徒の `score` が bootstrap に近い
- やり方: BC 完了後に短い微調整だけ（例: `--episodes 50 --lr 0.002`）。既定はオフのまま
- それまでは BC（収集量・epoch）を厚くする方が先
