# 既知の問題・あとで直すこと

## 後回し

- **`10.png` ビジョン**: 同系色マスク融合＋枠外イチゴ。切り出し再設計が必要で、方策／held よりコスパが悪い
- **学習エピソード長**: `max_steps` を増やし、`episodes` を減らす（長い対局を少なめに）

## 着手タイミング

- 動いている盤で x を決めない。先読みより静止待ちを優先 (`src/settle.py`)
- 瞬間速度だけでなく、静穏中の横ドリフトでも creep を待つ

## 方策 (bootstrap)

- `src/policy.py` は RL 前の薄い方策。合成・危険高さ・埋め込み・薄い大小順・転がり／弾かれの事故防止だけ
- 手の採点は `eval = score - penalties`。加点は本家点だけで、積み上げ・事故・埋め込みは減点で表す
- 埋め込みが主減点。同種ペア待ちを、より大きい異種で直上・肩から塞ぐ手を強く引く
- 異種中央狙い (`FOREIGN_AIM`) と同種過多 (`EXCESS_SAME` = 超過1個あたり20) で崩し・遅延合成を抑える。谷・肩への異種積みは禁じない
- 大きい実の谷は育成枠。`GAP_JUNK` は廃止。谷着地は高さ / wrong_side / ideal / 大小順で潰さない
- 持たないもの: 押し込み合成、復元押し、連鎖隙間空け、一段上への強制寄り
- 具体手順の UT は増やさない。壊れたら事故防止か観測側を見る

## 学習まわり

- **score / eval**: `score` は本家の合成点 (1〜65、減点なし)、`penalties` は事故・悪手の減点、`eval = score - penalties`。方策の手選びと生徒の良し悪しは eval、**RL の報酬は score のまま**（密な減点は報酬にしない）
- `src/reward.py`: `merge_score(merge_types)` が本家と同じ合成点のみ (cherry→0 … watermelon 55、ダブル消去 65)。生存加点・死亡減点なし。エピソード終了は従来どおり (負けライン / ダブル消去)
- `src/encode.py`: 固定長観測ベクトル
- `src/sim_env.py`: 画面なし落下 sim (`policy.simulate_drop`)。`SimStep` は `score` と `eval_score` を持つ
- 評価: `python scripts/eval_policy.py` (`--policy bootstrap|learned`。`--workers` 既定=論理コア/2)
- 学習: `python scripts/train_sim.py` (収集 → オフライン BC。既定 max-steps=100 は打ち切りであり負けラインではない)。ログは score と eval を併記し、best は eval で選ぶ
- 教師収集は `ProcessPool` 並列 (既定 workers=論理コア/2。9700X なら 8。`--workers 1` で直列)
- `src/agent.py`: 離散列 32 ビン / hidden 128 の MLP (旧 20/64 の npz は再学習が必要)
- 実プレイ: `python main.py` (npz があれば既定 learned。`L` で bootstrap 切替、`--policy bootstrap`)

## 予定: RL (REINFORCE)

- いまはまだ早い。BC が浅いと素の REINFORCE は壊しやすい（過去に確認済み）
- 足す条件: `match` が高め（目安 60–70%+）かつ生徒の `score` / `eval` が bootstrap に近い
- やり方: BC 完了後に短い微調整だけ（例: `--episodes 50 --lr 0.002`）。既定はオフのまま
- それまでは BC（収集量・epoch）を厚くする方が先
