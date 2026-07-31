# 既知の問題・あとで直すこと

## 後回し

- **`10.png` ビジョン**: 同系色マスク融合＋枠外イチゴ。切り出し再設計が必要で、方策／held よりコスパが悪い
- **`FOREIGN_CENTER_PENALTY`**: 異種真上は物理で転がすようにした。減点は残してあるが、もう必要か・いつ消すかは要検討
- **`GAP_JUNK_PENALTY`**: 隙間埋めは事故にも連鎖の種にもなる。手で禁じずスコア結果だけで見た方が強い。後で消す
- **学習エピソード長**: `max_steps` を増やし、`episodes` を減らす（長い対局を少なめに）

## 方策 (bootstrap)

- `src/policy.py` は RL 前の薄い方策。合成・危険高さ・埋め込み・薄い大小順・転がり／弾かれ／隙間ゴミの事故防止だけ
- 持たないもの: 押し込み合成、復元押し、育成優先、連鎖隙間空け、一段上への強制寄り
- 具体手順の UT は増やさない。壊れたら事故防止か観測側を見る

## 学習まわり

- `src/reward.py`: 本家と同じ合成点のみ (cherry→0 … watermelon 55、ダブル消去 65)。生存加点・死亡減点なし。エピソード終了は従来どおり (負けライン / ダブル消去)
- `src/encode.py`: 固定長観測ベクトル
- `src/sim_env.py`: 画面なし落下 sim (`policy.simulate_drop`)
- 評価: `python scripts/eval_policy.py` (`--policy bootstrap|learned`。`--workers` 既定=論理コア/2)
- 学習: `python scripts/train_sim.py` (収集 → オフライン BC。既定 max-steps=100 は打ち切りであり負けラインではない)
- 教師収集は `ProcessPool` 並列 (既定 workers=論理コア/2。9700X なら 8。`--workers 1` で直列)
- `src/agent.py`: 離散列 32 ビン / hidden 128 の MLP (旧 20/64 の npz は再学習が必要)
- 実プレイ: `python main.py` (npz があれば既定 learned。`L` で bootstrap 切替、`--policy bootstrap`)

## 予定: RL (REINFORCE)

- いまはまだ早い。BC が浅いと素の REINFORCE は壊しやすい（過去に確認済み）
- 足す条件: `match` が高め（目安 60–70%+）かつ `student_r` が bootstrap（~70）に近い
- やり方: BC 完了後に短い微調整だけ（例: `--episodes 50 --lr 0.002`）。既定はオフのまま
- それまでは BC（収集量・epoch）を厚くする方が先
