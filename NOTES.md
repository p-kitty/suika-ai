# 既知の問題・あとで直すこと

## 後回し

- **`10.png` ビジョン**: 同系色マスク融合＋枠外イチゴ。切り出し再設計が必要で、ポリシー／held よりコスパが悪い

## 方策 (bootstrap)

- `src/policy.py` は RL 前の薄い方策。合成・危険高さ・埋め込み・薄い大小順・転がり／弾かれ／隙間ゴミの事故防止だけ
- 持たないもの: 押し込み合成、復元押し、育成優先、連鎖隙間空け、一段上への強制寄り
- 具体手順の UT は増やさない。壊れたら事故防止か観測側を見る

## 学習まわり

- `src/reward.py`: 生存・合成・最大段階・スイカ増／ダブル到達／スイカ消去・死亡 (維持加点なし)
- `src/encode.py`: 固定長観測ベクトル
- `src/sim_env.py`: 画面なし落下 sim (`policy.simulate_drop`)
- 評価: `python scripts/eval_bootstrap.py`
- 学習: `python scripts/train_sim.py` (bootstrap を先生に soft-BC+replay。最良 student_r を保存。RL 既定オフ)
- `src/agent.py`: 離散列 20 ビンの MLP 方策
