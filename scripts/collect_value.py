"""価値関数の学習データを集める。教師 (`choose_x`) の軌道 + 実リターン。

`train_sim.py bc` が集めるのは「観測 -> 教師の行動」で、これは match 30% で
頭打ちになると測れている (NOTES「BC が match 60-70% に届かない」)。こちらは
「**落下後の盤** -> **その先で実際に取った点**」を集める。教師の eval を
回帰しないのは、同点帯の中が本当に無差別だと n=133 で確かめてあるためで
(NOTES「決着: 同点帯は本当に無差別」)、eval を近似しても同じ天井に当たる。

候補表も `--candidate-stride` 手おきに残す。これは順位の検証用 (学習した V が
教師の同点帯に順序を入れられているか) で、学習のラベルではない。

`--workers` は既定で論理コア/2。上げると同じ機での本家プレイがカクつく
(→AGENTS.md「CPU を埋める実行をするとき」)。

用法:
  python scripts/collect_value.py --episodes 4 --max-steps 60
  python scripts/collect_value.py --episodes 100 --out artifacts/value_100.npz
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import ROOT

from src.training.collect import (
    CANDIDATE_STRIDE,
    collect_value_episodes,
    save_value_dataset,
)
from src.training.features import FEATURE_NAMES
from src.util.parallel import default_workers

DEFAULT_OUT = ROOT / "artifacts" / "value_dataset.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    # 他のスクリプトと揃えて 400。300 だと 5〜10% が打ち切られ、リターンが
    # 欠けた値になる (NOTES「測定のしかた」の打ち切りの項)。
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0, help="並列数 (0=自動)")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--candidate-stride",
        type=int,
        default=CANDIDATE_STRIDE,
        help=f"候補表を残す間隔 (既定 {CANDIDATE_STRIDE}、0 で残さない)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    workers = args.workers if args.workers > 0 else default_workers()
    print(
        f"=== collect value ({args.episodes} ep, max_steps={args.max_steps}, "
        f"workers={workers}, stride={args.candidate_stride}) ===",
        flush=True,
    )
    started = time.time()
    data = collect_value_episodes(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        workers=workers,
        log_every=args.log_every,
        candidate_stride=args.candidate_stride,
    )
    elapsed = time.time() - started

    n = len(data.feats)
    if n == 0:
        raise SystemExit("1 手も集まらなかった")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_value_dataset(data, args.out)

    # エピソードごとの要約。打ち切りは学習側で落とす判断に要る。
    ep_ids = sorted(set(data.episodes.tolist()))
    totals = [float(data.rewards[data.episodes == e].sum()) for e in ep_ids]
    lengths = [int((data.episodes == e).sum()) for e in ep_ids]
    cut = sum(1 for e in ep_ids if bool(data.truncated[data.episodes == e][0]))
    corner = sum(1 for e in ep_ids if bool(data.cornered[data.episodes == e][0]))

    print(f"\n保存: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"手 {n}   候補行 {len(data.cand_row)}   {elapsed / 60:.1f} 分")
    print(
        f"score 平均 {statistics.fmean(totals):.1f}   "
        f"手数 平均 {statistics.fmean(lengths):.1f}"
    )
    print(f"打ち切り {cut}/{len(ep_ids)}   角スイカ {corner}/{len(ep_ids)}")
    print(f"リターン 平均 {data.returns.mean():.1f}  最大 {data.returns.max():.1f}")
    print(f"\n特徴 {len(FEATURE_NAMES)} 本  (平均 / 標準偏差)")
    for i, name in enumerate(FEATURE_NAMES):
        col = data.feats[:, i]
        print(f"  {name:<18}{col.mean():9.3f} {col.std():9.3f}")


if __name__ == "__main__":
    main()
