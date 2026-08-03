"""scripts が共有するリポジトリルート。

各スクリプトは、このモジュールを import できる時点で ROOT を sys.path に
入れ終えている (直接実行なら先頭の sys.path.insert、`python -m` ならその
仕組み自体が入れる)。なので path を足す関数は持たない。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
