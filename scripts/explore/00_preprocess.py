"""
00_preprocess.py
데이터셋을 파이프라인 표준 형식(JSONL)으로 변환합니다.
레이블 매핑은 src/qcd/loaders.py 의 각 Loader 클래스에서 관리합니다.

실행 예시:
  uv run python scripts/explore/00_preprocess.py
  uv run python scripts/explore/00_preprocess.py data.source=talksets
  uv run python scripts/explore/00_preprocess.py data.input=data/raw/other.json data.source=talksets

새 데이터셋 추가 시:
  1. src/qcd/loaders.py 에 Loader 클래스 추가
  2. LOADERS 딕셔너리에 한 줄 등록
  3. configs/config.yaml 의 data.source 값 변경
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig

from qcd.loaders import LOADERS


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    source: str = cfg.data.source
    in_path = Path(cfg.data.input)
    out_path = Path(cfg.data.output)

    # ── 로더 선택 ────────────────────────────────────────────────────────────
    if source not in LOADERS:
        print(
            f"[오류] 알 수 없는 source: '{source}'\n"
            f"  사용 가능: {list(LOADERS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not in_path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {in_path}", file=sys.stderr)
        sys.exit(1)

    loader = LOADERS[source]()
    print(f"[로더] {loader.__class__.__name__}  ← {in_path}")
    print(f"[카테고리] {loader.CATEGORIES}")

    # ── 로드 & 변환 ──────────────────────────────────────────────────────────
    rows = loader.load(in_path)
    if not rows:
        print("[오류] 변환된 데이터가 0개입니다.", file=sys.stderr)
        sys.exit(1)

    # ── 저장 ─────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[완료] {len(rows)}개 저장 → {out_path}")
    for label, cnt in sorted(Counter(r["label"] for r in rows).items()):
        print(f"  {label}: {cnt}개")


if __name__ == "__main__":
    main()
