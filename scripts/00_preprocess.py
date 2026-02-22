"""
00_preprocess.py
실제 데이터셋(CSV / Excel)을 파이프라인 표준 형식(JSONL)으로 변환합니다.

실행 예시:
  uv run python scripts/00_preprocess.py \
      data.input=data/raw/aihub_558.xlsx \
      data.text_col="검색어" \
      data.label_col="분류"

오버라이드 예시 (다른 데이터):
  uv run python scripts/00_preprocess.py \
      data.input=data/raw/other.csv \
      data.text_col=text \
      data.label_col=label \
      data.source=other_dataset
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig

# 원본 레이블 → 내부 카테고리 매핑
# 데이터셋에 맞게 수정하세요.
LABEL_MAP: dict[str, str] = {
    # AIHub 558 계열 (실제 컬럼 확인 후 수정)
    "위기": "distress",
    "자해": "self_harm",
    "도움요청": "help_seeking",
    "일반": "neutral",
    # 영문 레이블이면 그대로 통과
    "distress": "distress",
    "self_harm": "self_harm",
    "help_seeking": "help_seeking",
    "neutral": "neutral",
}


def load_tabular(path: Path, text_col: str, label_col: str, source: str) -> list[dict]:
    """CSV 또는 Excel을 읽어 표준 딕셔너리 리스트로 반환."""
    import pandas as pd

    if path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    rows: list[dict] = []
    skipped = 0
    for _, row in df.iterrows():
        raw_label = str(row[label_col]).strip()
        label = LABEL_MAP.get(raw_label)
        if label is None:
            skipped += 1
            continue
        text = str(row[text_col]).strip()
        if not text:
            continue
        rows.append({"text": text, "label": label, "source": source})

    if skipped:
        print(f"  [경고] 알 수 없는 레이블로 건너뜀: {skipped}행 → LABEL_MAP 확인 필요", file=sys.stderr)

    return rows


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    in_path = Path(cfg.data.input)
    out_path = Path(cfg.data.output)
    text_col: str = cfg.data.text_col
    label_col: str = cfg.data.label_col
    source: str = cfg.data.source or in_path.stem

    if not in_path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[로드] {in_path}  (text_col={text_col!r}, label_col={label_col!r})")
    rows = load_tabular(in_path, text_col, label_col, source)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[완료] {len(rows)}개 저장 → {out_path}")
    for label, cnt in sorted(Counter(r["label"] for r in rows).items()):
        print(f"  {label}: {cnt}개")


if __name__ == "__main__":
    main()
