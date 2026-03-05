"""
scripts/finetune/11_build_en_dataset.py
ourafla + cssrs 원본 CSV를 읽어 파인튜닝용 jsonl로 변환.

출력:
  data/finetune_processed/en_ourafla.jsonl   — ourafla 단독
  data/finetune_processed/en_combined.jsonl  — ourafla + cssrs 합산

cssrs severity → 4개 카테고리 매핑:
  0        → Normal
  1, 2     → Depression
  3, 4, 5, 6 → Suicidal
  (Anxiety 없음 — cssrs는 자살위험도 특화)

실행:
  uv run python scripts/finetune/11_build_en_dataset.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
OURAFLA_TRAIN = Path("data/finetune_raw/ourafla/train.csv")
CSSRS_FILE    = Path("data/finetune_raw/cssrs/human_n_llm_labeled_rSuicidewatch_posts.csv")
OUT_DIR       = Path("data/finetune_processed")

# cssrs severity(0~6) → 4개 카테고리
CSSRS_MAP: dict[str, str] = {
    "0": "Normal",
    "1": "Depression",
    "2": "Depression",
    "3": "Suicidal",
    "4": "Suicidal",
    "5": "Suicidal",
    "6": "Suicidal",
}


def load_ourafla(path: Path) -> list[dict]:
    """ourafla train.csv → {text, label} 리스트."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text  = row.get("text", "").strip()
            label = row.get("status", "").strip()
            if text and label in {"Anxiety", "Depression", "Suicidal", "Normal"}:
                rows.append({"text": text, "label": label})
    return rows


def load_cssrs(path: Path) -> list[dict]:
    """cssrs CSV → {text, label} 리스트 (severity 매핑 적용)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text  = row.get("content", "").strip()
            sev   = str(row.get("severity", "")).strip()
            label = CSSRS_MAP.get(sev)
            if text and label:
                rows.append({"text": text, "label": label})
    return rows


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    dist = Counter(r["label"] for r in records)
    print(f"  → {path}  ({len(records):,}개)")
    for label, cnt in sorted(dist.items()):
        print(f"      {label:<12}: {cnt:,}")


def main() -> None:
    print("[ourafla 로드]")
    ourafla = load_ourafla(OURAFLA_TRAIN)
    print(f"  {len(ourafla):,}개")

    print("\n[cssrs 로드]")
    cssrs = load_cssrs(CSSRS_FILE)
    print(f"  {len(cssrs):,}개")

    print("\n[저장]")
    write_jsonl(ourafla,           OUT_DIR / "en_ourafla.jsonl")
    write_jsonl(ourafla + cssrs,   OUT_DIR / "en_combined.jsonl")
    print("\n완료")


if __name__ == "__main__":
    main()
