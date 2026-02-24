"""
감성대화 → 2카테고리 JSONL 전처리 스크립트

출력: data/finetune_processed/ko_labeled.jsonl
형식: {"text": "...", "label": "emotional|neutral", "source": "감성대화", "ecode": "E10"}

라벨 체계:
  emotional: 분노/슬픔/우울/불안 통합 (E10~E59)
  neutral  : 긍정/중립                (E60~E69)
"""

import json
import random
from pathlib import Path

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
SRC_DIR = Path("data/finetune_raw/감성대화")
SRC_JSONS = [
    SRC_DIR / "감성대화말뭉치(최종데이터)_Training.json",
    SRC_DIR / "감성대화말뭉치(최종데이터)_Validation.json",
]
OUT_DIR = Path("data/finetune_processed")
OUT_FILE = OUT_DIR / "ko_labeled.jsonl"

# 카테고리별 최대 샘플 수 (None이면 제한 없음)
# emotional은 원본이 많으니 neutral과 균형 유지
MAX_PER_CATEGORY: dict[str, int | None] = {
    "emotional": 15_000,
    "neutral":   15_000,
}

MIN_TEXT_LEN = 10   # 너무 짧은 발화 제외 (자)
SEED = 42

# ──────────────────────────────────────────────
# E코드 → 카테고리 매핑 (2분류)
# ──────────────────────────────────────────────
# emotional: E10~E59 (분노+슬픔+우울+불안 통합)
# neutral  : E60~E69 (긍정/중립)
# 근거: distress고 anger 프로토타입 유사도 0.991로 구분 불가 확인

def _make_ecode_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    # emotional: E10~E59 전부
    for i in range(10, 60):
        mapping[f"E{i:02d}"] = "emotional"
    # neutral: E60~E69
    for i in range(60, 70):
        mapping[f"E{i:02d}"] = "neutral"
    return mapping

ECODE_MAP = _make_ecode_map()


def extract_utterances(record: dict) -> list[str]:
    """한 레코드의 사람 발화(HS01, HS02)를 추출."""
    content = record.get("talk", {}).get("content", {})
    texts = []
    for key in ("HS01", "HS02"):
        t = content.get(key, "").strip()
        if t and len(t) >= MIN_TEXT_LEN:
            texts.append(t)
    return texts


def build_samples(srcs: list[Path]) -> dict[str, list[dict]]:
    """여러 JSON 파일을 읽어 카테고리별 샘플 리스트 반환."""
    buckets: dict[str, list[dict]] = {"emotional": [], "neutral": []}

    for src in srcs:
        print(f"[로딩] {src.name}")
        with open(src, encoding="utf-8") as f:
            data = json.load(f)

        skip_count = 0
        for record in data:
            ecode = record.get("profile", {}).get("emotion", {}).get("type", "")
            category = ECODE_MAP.get(ecode)
            if category is None:
                skip_count += 1
                continue

            for text in extract_utterances(record):
                buckets[category].append(
                    {"text": text, "label": category, "source": "감성대화", "ecode": ecode}
                )

        print(f"  → 총 레코드: {len(data):,}  스킵(E코드 미매핑): {skip_count:,}")

    return buckets


def balance_and_shuffle(
    buckets: dict[str, list[dict]],
    max_per: dict[str, int | None],
    seed: int,
) -> list[dict]:
    """카테고리별 상한 적용 후 전체 셔플."""
    rng = random.Random(seed)
    result = []
    for cat, samples in buckets.items():
        rng.shuffle(samples)
        cap = max_per.get(cat)
        chosen = samples[:cap] if cap is not None else samples
        result.extend(chosen)
        print(f"  {cat:>8}: 원본 {len(samples):>6,}개 → 사용 {len(chosen):>6,}개")
    rng.shuffle(result)
    return result


def save_jsonl(samples: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n저장 완료: {out}  ({len(samples):,}라인)")


def main() -> None:
    buckets = build_samples(SRC_JSONS)

    print("\n[카테고리별 샘플 수 조정]")
    samples = balance_and_shuffle(buckets, MAX_PER_CATEGORY, SEED)

    save_jsonl(samples, OUT_FILE)

    # 간단 검증: 앞 5개 출력
    print("\n[샘플 미리보기 (5개)]")
    for s in samples[:5]:
        print(f"  [{s['label']}] ({s['ecode']}) {s['text'][:60]}")


if __name__ == "__main__":
    main()
