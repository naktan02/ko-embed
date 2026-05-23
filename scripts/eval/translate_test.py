"""
scripts/eval/translate_test.py
==============================
Step 1: NLLB-200 번역 단독 검증.

ko_queries.jsonl에서 카테고리별 샘플을 뽑아
한국어 → 영어 번역 결과를 출력합니다.
번역 품질을 육안으로 확인하는 용도입니다.

실행:
    # jsonl 샘플 번역
    uv run python scripts/eval/translate_test.py
    uv run python scripts/eval/translate_test.py --n 5

    # 직접 입력한 텍스트 번역
    uv run python scripts/eval/translate_test.py --text "죽고 싶어"
    uv run python scripts/eval/translate_test.py --text "공황장애 증상" "살 이유 없어" "오늘 날씨"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcd.translation.nllb import NLLBTranslator

KO_QUERIES = ROOT / "data" / "processed" / "ko_queries.jsonl"
CATS_ORDER  = ["Anxiety", "Depression", "Suicidal", "Normal"]


def load_samples(path: Path, n: int) -> dict[str, list[str]]:
    """카테고리별 n개씩 로드 (앞에서 순서대로)."""
    buckets: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cat, text = item["label"], item["text"]
            if len(buckets[cat]) < n:
                buckets[cat].append(text)
    return dict(buckets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="카테고리별 출력 개수 (기본: 3)")
    parser.add_argument("--text", nargs="+", default=None, help="직접 번역할 한국어 텍스트")
    args = parser.parse_args()

    # 1. 번역기 로드
    translator = NLLBTranslator()

    # 2a. --text 직접 입력 모드
    if args.text:
        print(f"\n[번역] {len(args.text)}개 문장")
        t0 = time.time()
        results = translator.translate_batch(args.text)
        elapsed = time.time() - t0
        print(f"[완료] {elapsed:.1f}s  (평균 {elapsed / len(args.text) * 1000:.0f}ms/문장)\n")
        for ko, en in zip(args.text, results):
            print(f"  KO: {ko}")
            print(f"  EN: {en}")
            print()
        return

    # 2b. jsonl 샘플 모드
    samples = load_samples(KO_QUERIES, args.n)
    all_ko: list[str] = []
    all_cats: list[str] = []
    for cat in CATS_ORDER:
        for text in samples.get(cat, []):
            all_ko.append(text)
            all_cats.append(cat)

    print(f"\n[번역 시작] 총 {len(all_ko)}개 문장 ({args.n}개 × {len(CATS_ORDER)}카테고리)")

    # 3. 배치 번역 + 시간 측정
    t0 = time.time()
    all_en = translator.translate_batch(all_ko)
    elapsed = time.time() - t0

    print(f"[완료] {elapsed:.1f}s  (평균 {elapsed / len(all_ko) * 1000:.0f}ms/문장)\n")

    # 4. 결과 출력
    idx = 0
    for cat in CATS_ORDER:
        print(f"{'='*60}")
        print(f"  [{cat}]")
        print(f"{'='*60}")
        n_cat = len(samples.get(cat, []))
        for _ in range(n_cat):
            print(f"  KO: {all_ko[idx]}")
            print(f"  EN: {all_en[idx]}")
            print()
            idx += 1


if __name__ == "__main__":
    main()
