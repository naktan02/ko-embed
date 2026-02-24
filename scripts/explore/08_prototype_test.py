"""
08_prototype_test.py
====================
3개 카테고리(distress / anger / neutral) 프로토타입 벡터 빌드 후
cosine 유사도 매칭 품질 확인.

프로토타입 소스 (우선 순위):
  1. data/finetune_processed/ko_labeled.jsonl  — 감성대화 데이터 기반 (평균 벡터)
  2. PROTOTYPES dict                           — 수동 작성 문장 (폴백)

실행:
    uv run python scripts/explore/08_prototype_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# src 패키지 경로 추가 (uv 환경에서 editable install 없이 실행 시 대비)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcd.embedding.bge_m3 import BGEM3Encoder
from qcd.embedding.sbert import SentenceTransformerEncoder


# ---------------------------------------------------------------------------
# 테스트 쿼리 — 실제 청소년 검색어 스타일 (2분류: emotional / neutral)
# ---------------------------------------------------------------------------
TEST_QUERIES: list[tuple[str, str]] = [
    # ── emotional (distress + anger 통합) ─────────────────────────────────
    ("emotional", "죽고싶다"),
    ("emotional", "우울증 테스트"),
    ("emotional", "자해 안하는 법"),
    ("emotional", "무기력증 극복"),
    ("emotional", "불면증 원인"),
    ("emotional", "공황발작 왔을때"),
    ("emotional", "욱하는 거 고치는 법"),
    ("emotional", "분노조절장애 자가진단"),
    ("emotional", "짜증날때 해소법"),
    ("emotional", "때리고싶다 어떡해"),
    ("emotional", "화병 증상"),
    # ── neutral ───────────────────────────────────────────────────────────
    ("neutral", "수능 d-100 공부법"),
    ("neutral", "배달의민족 쿠폰"),
    ("neutral", "오늘 날씨 서울"),
    ("neutral", "아이폰 배터리 교체 비용"),
    ("neutral", "독서실 이용요금"),
]


# ---------------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------------

KO_LABELED_JSONL = ROOT / "data" / "finetune_processed" / "ko_labeled.jsonl"


def build_prototypes_from_data(
    jsonl_path: Path,
    encoder,
    max_per_cat: int | None = 5000,
) -> dict[str, np.ndarray]:
    """
    ko_labeled.jsonl에서 카테고리별 중심 벡터 계산.
    감성대화 데이터 N개의 임베딩 평균으로 프로토타입 벡터를 정의한다.
    (Prototypical Networks, Snell et al. 2017 방식)
    """
    from collections import defaultdict
    import random

    buckets: dict[str, list[str]] = defaultdict(list)
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            buckets[item["label"]].append(item["text"])

    result: dict[str, np.ndarray] = {}
    for cat, texts in sorted(buckets.items()):
        if max_per_cat and len(texts) > max_per_cat:
            texts = random.sample(texts, max_per_cat)
        print(f"  [{cat}] {len(texts):,}개 인코딩 중...")
        vecs = encoder.encode(texts)
        mean_vec = vecs.mean(axis=0)
        mean_vec /= np.linalg.norm(mean_vec) + 1e-9
        result[cat] = mean_vec
    return result


def classify(
    query_vec: np.ndarray,
    proto_vecs: dict[str, np.ndarray],
) -> tuple[str, dict[str, float]]:
    """cosine 유사도 기준 최근접 카테고리 반환."""
    scores = {cat: float(np.dot(query_vec, pv)) for cat, pv in proto_vecs.items()}
    predicted = max(scores, key=scores.__getitem__)
    return predicted, scores


def run_test(model_name: str, encoder) -> None:
    print(f"\n{'='*60}")
    print(f"  모델: {model_name}")
    print(f"{'='*60}")

    print(f"\n[프로토타입 빌드 중] 소스: {KO_LABELED_JSONL}")
    proto_vecs = build_prototypes_from_data(KO_LABELED_JSONL, encoder)

    # 테스트 쿼리 평가
    correct = 0
    print(f"\n{'쿼리':<30}  {'정답':<10}  {'예측':<10}  {'distress':>8}  {'anger':>6}  {'neutral':>7}")
    cats = list(proto_vecs.keys())
    header = f"{'쿼리':<30}  {'정답':<10}  {'예측':<10}" + "".join(f"  {c:>10}" for c in cats)
    print(header)
    print("-" * (80 + 12 * len(cats)))
    for true_label, query in TEST_QUERIES:
        q_vec = encoder.encode([query])[0]
        pred, scores = classify(q_vec, proto_vecs)
        ok = "✓" if pred == true_label else "✗"
        if pred == true_label:
            correct += 1
        score_cols = "".join(f"  {scores[c]:>10.3f}" for c in cats)
        print(f"{query:<30}  {true_label:<10}  {pred:<10}{score_cols}  {ok}")

    acc = correct / len(TEST_QUERIES) * 100
    print(f"\n  정확도: {correct}/{len(TEST_QUERIES)} = {acc:.1f}%")

    # 카테고리 간 프로토타입 유사도 행렬 (분리 품질 확인)
    print("\n[카테고리 간 프로토타입 유사도 행렬]")
    cats = list(proto_vecs.keys())
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in cats)
    print(header)
    for c1 in cats:
        row = f"{c1:>10}"
        for c2 in cats:
            sim = float(np.dot(proto_vecs[c1], proto_vecs[c2]))
            row += f"{sim:>10.3f}"
        print(row)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
FINETUNED_BASE = Path("models/finetuned")

if __name__ == "__main__":
    models = [
        # ── 사전학습 모델 (베이스라인) ──
        (
            "[pretrained] dragonkue/BGE-m3-ko",
            BGEM3Encoder(model_id="dragonkue/BGE-m3-ko", batch_size=16),
        ),
        (
            "[pretrained] nlpai-lab/KURE-v1",
            SentenceTransformerEncoder(model_id="nlpai-lab/KURE-v1", batch_size=32),
        ),
    ]

    # ── 파인튜닝 모델 (저장된 경우에만 추가) ──
    for key, label in [("bge_m3_ko", "BGE-m3-ko"), ("kure_v1", "KURE-v1")]:
        model_path = FINETUNED_BASE / key
        if model_path.exists():
            models.append((
                f"[finetuned] {label}",
                SentenceTransformerEncoder(model_id=str(model_path), batch_size=32),
            ))

    for model_name, encoder in models:
        run_test(model_name, encoder)
