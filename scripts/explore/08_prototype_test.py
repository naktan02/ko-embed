"""
scripts/explore/08_prototype_test.py
==============================================
4가지 시나리오로 카테고리 프로토타입 분류 성능 비교.

평가셋: ourafla test.csv (248개 × 4카테고리 = 992개)

시나리오:
  [1] pretrained mxbai  + ourafla 프로토타입
  [2] finetuned  mxbai  + ourafla 프로토타입  (models/finetuned/mxbai_large_ourafla)
  [3] pretrained mxbai  + combined 프로토타입
  [4] finetuned  mxbai  + combined 프로토타입 (models/finetuned/mxbai_large_combined)

4개 카테고리: Anxiety / Depression / Suicidal / Normal

실행:
  uv run python scripts/explore/08_prototype_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcd.embedding.sbert import SentenceTransformerEncoder

# ── 데이터 경로 ──────────────────────────────────────────────────────────────────────────────
OURAFLA_TRAIN_JSONL = ROOT / "data" / "finetune_processed" / "en_ourafla.jsonl"
COMBINED_JSONL      = ROOT / "data" / "finetune_processed" / "en_combined.jsonl"
OURAFLA_TEST_CSV    = ROOT / "data" / "finetune_raw" / "ourafla" / "test.csv"
FINETUNED_BASE      = ROOT / "models" / "finetuned"

CATS_ORDER = ["Anxiety", "Depression", "Suicidal", "Normal"]


def load_test_queries(path: Path) -> list[tuple[str, str]]:
    """ourafla test.csv 에서 (label, text) 추출."""
    import csv
    queries = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text  = row.get("text", "").strip()
            label = row.get("status", "").strip()
            if text and label:
                queries.append((label, text))
    return queries


def build_prototypes(
    jsonl_path: Path,
    encoder,
    max_per_cat: int | None = 5000,
) -> dict[str, np.ndarray]:
    """jsonl에서 카테고리별 평균 임베딩(프로토타입) 계산."""
    import random
    from collections import defaultdict

    buckets: dict[str, list[str]] = defaultdict(list)
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            buckets[item["label"]].append(item["text"])

    result: dict[str, np.ndarray] = {}
    for cat in sorted(buckets.keys()):
        texts = buckets[cat]
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
    return max(scores, key=scores.__getitem__), scores


def run_test(scenario: str, encoder, jsonl_path: Path) -> float:
    cats_order = CATS_ORDER

    print(f"\n{'='*65}")
    print(f"  시나리오: {scenario}")
    print(f"  데이터  : {jsonl_path.name}")
    print(f"{'='*65}")

    print("\n[프로토타입 빌드]")
    proto_vecs = build_prototypes(jsonl_path, encoder)

    # 평가셋 로드: ourafla test.csv (992개)
    test_queries = load_test_queries(OURAFLA_TEST_CSV)
    print(f"\n[평가셋] {OURAFLA_TEST_CSV.name}: {len(test_queries)}개")

    # 직접 배치 인코딩으로 속도 개선
    texts = [q for _, q in test_queries]
    labels = [l for l, _ in test_queries]

    print("  임베딩 계산 중...")
    all_vecs = encoder.encode(texts)  # (N, dim)

    # 카테고리별 정확도 집계
    from collections import defaultdict
    cat_correct: dict[str, int] = defaultdict(int)
    cat_total:   dict[str, int] = defaultdict(int)
    cat_pred_as: dict[str, dict[str, int]] = {c: defaultdict(int) for c in cats_order}
    errors: list[tuple[str, str, str]] = []  # (true, pred, text)

    for vec, true_label, text in zip(all_vecs, labels, texts):
        pred, _ = classify(vec, proto_vecs)
        cat_total[true_label] += 1
        cat_pred_as[true_label][pred] += 1
        if pred == true_label:
            cat_correct[true_label] += 1
        else:
            errors.append((true_label, pred, text))

    total  = len(test_queries)
    correct = sum(cat_correct.values())
    acc = correct / total * 100

    # 카테고리별 정확도 출력
    print(f"\n  {'':>12}  {'n':>6}  {'correct':>8}  {'acc':>7}  {'confusion (pred as...)':<40}")
    print(f"  {'-'*75}")
    for cat in cats_order:
        n   = cat_total.get(cat, 0)
        ok  = cat_correct.get(cat, 0)
        cat_acc = ok / n * 100 if n else 0
        conf = "  ".join(
            f"{c}:{cat_pred_as[cat].get(c,0)}"
            for c in cats_order if c != cat and cat_pred_as[cat].get(c, 0) > 0
        )
        print(f"  {cat:>12}  {n:>6}  {ok:>8}  {cat_acc:>6.1f}%  {conf}")

    print(f"  {'-'*75}")
    print(f"  {'TOTAL':>12}  {total:>6}  {correct:>8}  {acc:>6.1f}%")

    # 오답 샘플 출력 (카테고리별 최대 3개)
    if errors:
        print(f"\n  [오답 샘플 — 카테고리별 최대 3개]")
        from collections import defaultdict
        error_buckets: dict[tuple, list] = defaultdict(list)
        for true_l, pred_l, text in errors:
            error_buckets[(true_l, pred_l)].append(text)
        for (true_l, pred_l), txts in sorted(error_buckets.items()):
            print(f"    {true_l} -> {pred_l} ({len(txts)}개):")
            for t in txts[:3]:
                print(f"      '{t[:80]}'")

    # 카테고리 간 프로토타입 유사도 행렬
    print("\n[카테고리 간 프로토타입 유사도 행렬]  (대각=1.0, 0에 가까울수록 잘 분리)")
    cats = [c for c in cats_order if c in proto_vecs]
    print(f"{'':>12}" + "".join(f"{c:>12}" for c in cats))
    for c1 in cats:
        row = f"{c1:>12}"
        for c2 in cats:
            sim = float(np.dot(proto_vecs[c1], proto_vecs[c2]))
            row += f"{sim:>12.3f}"
        print(row)

    return acc


if __name__ == "__main__":
    MXBAI_ID = "mixedbread-ai/mxbai-embed-large-v1"

    scenarios: list[tuple[str, str, Path]] = [
        ("[1] pretrained + ourafla",         MXBAI_ID,                                       OURAFLA_TRAIN_JSONL),
        ("[2] finetuned(ourafla) + ourafla",  str(FINETUNED_BASE / "mxbai_large_ourafla"),    OURAFLA_TRAIN_JSONL),
        ("[3] pretrained + combined",          MXBAI_ID,                                       COMBINED_JSONL),
        ("[4] finetuned(combined) + combined", str(FINETUNED_BASE / "mxbai_large_combined"),   COMBINED_JSONL),
    ]

    results: list[tuple[str, float]] = []
    for scenario_name, model_path, jsonl_path in scenarios:
        if not Path(model_path).exists() and model_path != MXBAI_ID:
            print(f"\n[스킵] {scenario_name}  — 모델 없음: {model_path}")
            continue
        if not jsonl_path.exists():
            print(f"\n[스킵] {scenario_name}  — 데이터 없음: {jsonl_path}")
            continue

        encoder = SentenceTransformerEncoder(
            model_id=model_path,
            batch_size=16,
            normalize_embeddings=True,
        )
        acc = run_test(scenario_name, encoder, jsonl_path)
        results.append((scenario_name, acc))

    if len(results) > 1:
        print(f"\n{'='*50}")
        print("  시나리오별 정확도 요약")
        print(f"{'='*50}")
        for name, acc in results:
            print(f"  {name:<40}  {acc:>6.1f}%")
