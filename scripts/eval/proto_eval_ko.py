"""
scripts/eval/proto_eval_ko.py
==============================================
한국어 쿼리 → NLLB-200 번역 → 임베딩 → 프로토타입 분류 평가.

6가지 시나리오를 한 번에 실행한다:
  [1] mxbai pretrained       + ourafla
  [2] mxbai finetuned_ourafla + ourafla
  [3] mxbai pretrained       + combined
  [4] mxbai finetuned_combined + combined
  [5] mpnet pretrained       + ourafla
  [6] mpnet pretrained       + combined

번역은 한 번만 수행하고 시나리오마다 재사용한다.

실행:
  uv run python scripts/eval/proto_eval_ko.py
  uv run python scripts/eval/proto_eval_ko.py --show-translation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcd.embedding.sbert import SentenceTransformerEncoder
from qcd.translation.nllb import NLLBTranslator

# ── 경로 ────────────────────────────────────────────────────────────────────
KO_QUERIES_JSONL = ROOT / "data" / "processed" / "ko_queries.jsonl"
OURAFLA_JSONL    = ROOT / "data" / "finetune_processed" / "en_ourafla.jsonl"
COMBINED_JSONL   = ROOT / "data" / "finetune_processed" / "en_combined.jsonl"
FINETUNED_BASE   = ROOT / "models" / "finetuned"
MXBAI_ID         = "mixedbread-ai/mxbai-embed-large-v1"
MPNET_ID         = "sentence-transformers/all-mpnet-base-v2"

CATS_ORDER = ["Anxiety", "Depression", "Suicidal", "Normal"]
PRETRAINED_IDS = (MXBAI_ID, MPNET_ID)

# ── 6가지 시나리오 정의 ──────────────────────────────────────────────────────
ALL_SCENARIOS = [
    ("[1] mxbai pretrained + ourafla",          MXBAI_ID,                                                    OURAFLA_JSONL),
    ("[2] mxbai finetuned(ourafla) + ourafla",  str(FINETUNED_BASE / "mxbai_large_ourafla"  / "final"),      OURAFLA_JSONL),
    ("[3] mxbai pretrained + combined",         MXBAI_ID,                                                    COMBINED_JSONL),
    ("[4] mxbai finetuned(combined) + combined",str(FINETUNED_BASE / "mxbai_large_combined" / "final"),      COMBINED_JSONL),
    ("[5] mpnet pretrained + ourafla",           MPNET_ID,                                                    OURAFLA_JSONL),
    ("[6] mpnet pretrained + combined",          MPNET_ID,                                                    COMBINED_JSONL),
]


# ── 헬퍼 ────────────────────────────────────────────────────────────────────

def load_ko_queries(path: Path) -> list[tuple[str, str]]:
    """ko_queries.jsonl에서 (label, text) 로드."""
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries.append((item["label"], item["text"]))
    return queries


def build_prototypes(
    jsonl_path: Path,
    encoder,
    max_per_cat: int | None = None,
) -> dict[str, np.ndarray]:
    """jsonl에서 카테고리별 평균 임베딩(프로토타입) 계산."""
    import random

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
) -> str:
    scores = {cat: float(np.dot(query_vec, pv)) for cat, pv in proto_vecs.items()}
    return max(scores, key=scores.__getitem__)


# ── 단일 시나리오 평가 ────────────────────────────────────────────────────────

def run_scenario(
    scenario_name: str,
    model_id: str,
    jsonl_path: Path,
    en_texts: list[str],
    ko_texts: list[str],
    labels: list[str],
) -> float:
    print(f"\n{'='*65}")
    print(f"  시나리오: {scenario_name}")
    print(f"  모델    : {model_id}")
    print(f"  프로토  : {jsonl_path.name}")
    print(f"{'='*65}")

    encoder = SentenceTransformerEncoder(
        model_id=model_id,
        batch_size=16,
        normalize_embeddings=True,
    )

    print("\n[프로토타입 빌드]")
    proto_vecs = build_prototypes(jsonl_path, encoder)

    print("\n  임베딩 계산 중...")
    all_vecs = encoder.encode(en_texts)

    cat_correct: dict[str, int] = defaultdict(int)
    cat_total:   dict[str, int] = defaultdict(int)
    cat_pred_as: dict[str, dict[str, int]] = {c: defaultdict(int) for c in CATS_ORDER}
    errors: list[tuple[str, str, str, str]] = []

    for vec, true_label, ko, en in zip(all_vecs, labels, ko_texts, en_texts):
        pred = classify(vec, proto_vecs)
        cat_total[true_label] += 1
        cat_pred_as[true_label][pred] += 1
        if pred == true_label:
            cat_correct[true_label] += 1
        else:
            errors.append((true_label, pred, ko, en))

    total   = len(labels)
    correct = sum(cat_correct.values())
    acc     = correct / total * 100

    print(f"\n  {'':>12}  {'n':>6}  {'correct':>8}  {'acc':>7}  confusion")
    print(f"  {'-'*70}")
    for cat in CATS_ORDER:
        n      = cat_total.get(cat, 0)
        ok     = cat_correct.get(cat, 0)
        cat_acc = ok / n * 100 if n else 0
        conf = "  ".join(
            f"{c}:{cat_pred_as[cat].get(c,0)}"
            for c in CATS_ORDER if c != cat and cat_pred_as[cat].get(c, 0) > 0
        )
        print(f"  {cat:>12}  {n:>6}  {ok:>8}  {cat_acc:>6.1f}%  {conf}")
    print(f"  {'-'*70}")
    print(f"  {'TOTAL':>12}  {total:>6}  {correct:>8}  {acc:>6.1f}%")

    if errors:
        print(f"\n  [오답 샘플 — 카테고리별 최대 2개]")
        error_buckets: dict[tuple, list] = defaultdict(list)
        for true_l, pred_l, ko, en in errors:
            error_buckets[(true_l, pred_l)].append((ko, en))
        for (true_l, pred_l), items in sorted(error_buckets.items()):
            print(f"    {true_l} -> {pred_l} ({len(items)}개):")
            for ko, en in items[:2]:
                print(f"      KO: '{ko[:70]}'")
                print(f"      EN: '{en[:70]}'")

    return acc


# ── 진입점 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-translation", action="store_true")
    args = parser.parse_args()

    # 1. 번역 (한 번만)
    translator = NLLBTranslator()
    queries  = load_ko_queries(KO_QUERIES_JSONL)
    labels   = [l for l, _ in queries]
    ko_texts = [t for _, t in queries]

    print(f"\n[번역] 한국어 → 영어 ({len(ko_texts)}개)...")
    t0 = time.time()
    en_texts = translator.translate_batch(ko_texts)
    print(f"[번역 완료] {time.time()-t0:.1f}s")

    if args.show_translation:
        print("\n[번역 샘플 — 카테고리별 2개]")
        shown: dict[str, int] = defaultdict(int)
        for label, ko, en in zip(labels, ko_texts, en_texts):
            if shown[label] < 2:
                print(f"  [{label}] KO: {ko[:65]}")
                print(f"           EN: {en[:65]}")
                shown[label] += 1

    # 2. 6가지 시나리오 평가
    results: list[tuple[str, float]] = []
    for name, model_id, jsonl_path in ALL_SCENARIOS:
        model_path = Path(model_id)
        if not (model_path.exists() or model_id in PRETRAINED_IDS):
            print(f"\n[스킵] {name} — 모델 없음: {model_id}")
            continue
        acc = run_scenario(name, model_id, jsonl_path, en_texts, ko_texts, labels)
        results.append((name, acc))

    # 3. 종합 요약
    print(f"\n{'='*65}")
    print(f"  한국어 쿼리 평가 요약 (KO → NLLB → 임베딩 → 분류)")
    print(f"{'='*65}")
    en_baselines = {"[1]": 71.2, "[2]": 84.5, "[3]": 70.6, "[4]": 84.4, "[5]": 70.6, "[6]": 70.3}
    print(f"  {'시나리오':<43}  {'KO acc':>7}  {'EN acc(기준)':>12}  {'차이':>7}")
    print(f"  {'-'*75}")
    for name, acc in results:
        key = name[:3]
        en_acc = en_baselines.get(key, 0)
        diff = acc - en_acc
        sign = "+" if diff >= 0 else ""
        print(f"  {name:<43}  {acc:>6.1f}%  {en_acc:>11.1f}%  {sign}{diff:>5.1f}%p")
