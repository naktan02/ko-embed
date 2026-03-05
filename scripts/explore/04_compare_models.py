"""
04_compare_models.py
여러 모델의 프로토타입 분류 정확도를 비교해 요약표를 출력합니다.
(각 모델에 대해 01_embed.py + 02_score.py 가 먼저 실행되어 있어야 합니다)

실행 예시:
  uv run python scripts/explore/04_compare_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def evaluate_model(artifacts_dir: Path, model_name: str) -> dict | None:
    """저장된 임베딩과 점수로 모델 성능 지표를 계산."""
    score_dir = artifacts_dir / "scores" / model_name
    embed_dir = artifacts_dir / "embeddings" / model_name
    lbl_path = embed_dir / "labels.json"

    if not score_dir.exists() or not lbl_path.exists():
        return None

    labels: list[str] = json.loads(lbl_path.read_text(encoding="utf-8"))
    categories = sorted(set(labels))

    # 각 카테고리 점수 로드
    score_files = list(score_dir.glob("score_*.npy"))
    if not score_files:
        return None

    scores: dict[str, np.ndarray] = {}
    for sf in score_files:
        cat = sf.stem.replace("score_", "")
        scores[cat] = np.load(sf)

    # 예측
    pred_labels = [
        max(scores, key=lambda c: scores[c][i])
        for i in range(len(labels))
    ]

    # 전체 정확도
    correct = sum(p == g for p, g in zip(pred_labels, labels))
    accuracy = correct / len(labels) * 100

    # 카테고리별 F1
    per_cat: dict[str, float] = {}
    for cat in categories:
        tp = sum(p == cat and g == cat for p, g in zip(pred_labels, labels))
        fp = sum(p == cat and g != cat for p, g in zip(pred_labels, labels))
        fn = sum(p != cat and g == cat for p, g in zip(pred_labels, labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        per_cat[cat] = f1

    macro_f1 = sum(per_cat.values()) / len(per_cat) if per_cat else 0.0

    return {
        "model": model_name,
        "n_samples": len(labels),
        "accuracy": accuracy,
        "macro_f1": macro_f1 * 100,
        "per_cat_f1": {k: v * 100 for k, v in per_cat.items()},
    }


def main() -> None:
    artifacts_dir = Path("artifacts")
    embed_root = artifacts_dir / "embeddings"

    if not embed_root.exists():
        print("[오류] artifacts/embeddings/ 폴더가 없습니다.")
        print("먼저 scripts/explore/01_embed.py → 02_score.py 를 실행하세요.")
        return

    model_names = [d.name for d in embed_root.iterdir() if d.is_dir()]
    if not model_names:
        print("[오류] 평가할 모델이 없습니다.")
        return

    print(f"[평가] 모델 {len(model_names)}개: {model_names}\n")

    results = []
    for name in sorted(model_names):
        result = evaluate_model(artifacts_dir, name)
        if result is None:
            print(f"  [{name}] 점수 파일 없음 — 02_score.py 먼저 실행하세요.")
            continue
        results.append(result)

    if not results:
        print("평가 가능한 모델이 없습니다.")
        return

    # ── 요약표 출력 ────────────────────────────────────────────────────────
    print(f"{'모델':<25} {'샘플':>7} {'정확도':>9} {'Macro F1':>10}")
    print("-" * 58)
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        print(
            f"{r['model']:<25} {r['n_samples']:>7} "
            f"{r['accuracy']:>8.1f}%  {r['macro_f1']:>8.1f}%"
        )

    # ── 카테고리별 F1 상세 ─────────────────────────────────────────────────
    print("\n[카테고리별 F1]")
    all_cats = sorted({
        cat for r in results for cat in r["per_cat_f1"]
    })
    header = f"{'모델':<25}" + "".join(f"{c:>12}" for c in all_cats)
    print(header)
    print("-" * (25 + 12 * len(all_cats)))
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        row = f"{r['model']:<25}"
        for cat in all_cats:
            val = r["per_cat_f1"].get(cat, float("nan"))
            row += f"{val:>11.1f}%"
        print(row)


if __name__ == "__main__":
    main()
