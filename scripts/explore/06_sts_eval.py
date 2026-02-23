"""
06_sts_eval.py
KLUE-STS 벤치마크로 임베딩 모델의 의미 유사도 성능을 평가합니다.

평가 방식:
  - KLUE-STS validation split (519쌍) 사용
  - 각 문장쌍의 임베딩 → 코사인 유사도 계산
  - 사람이 붙인 유사도 정답(0~5점)과 Spearman 상관계수 비교
  - Spearman ρ가 1에 가까울수록 "유사한 게 가깝게, 먼 게 멀게" 잘 배치된 것

실행 예시:
  uv run python scripts/explore/06_sts_eval.py
  uv run python scripts/explore/06_sts_eval.py model=bge_m3_ko
  uv run python scripts/explore/06_sts_eval.py model=bge_m3_korean

모든 모델 순번별 비교:
  python scripts/explore/06_sts_eval.py  (model=bge_m3)
  python scripts/explore/06_sts_eval.py model=bge_m3_ko
  python scripts/explore/06_sts_eval.py model=bge_m3_korean
"""

from __future__ import annotations

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """쌍별 코사인 유사도. a, b: (N, D)"""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a * b).sum(axis=1)  # (N,)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from datasets import load_dataset
    from scipy.stats import pearsonr, spearmanr

    model_name: str = cfg.model.get("name", "model")

    # ── KLUE-STS 로드 ──────────────────────────────────────────────────────────
    print("[데이터] KLUE-STS validation 로드 중..")
    ds = load_dataset("klue", "sts", split="validation")
    print(f"  {len(ds)}개 문장쌍 로드 완료")

    # 문장쌍 & 정답 유사도 추출
    sents_a: list[str] = [ex["sentence1"] for ex in ds]
    sents_b: list[str] = [ex["sentence2"] for ex in ds]
    # 정답 레이블: real_label (0~5점 실수)
    gold_scores = np.array([ex["labels"]["real-label"] for ex in ds], dtype=np.float32)

    # ── 모델 로드 & 인코딩 ─────────────────────────────────────────────────────
    print(f"\n[모델] {model_name} 로드 중..")
    encoder = instantiate(cfg.model)

    print(f"[인코딩] sentence1 ({len(sents_a)}개)...")
    vecs_a = encoder.encode(sents_a)  # (N, D)
    print(f"[인코딩] sentence2 ({len(sents_b)}개)...")
    vecs_b = encoder.encode(sents_b)  # (N, D)

    # ── 코사인 유사도 & 상관계수 ────────────────────────────────────────────────
    pred_sims = cosine_sim(vecs_a, vecs_b)  # (N,)

    spearman_r, spearman_p = spearmanr(gold_scores, pred_sims)
    pearson_r,  pearson_p  = pearsonr(gold_scores,  pred_sims)

    # ── 구간별 분석 (낮음/중간/높음 유사도 구간에서 얼마나 잘하는지) ──────────────
    bins = [(0, 1, "낮음 (0~1)"), (1, 3, "중간 (1~3)"), (3, 5, "높음 (3~5)")]

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  KLUE-STS 평가 결과  |  모델: {model_name}")
    print(sep)
    print(f"  Spearman ρ   : {spearman_r:.4f}  (p={spearman_p:.2e})")
    print(f"  Pearson  r   : {pearson_r:.4f}  (p={pearson_p:.2e})")
    print(f"\n  ▶ 해석:")
    print(f"    0.85↑: 최고 수준 (SOTA 한국어 모델)")
    print(f"    0.80↑: 우수")
    print(f"    0.70↑: 양호")
    print(f"    0.60↓: 개선 필요")
    print(sep)

    print(f"\n  유사도 구간별 분석 (구간 내 Spearman ρ):")
    for lo, hi, label in bins:
        mask = (gold_scores >= lo) & (gold_scores < hi)
        n = int(mask.sum())
        if n >= 5:
            try:
                r, _ = spearmanr(gold_scores[mask], pred_sims[mask])
                print(f"    {label:12s}: ρ={r:.3f}  (n={n})")
            except Exception:
                print(f"    {label:12s}: 계산 실패 (n={n})")
        else:
            print(f"    {label:12s}: 샘플 부족 (n={n})")

    # ── 샘플 출력: 잘 맞춘 케이스 & 틀린 케이스 ────────────────────────────────
    errors = np.abs(pred_sims - gold_scores / 5.0)  # 정규화 후 절댓값 오차

    print(f"\n  [모델이 잘 맞춘 쌍] (오차 최소 top-5)")
    best_idx = np.argsort(errors)[:5]
    for idx in best_idx:
        print(f"    정답={gold_scores[idx]:.1f}  예측sim={pred_sims[idx]:.3f}")
        print(f"    A: {sents_a[idx][:45]}")
        print(f"    B: {sents_b[idx][:45]}")
        print()

    print(f"  [모델이 틀린 쌍] (오차 최대 top-5)")
    worst_idx = np.argsort(errors)[-5:][::-1]
    for idx in worst_idx:
        print(f"    정답={gold_scores[idx]:.1f}  예측sim={pred_sims[idx]:.3f}")
        print(f"    A: {sents_a[idx][:45]}")
        print(f"    B: {sents_b[idx][:45]}")
        print()


if __name__ == "__main__":
    main()
