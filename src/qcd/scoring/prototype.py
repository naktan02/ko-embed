"""프로토타입 기반 카테고리 점수 계산."""

from __future__ import annotations

import numpy as np


def compute_prototypes(
    embeddings: np.ndarray,
    labels: list[str],
    categories: list[str],
) -> dict[str, np.ndarray]:
    """카테고리별 평균 벡터(프로토타입)를 계산.

    Args:
        embeddings: (N, D) 임베딩 행렬
        labels:     각 임베딩의 카테고리 레이블 (길이 N)
        categories: 사용할 카테고리 목록

    Returns:
        {category: prototype_vector} 딕셔너리
    """
    prototypes: dict[str, np.ndarray] = {}
    for cat in categories:
        mask = np.array([l == cat for l in labels])
        if mask.sum() == 0:
            raise ValueError(f"카테고리 '{cat}'에 해당하는 샘플이 없습니다.")
        proto = embeddings[mask].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-9)
        prototypes[cat] = proto
    return prototypes


def cosine_scores(
    embeddings: np.ndarray,
    prototypes: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """각 임베딩과 프로토타입 간 코사인 유사도를 계산.

    Returns:
        {category: (N,) 유사도 배열}
    """
    scores: dict[str, np.ndarray] = {}
    for cat, proto in prototypes.items():
        scores[cat] = embeddings @ proto  # L2-정규화되어 있으므로 내적 = 코사인
    return scores
