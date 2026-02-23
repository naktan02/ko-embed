"""
05_search.py
저장된 임베딩에서 쿼리 문장과 코사인 유사도로 가장 유사한 문장을 찾습니다.

실행 예시:
  uv run python scripts/explore/05_search.py query="죽고 싶지 않아"
  uv run python scripts/explore/05_search.py query="죽고 싶지 않아" search.topk=20
  uv run python scripts/explore/05_search.py query="죽고 싶지 않아" search.topk=10 model=sroberta

출력:
  - 코사인 유사도 상위 K개 문장 + 카테고리
  - 카테고리별 분포 요약
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cosine_similarity(query_vec: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """L2-정규화된 벡터 간의 내적 = 코사인 유사도"""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    # corpus가 이미 정규화돼 있으면 내적만으로 충분
    return corpus @ q  # (N,)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # ── 설정 ──────────────────────────────────────────────────────────────────
    search_cfg = cfg.get("search", OmegaConf.create({}))
    query: str = search_cfg.get("query", "")
    topk: int   = search_cfg.get("topk", 10)

    if not query:
        raise ValueError(
            "쿼리 문장을 입력하세요.\n"
            "  예: uv run python scripts/explore/05_search.py search.query=\"죽고 싶지 않아\""
        )

    model_name: str = cfg.model.get("name", "model")
    artifacts_dir = Path(cfg.data.artifacts_dir)
    emb_path  = artifacts_dir / "embeddings" / model_name / "embeddings.npy"
    meta_path = artifacts_dir / "embeddings" / model_name / "metadata.jsonl"

    if not emb_path.exists():
        raise FileNotFoundError(f"임베딩 없음: {emb_path}\n먼저 01_embed.py 실행")

    # ── 코퍼스 로드 ────────────────────────────────────────────────────────────
    print(f"\n[로드] {emb_path}")
    corpus = np.load(emb_path)               # (N, D) float32
    rows   = load_jsonl(meta_path)
    texts  = [r["text"]  for r in rows]
    labels = [r["label"] for r in rows]
    print(f"  코퍼스: {len(texts):,}개 문장 (dim={corpus.shape[1]})")

    # ── 쿼리 인코딩 ────────────────────────────────────────────────────────────
    print(f"\n[쿼리] \"{query}\"")
    print(f"[모델] {model_name} 로드 중..")
    encoder = instantiate(cfg.model)
    q_vec = encoder.encode([query])[0]       # (D,)

    # ── 유사도 계산 ────────────────────────────────────────────────────────────
    sims = cosine_similarity(q_vec, corpus)  # (N,)
    top_idx = np.argsort(sims)[::-1][:topk]

    # ── 결과 출력 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  상위 {topk}개 유사 문장  (모델: {model_name})")
    print(f"{'='*65}")
    print(f"{'순위':>4}  {'유사도':>6}  {'카테고리':<14}  문장")
    print(f"{'-'*65}")
    for rank, idx in enumerate(top_idx, 1):
        sim   = sims[idx]
        label = labels[idx]
        text  = texts[idx]
        text_disp = text[:50] + ("…" if len(text) > 50 else "")
        print(f"{rank:>4}  {sim:>6.4f}  {label:<14}  {text_disp}")

    # ── 카테고리 분포 ──────────────────────────────────────────────────────────
    top_labels = [labels[i] for i in top_idx]
    from collections import Counter
    dist = Counter(top_labels)

    print(f"\n{'─'*40}")
    print(f"  상위 {topk}개 카테고리 분포")
    print(f"{'─'*40}")
    for cat, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        bar = "■" * cnt
        print(f"  {cat:<14} {cnt:>3}개  {bar}")

    print(f"\n[해석]")
    top_cat = dist.most_common(1)[0][0]
    if top_cat != "neutral":
        print(f"  → 상위 이웃의 주요 카테고리: '{top_cat}'")
        print(f"     쿼리가 해당 카테고리 임베딩 공간 근처에 배치됨")
    else:
        print(f"  → 상위 이웃 대부분이 neutral")
        print(f"     쿼리가 위기/고위험 표현과 멀리 배치됨")

    unique_cats = len(dist)
    if unique_cats >= 4:
        print(f"  → {unique_cats}개 카테고리가 혼재 → 모델이 해당 표현을 모호하게 인식")
    elif unique_cats <= 2:
        print(f"  → {unique_cats}개 카테고리만 등장 → 모델이 해당 표현을 명확히 구분")


if __name__ == "__main__":
    main()
