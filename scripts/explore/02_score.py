"""
02_score.py
임베딩에서 카테고리별 프로토타입을 계산하고 코사인 점수를 저장합니다.

실행 예시:
  uv run python scripts/explore/02_score.py
  uv run python scripts/explore/02_score.py model=bge_m3_ko
  uv run python scripts/explore/02_score.py model=kure_v1
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from qcd.scoring.prototype import compute_prototypes, cosine_scores


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    embed_dir = Path(cfg.data.artifacts_dir) / "embeddings" / cfg.model.name
    score_dir = Path(cfg.data.artifacts_dir) / "scores" / cfg.model.name

    # ── 임베딩 로드 ──────────────────────────────────────────────────────────
    emb_path = embed_dir / "embeddings.npy"
    lbl_path = embed_dir / "labels.json"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"임베딩이 없습니다: {emb_path}\n"
            "먼저 scripts/explore/01_embed.py 를 실행하세요."
        )

    embeddings = np.load(emb_path)
    labels: list[str] = json.loads(lbl_path.read_text(encoding="utf-8"))
    categories = sorted(set(labels))

    print(f"[로드] embeddings {embeddings.shape}, 카테고리: {categories}")

    # ── 프로토타입 계산 ───────────────────────────────────────────────────────
    prototypes = compute_prototypes(embeddings, labels, categories)
    for cat, proto in prototypes.items():
        print(f"  프로토타입 [{cat}] norm={np.linalg.norm(proto):.4f}")

    # ── 코사인 점수 계산 ──────────────────────────────────────────────────────
    scores = cosine_scores(embeddings, prototypes)

    # ── 저장 ─────────────────────────────────────────────────────────────────
    score_dir.mkdir(parents=True, exist_ok=True)

    # 프로토타입 저장
    proto_dict = {cat: vec.tolist() for cat, vec in prototypes.items()}
    (score_dir / "prototypes.json").write_text(
        json.dumps(proto_dict, ensure_ascii=False), encoding="utf-8"
    )

    # 점수 저장 (카테고리별 .npy)
    for cat, score_arr in scores.items():
        np.save(score_dir / f"score_{cat}.npy", score_arr)

    # 요약 통계
    pred_labels = [
        max(scores, key=lambda c: scores[c][i])
        for i in range(len(labels))
    ]
    correct = sum(p == g for p, g in zip(pred_labels, labels))
    acc = correct / len(labels) * 100
    print(f"\n[결과] 프로토타입 분류 정확도: {acc:.1f}%  ({correct}/{len(labels)})")
    print(f"[완료] {score_dir}")


if __name__ == "__main__":
    main()
