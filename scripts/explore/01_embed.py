"""
01_embed.py
JSONL 데이터를 임베딩 벡터로 변환해 artifacts/ 에 저장합니다.

실행 예시:
  uv run python scripts/explore/01_embed.py
  uv run python scripts/explore/01_embed.py model=bge_m3_ko
  uv run python scripts/explore/01_embed.py model=kure_v1
  uv run python scripts/explore/01_embed.py data.output_dir=artifacts/embeddings/test
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    processed_path = Path(cfg.data.output)  # data/processed/queries.jsonl
    artifacts_dir = Path(cfg.data.artifacts_dir) / "embeddings" / cfg.model.name

    if not processed_path.exists():
        raise FileNotFoundError(
            f"처리된 데이터가 없습니다: {processed_path}\n"
            "먼저 scripts/explore/00_preprocess.py 를 실행하세요."
        )

    # ── JSONL 로드 ──────────────────────────────────────────────────────────
    records: list[dict] = []
    with processed_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    print(f"[로드] {len(texts)}개 텍스트  ← {processed_path}")

    # ── 인코더 초기화 ────────────────────────────────────────────────────────
    print(f"[모델] {cfg.model.name}  ({cfg.model.model_id})")
    encoder = hydra.utils.instantiate(cfg.model)

    # ── 임베딩 생성 ──────────────────────────────────────────────────────────
    embeddings: np.ndarray = encoder.encode(texts)
    print(f"[임베딩] shape={embeddings.shape}, dtype={embeddings.dtype}")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / "embeddings.npy", embeddings)
    (artifacts_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False), encoding="utf-8"
    )
    (artifacts_dir / "texts.json").write_text(
        json.dumps(texts, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[완료] {artifacts_dir}")
    print(f"  embeddings.npy  {embeddings.shape}")
    print(f"  labels.json     {len(labels)}개")


if __name__ == "__main__":
    main()
