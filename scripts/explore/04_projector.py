"""
04_projector.py
임베딩을 TensorFlow Embedding Projector용 TSV로 변환합니다.
UMAP 없이, 원본 임베딩 고차원 그대로 카테고리 균등 샘플링하여 저장합니다.

실행 예시:
  uv run python scripts/explore/04_projector.py
  uv run python scripts/explore/04_projector.py model=sroberta
  uv run python scripts/explore/04_projector.py umap.max_samples=5000

출력:
  artifacts/projector/{model_name}/tensor.tsv   — 벡터 (헤더 없음)
  artifacts/projector/{model_name}/metadata.tsv — 메타데이터(첫 번째 컬럼명)
  → projector.tensorflow.org 에서 두 파일 업로드
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from qcd.loaders import LOADERS


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _clean(value: str) -> str:
    """TSV 저장 전에 안전하게 이어가도록 문자로 변환.
    \\t \\r \\n 을 공백으로 대체하여 TSV 파싱 오류 방지.
    """
    return value.replace("\r", " ").replace("\n", " ").replace("\t", " ")


def save_projector_tsv(
    embeddings: np.ndarray,
    rows: list[dict],
    out_dir: Path,
    label_only: bool = False,
) -> None:
    """Embedding Projector (projector.tensorflow.org)용 TSV 파일 생성.

    - tensor.tsv : 헤더 없음, 탭으로 구분된 벡터 (행 수 = 샘플 수)
    - metadata.tsv:
        * label_only=False (기본): 첫 번째 컬럼 헤더 있음, 이후 데이터
          — 컬럼이 2개 이상이면 Projector가 헤더를 자동 인식
          — 행 수 = tensor 행 수 + 1 (헤더)
        * label_only=True: 헤더 없이 레이블 1열만 저장
          — 행 수 = tensor 행 수와 동일해야 함

    주의: newline='\\n' 으로 고정하여 Windows CRLF 자동변환 방지
    """
    assert len(embeddings) == len(rows), (
        f"TSV 불일치: tensor={len(embeddings)}, metadata={len(rows)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── tensor.tsv ────────────────────────────────────────────────────────────
    tensor_path = out_dir / "tensor.tsv"
    with tensor_path.open("w", encoding="utf-8", newline="\n") as f:
        for vec in embeddings:
            f.write("\t".join(f"{v:.6f}" for v in vec) + "\n")

    # ── metadata.tsv ──────────────────────────────────────────────────────────
    meta_path = out_dir / "metadata.tsv"

    if label_only:
        # 순수 레이블 1열, 헤더 없음 (tensor 행 수 == metadata 행 수)
        with meta_path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(_clean(str(row.get("label", ""))) + "\n")
        meta_mode = "label-only (헤더 없음)"
    else:
        # 멀티컬럼 + 헤더 (tensor 행 수 == metadata 행 수 - 1)
        skip_keys = {"text"}
        if rows:
            scalar_keys = [
                k for k in rows[0].keys()
                if k not in skip_keys and not isinstance(rows[0][k], (list, dict))
            ]
            headers = scalar_keys + ["text"]
        else:
            headers = ["text"]

        with meta_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(headers) + "\n")
            for row in rows:
                values = []
                for h in headers:
                    raw = row.get(h, "")
                    if isinstance(raw, list):
                        v = _clean(", ".join(str(x) for x in raw))
                    else:
                        v = _clean(str(raw))
                    values.append(v)
                f.write("\t".join(values) + "\n")
        meta_mode = f"멀티컬럼 {len(headers)}열 + 헤더"

    print(f"[저장] Embedding Projector TSV → {out_dir}/")
    print(f"  tensor.tsv   : {len(embeddings):,}행")
    print(f"  metadata.tsv : {len(rows):,}행  [{meta_mode}]")
    print(f"  → projector.tensorflow.org 에서 tensor.tsv + metadata.tsv 업로드")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_name: str = cfg.model.get("name", "model")
    artifacts_dir = Path(cfg.data.artifacts_dir)
    emb_dir = artifacts_dir / "embeddings" / model_name
    emb_path = emb_dir / "embeddings.npy"
    meta_path = emb_dir / "metadata.jsonl"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"임베딩 없음: {emb_path}\n먼저 01_embed.py를 실행하세요."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"메타데이터 없음: {meta_path}\n먼저 01_embed.py를 실행하세요."
        )

    # ── 로드 ──────────────────────────────────────────────────────────────────
    embeddings = np.load(emb_path)
    rows = load_jsonl(meta_path)
    labels = [r["label"] for r in rows]

    print(f"[로드] {embeddings.shape[0]:,}개 임베딩 (dim={embeddings.shape[1]})")

    # ── 카테고리 (로더 LABEL_MAP 자동 추출) ────────────────────────────────────
    source = cfg.data.source
    loader_cls = LOADERS.get(source)
    if loader_cls is None:
        raise ValueError(f"알 수 없는 source: {source!r}")
    categories: list[str] = loader_cls.CATEGORIES

    # ── 서브샘플링 (카테고리 균등) ──────────────────────────────────────────────
    umap_cfg = cfg.get("umap", OmegaConf.create({}))
    max_samples: int = umap_cfg.get("max_samples", 10000)

    if len(rows) > max_samples:
        print(f"[샘플링] {len(rows):,} → 최대 {max_samples:,}개 (카테고리 균등)")
        rng = random.Random(cfg.project.seed)
        indices_by_cat: dict[str, list[int]] = {c: [] for c in categories}
        for i, lb in enumerate(labels):
            if lb in indices_by_cat:
                indices_by_cat[lb].append(i)
        per_cat = max_samples // len(categories)
        selected: list[int] = []
        for cat, idxs in indices_by_cat.items():
            sampled = rng.sample(idxs, min(per_cat, len(idxs)))
            selected.extend(sampled)
            print(f"  {cat:20s}: {len(sampled):,}개 (전체 {len(idxs):,}개)")
        selected.sort()
        sub_embeddings = embeddings[selected]
        sub_rows = [rows[i] for i in selected]
        print(f"  → 최종 선택: {len(selected):,}개")
    else:
        sub_embeddings = embeddings
        sub_rows = rows
        print(f"  샘플링 생략 (전체 {len(rows):,}개 ≤ max_samples {max_samples:,})")

    # ── TSV 저장 ──────────────────────────────────────────────────────────────
    projector_dir = artifacts_dir / "projector" / model_name
    save_projector_tsv(sub_embeddings, sub_rows, projector_dir)


if __name__ == "__main__":
    main()
