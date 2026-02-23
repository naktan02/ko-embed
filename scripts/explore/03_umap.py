"""
03_umap.py
임베딩을 UMAP으로 2D 축소 후 시각화합니다.

실행 예시:
  uv run python scripts/explore/03_umap.py
  uv run python scripts/explore/03_umap.py model=bge_m3_ko
  uv run python scripts/explore/03_umap.py umap.n_neighbors=30 umap.min_dist=0.1
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    embed_dir = Path(cfg.data.artifacts_dir) / "embeddings" / cfg.model.name
    out_dir = Path(cfg.data.artifacts_dir) / "umap" / cfg.model.name

    # ── 임베딩 로드 ──────────────────────────────────────────────────────────
    emb_path = embed_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"임베딩이 없습니다: {emb_path}\n"
            "먼저 scripts/explore/01_embed.py 를 실행하세요."
        )

    embeddings: np.ndarray = np.load(emb_path)
    labels: list[str] = json.loads(
        (embed_dir / "labels.json").read_text(encoding="utf-8")
    )
    texts: list[str] = json.loads(
        (embed_dir / "texts.json").read_text(encoding="utf-8")
    )
    categories = sorted(set(labels))
    print(f"[로드] embeddings {embeddings.shape}, 카테고리: {categories}")

    # ── 균등 샘플링 (max_samples 초과 시) ────────────────────────────────────
    max_samples: int = cfg.umap.max_samples
    if len(embeddings) > max_samples:
        rng = np.random.default_rng(cfg.project.seed)
        n_per_cat = max_samples // len(categories)
        idx: list[int] = []
        for cat in categories:
            cat_idx = [i for i, l in enumerate(labels) if l == cat]
            sampled = rng.choice(cat_idx, size=min(n_per_cat, len(cat_idx)), replace=False)
            idx.extend(sampled.tolist())
        rng.shuffle(idx)
        embeddings = embeddings[idx]
        labels = [labels[i] for i in idx]
        texts = [texts[i] for i in idx]
        print(f"[샘플링] {len(embeddings)}개 (카테고리당 최대 {n_per_cat}개)")

    # ── UMAP 차원 축소 ────────────────────────────────────────────────────────
    try:
        from umap import UMAP  # type: ignore
    except ImportError:
        raise ImportError("umap-learn 이 필요합니다: uv add umap-learn")

    print(
        f"[UMAP] n_neighbors={cfg.umap.n_neighbors}, "
        f"min_dist={cfg.umap.min_dist} ..."
    )
    reducer = UMAP(
        n_neighbors=cfg.umap.n_neighbors,
        min_dist=cfg.umap.min_dist,
        n_components=2,
        random_state=cfg.project.seed,
        metric="cosine",
    )
    coords: np.ndarray = reducer.fit_transform(embeddings)
    print(f"[UMAP] 완료 coords {coords.shape}")

    # ── 시각화 ───────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        raise ImportError("matplotlib 이 필요합니다: uv add matplotlib")

    palette = plt.cm.get_cmap("tab10", len(categories))
    cat2color = {cat: palette(i) for i, cat in enumerate(categories)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for cat in categories:
        mask = np.array([l == cat for l in labels])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cat2color[cat]], label=cat,
            alpha=0.6, s=8, linewidths=0,
        )
    ax.legend(markerscale=3, framealpha=0.7)
    ax.set_title(f"UMAP — {cfg.model.name}")
    ax.axis("off")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "umap.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[완료] {png_path}")

    # coords + labels도 저장 (04_projector 등 후속 스크립트에서 활용)
    np.save(out_dir / "coords.npy", coords)
    (out_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "texts.json").write_text(
        json.dumps(texts, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
