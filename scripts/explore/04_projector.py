"""
04_projector.py
?„ë² ?©ì�„ TensorFlow Embedding Projector??TSVë¡?ë³€?˜í•©?ˆë‹¤.
UMAP ?†ì�´, ?�ë³¸ ?„ë² ??ê³ ì°¨?? ê·¸ë?ë¡??�ëŠ” ì¹´í…Œê³ ë¦¬ ê· ë“± ?˜í”Œë§�í•˜???€?¥í•©?ˆë‹¤.

?¤í–‰ ?ˆì‹œ:
  uv run python scripts/explore/04_projector.py
  uv run python scripts/explore/04_projector.py model=sroberta
  uv run python scripts/explore/04_projector.py umap.max_samples=5000

ì¶œë ¥:
  artifacts/projector/{model_name}/tensor.tsv   ??ë²¡í„° (?¤ë�” ?†ì�Œ)
  artifacts/projector/{model_name}/metadata.tsv ??ë©”í??°ì�´??(ì²??‰ì�´ ì»¬ëŸ¼ëª?
  ??projector.tensorflow.org ?�ì„œ ???Œì�¼ ?…ë¡œ??
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
    """TSV ?€???ˆì „?˜ê²Œ ?¤ì–´ê°€??ë¬¸ìž�?´ë¡œ ë³€??
    \\t \\r \\n ??ê³µë°±?¼ë¡œ ?€ì²´í•˜??TSV ?Œì‹± ?¤ë¥˜ ë°©ì?.
    """
    return value.replace("\r", " ").replace("\n", " ").replace("\t", " ")


def save_projector_tsv(
    embeddings: np.ndarray,
    rows: list[dict],
    out_dir: Path,
    label_only: bool = False,
) -> None:
    """Embedding Projector (projector.tensorflow.org)??TSV ?Œì�¼ ?�ì„±.

    - tensor.tsv : ?¤ë�” ?†ì�Œ, ??œ¼ë¡?êµ¬ë¶„??ë²¡í„° (????= ??ë²¡í„°)
    - metadata.tsv:
        * label_only=False (ê¸°ë³¸): ì²??‰ì�´ ì»¬ëŸ¼ ?¤ë�”, ?´í›„ ?°ì�´??
          ??ì»¬ëŸ¼??2ê°??´ìƒ�?´ë©´ Projectorê°€ ?¤ë�”ë¥??�ë�™ ?¸ì‹�
          ?????? tensor????+ 1 (?¤ë�”)
        * label_only=True: ?¤ë�” ?†ì�´ ?¼ë²¨ 1?´ë§Œ ?€??
          ?????? tensor???˜ì? ?™ì�¼?´ì•¼ ??

    ì£¼ì�˜: newline='\\n' ?¼ë¡œ ?´ì–´ Windows CRLF ?�ë�™ë³€??ë°©ì?
    """
    assert len(embeddings) == len(rows), (
        f"TSV ë¶ˆì�¼ì¹? tensor={len(embeddings)}, metadata={len(rows)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ?€?€ tensor.tsv ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    tensor_path = out_dir / "tensor.tsv"
    with tensor_path.open("w", encoding="utf-8", newline="\n") as f:
        for vec in embeddings:
            f.write("\t".join(f"{v:.6f}" for v in vec) + "\n")

    # ?€?€ metadata.tsv ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    meta_path = out_dir / "metadata.tsv"

    if label_only:
        # ?¨ìˆœ ?¼ë²¨ 1?? ?¤ë�” ?†ì�Œ (tensor????== metadata????
        with meta_path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(_clean(str(row.get("label", ""))) + "\n")
        meta_mode = "label-only (?¤ë�” ?†ì�Œ)"
    else:
        # ë©€?°ì»¬??+ ?¤ë�” (tensor????== metadata????- 1)
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
        meta_mode = f"ë©€?°ì»¬??{len(headers)}??+ ?¤ë�”"

    print(f"[?€?? Embedding Projector TSV ??{out_dir}/")
    print(f"  tensor.tsv   : {len(embeddings):,}??)
    print(f"  metadata.tsv : {len(rows):,}?? [{meta_mode}]")
    print(f"  ??projector.tensorflow.org ?�ì„œ tensor.tsv + metadata.tsv ?…ë¡œ??)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_name: str = cfg.model.get("name", "model")
    artifacts_dir = Path(cfg.data.artifacts_dir)
    emb_dir = artifacts_dir / "embeddings" / model_name
    emb_path = emb_dir / "embeddings.npy"
    meta_path = emb_dir / "metadata.jsonl"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"?„ë² ???†ì�Œ: {emb_path}\në¨¼ì? 01_embed.pyë¥??¤í–‰?˜ì„¸??"
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"ë©”í??°ì�´???†ì�Œ: {meta_path}\në¨¼ì? 01_embed.pyë¥??¤í–‰?˜ì„¸??"
        )

    # ?€?€ ë¡œë“œ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    embeddings = np.load(emb_path)
    rows = load_jsonl(meta_path)
    labels = [r["label"] for r in rows]

    print(f"[ë¡œë“œ] {embeddings.shape[0]:,}ê°??„ë² ??(dim={embeddings.shape[1]})")

    # ?€?€ ì¹´í…Œê³ ë¦¬ (ë¡œë�” LABEL_MAP ?�ë�™ ?„ì¶œ) ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    source = cfg.data.source
    loader_cls = LOADERS.get(source)
    if loader_cls is None:
        raise ValueError(f"?????†ëŠ” source: {source!r}")
    categories: list[str] = loader_cls.CATEGORIES

    # ?€?€ ?œë¸Œ?˜í”Œë§?(ì¹´í…Œê³ ë¦¬ ê· ë“±) ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    umap_cfg = cfg.get("umap", OmegaConf.create({}))
    max_samples: int = umap_cfg.get("max_samples", 10000)

    if len(rows) > max_samples:
        print(f"[?˜í”Œë§? {len(rows):,} ??ìµœë? {max_samples:,}ê°?(ì¹´í…Œê³ ë¦¬ ê· ë“±)")
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
            print(f"  {cat:20s}: {len(sampled):,}ê°?(?„ì²´ {len(idxs):,}ê°?")
        selected.sort()
        sub_embeddings = embeddings[selected]
        sub_rows = [rows[i] for i in selected]
        print(f"  ???¤ì œ ? íƒ�: {len(selected):,}ê°?)
    else:
        sub_embeddings = embeddings
        sub_rows = rows
        print(f"  ?˜í”Œë§??�ëžµ (?„ì²´ {len(rows):,}ê°???max_samples {max_samples:,})")

    # ?€?€ TSV ?€???€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    projector_dir = artifacts_dir / "projector" / model_name
    save_projector_tsv(sub_embeddings, sub_rows, projector_dir)


if __name__ == "__main__":
    main()
