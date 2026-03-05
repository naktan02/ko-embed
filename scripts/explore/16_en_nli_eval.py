"""
scripts/explore/16_en_nli_eval.py
MultiNLI(영어)로 임베딩 모델의 의미 분리 능력 비교.

entailment(같은 의미) → cosine 높아야
contradiction(반대 의미) → cosine 낮아야
neutral → 중간

실행:
  uv run python scripts/explore/16_en_nli_eval.py model=bge_large_en
  uv run python scripts/explore/16_en_nli_eval.py model=mxbai_large
  uv run python scripts/explore/16_en_nli_eval.py model=bge_base_en
  uv run python scripts/explore/16_en_nli_eval.py model=mpnet_base
  uv run python scripts/explore/16_en_nli_eval.py model=bge_m3
"""

from __future__ import annotations

import random
from collections import defaultdict

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

N_SAMPLES   = 500   # 라벨당 샘플 수
SEED        = 42
LABEL_NAMES = ["entailment", "neutral", "contradiction"]  # label 0/1/2


def load_data() -> dict[str, list[tuple[str, str]]]:
    """MultiNLI validation_matched split에서 라벨별 N_SAMPLES 쌍 반환."""
    from datasets import load_dataset
    print("[데이터 로드] nyu-mll/multi_nli (validation_matched)...")
    ds = load_dataset("nyu-mll/multi_nli", split="validation_matched")

    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in ds:
        if row["label"] == -1:  # 레이블 없는 샘플 제외
            continue
        lbl = LABEL_NAMES[row["label"]]
        buckets[lbl].append((row["premise"], row["hypothesis"]))

    random.seed(SEED)
    return {
        lbl: random.sample(pairs, min(N_SAMPLES, len(pairs)))
        for lbl, pairs in buckets.items()
    }


def eval_model(
    name: str,
    enc,
    data: dict[str, list[tuple[str, str]]],
) -> dict[str, float]:
    """라벨별 평균 cosine similarity 계산."""
    print(f"\n[평가] {name}")
    result = {}
    for lbl, pairs in data.items():
        premises   = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        vecs = enc.encode(premises + hypotheses)
        p_vecs = vecs[:len(premises)]
        h_vecs = vecs[len(premises):]
        # 정규화 후 dot = cosine
        p_n = p_vecs / (np.linalg.norm(p_vecs, axis=1, keepdims=True) + 1e-9)
        h_n = h_vecs / (np.linalg.norm(h_vecs, axis=1, keepdims=True) + 1e-9)
        sims = (p_n * h_n).sum(axis=1)
        result[lbl] = float(sims.mean())
        print(f"  {lbl:>15}: {result[lbl]:.4f}  (n={len(pairs)})")
    return result


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_name: str = cfg.model.get("name", "model")
    encoder = instantiate(cfg.model)

    data = load_data()
    result = eval_model(model_name, encoder, data)

    sep = "=" * 60
    sep2 = "-" * 60
    print(f"\n{sep}")
    print(f"  MultiNLI 평가 결과  |  모델: {model_name}")
    print(sep2)
    print(f"  {'레이블':>15}  {'평균 cosine':>12}  {'해석':>10}")
    print(sep2)
    for lbl in LABEL_NAMES:
        hint = {"entailment": "높을수록 ↑", "neutral": "중간이 이상적", "contradiction": "낮을수록 ↑"}
        print(f"  {lbl:>15}  {result[lbl]:>12.4f}  {hint[lbl]:>10}")
    sep_score = result["entailment"] - result["contradiction"]
    print(sep2)
    print(f"  separation (E-C) : {sep_score:.4f}  (높을수록 의미 분리 능력 ↑)")
    print(f"  ▶ 해석: 0.25↑ 우수 / 0.20↑ 양호 / 0.15↓ 개선 필요")
    print(sep)


if __name__ == "__main__":
    main()
