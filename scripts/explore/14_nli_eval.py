"""
scripts/explore/14_nli_eval.py
KorNLI로 두 모델의 의미 분리 능력 비교.

entailment(같은 의미) → cosine 높아야
contradiction(반대 의미) → cosine 낮아야
neutral → 중간

HuggingFace datasets 라이브러리로 kakaobrain/kor_nli 로드.

실행:
    uv run python scripts/explore/14_nli_eval.py
"""

import sys, random
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcd.embedding.bge_m3 import BGEM3Encoder
from qcd.embedding.sbert import SentenceTransformerEncoder

# ── 설정 ───────────────────────────────────────────────────────────────────────
N_SAMPLES   = 500   # 라벨당 샘플 수
SEED        = 42
LABEL_NAMES = ["entailment", "neutral", "contradiction"]   # label 0/1/2
MODELS = [
    ("BGE-m3-ko", lambda: BGEM3Encoder(model_id="dragonkue/BGE-m3-ko",       batch_size=64)),
    ("KURE-v1",   lambda: SentenceTransformerEncoder(model_id="nlpai-lab/KURE-v1", batch_size=64)),
]


def load_data() -> dict[str, list[tuple[str, str]]]:
    """kor_nli mnli split에서 라벨별 N_SAMPLES 쌍 반환."""
    from datasets import load_dataset
    print("[데이터 로드] kakaobrain/kor_nli (mnli)...")
    ds = load_dataset("kakaobrain/kor_nli", "multi_nli", split="train")

    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in ds:
        lbl = LABEL_NAMES[row["label"]]
        buckets[lbl].append((row["premise"], row["hypothesis"]))

    random.seed(SEED)
    return {
        lbl: random.sample(pairs, min(N_SAMPLES, len(pairs)))
        for lbl, pairs in buckets.items()
    }


def eval_model(name: str, enc, data: dict[str, list[tuple[str, str]]]) -> dict[str, float]:
    """라벨별 평균 cosine similarity 계산."""
    print(f"\n[평가] {name}")
    result = {}
    for lbl, pairs in data.items():
        premises    = [p for p, _ in pairs]
        hypotheses  = [h for _, h in pairs]
        all_texts   = premises + hypotheses
        vecs = enc.encode(all_texts)
        p_vecs = vecs[:len(premises)]
        h_vecs = vecs[len(premises):]
        # 정규화 후 dot = cosine
        p_n = p_vecs / (np.linalg.norm(p_vecs, axis=1, keepdims=True) + 1e-9)
        h_n = h_vecs / (np.linalg.norm(h_vecs, axis=1, keepdims=True) + 1e-9)
        sims = (p_n * h_n).sum(axis=1)
        result[lbl] = float(sims.mean())
        print(f"  {lbl:>15}: {result[lbl]:.4f}  (n={len(pairs)})")
    return result


def main() -> None:
    data = load_data()

    all_results: dict[str, dict[str, float]] = {}
    for name, make_enc in MODELS:
        enc = make_enc()
        all_results[name] = eval_model(name, enc, data)

    # 최종 비교 출력
    print(f"\n{'='*60}")
    print(f"{'모델':<14}", "  ".join(f"{l:>15}" for l in LABEL_NAMES),
          "  separation(E-C)")
    print("-" * 60)
    for name, res in all_results.items():
        sep = res["entailment"] - res["contradiction"]
        row = f"{name:<14}" + "  ".join(f"{res[l]:>15.4f}" for l in LABEL_NAMES)
        print(f"{row}  {sep:>10.4f}")
    print(f"{'='*60}")
    print("separation = entailment - contradiction (높을수록 의미 분리 능력 ↑)")


if __name__ == "__main__":
    main()
