"""
scripts/explore/13_sim_compare.py
두 모델(BGE-m3-ko, KURE-v1)의 문장 쌍 cosine 유사도 비교.

실행:
    uv run python scripts/explore/13_sim_compare.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcd.embedding.bge_m3 import BGEM3Encoder
from qcd.embedding.sbert import SentenceTransformerEncoder

PAIRS = [
    ("죽고싶어 살기 싫어",         "살기 싫어",                "같은 의미 → 가까워야"),
    ("죽고싶어 살기 싫어",         "죽고 싶지 않아 살고 싶어",  "반대 의미 → 멀어야"),
    ("죽고싶어 살기 싫어",         "살고싶어 힘내자",           "반대 의미 → 멀어야"),
    ("죽고 싶지 않아 살고 싶어",   "살고싶어 힘내자",           "긍정끼리 → 가까워야"),
    ("나 죽어도 아무도 모를 것 같아", "죽고싶어 살기 싫어",       "같은 위기 → 가까워야"),
]

MODELS = [
    ("BGE-m3-ko", BGEM3Encoder(model_id="dragonkue/BGE-m3-ko", batch_size=32)),
    ("KURE-v1",   SentenceTransformerEncoder(model_id="nlpai-lab/KURE-v1", batch_size=32)),
]

# 모든 유니크 문장 한 번에 인코딩
all_sentences = list({s for pair in PAIRS for s in pair[:2]})

results: dict[str, dict[str, np.ndarray]] = {}
for name, enc in MODELS:
    print(f"\n[인코딩] {name}...")
    vecs = enc.encode(all_sentences)
    results[name] = {s: v for s, v in zip(all_sentences, vecs)}

# 결과 출력
WIDTH = 30
print(f"\n{'쌍':>{WIDTH*2+3}}    {'BGE-m3-ko':>10}  {'KURE-v1':>8}  비고")
print("-" * 90)
for a, b, note in PAIRS:
    sims = {}
    for name in results:
        sims[name] = float(np.dot(results[name][a], results[name][b]))
    better = "BGE ↑" if sims["BGE-m3-ko"] > sims["KURE-v1"] else "KURE ↑"
    diff = abs(sims["BGE-m3-ko"] - sims["KURE-v1"])
    print(
        f"{a[:WIDTH]:>{WIDTH}} ↔ {b[:WIDTH]:<{WIDTH}}"
        f"  {sims['BGE-m3-ko']:>10.3f}  {sims['KURE-v1']:>8.3f}"
        f"  {note}  (Δ{diff:.3f})"
    )
