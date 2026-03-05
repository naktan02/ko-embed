"""
scripts/explore/12_semantic_viz.py
의미 근접성 시각화 — BGE-m3-ko vs KURE-v1 나란히 비교.

문장들이 두 모델의 임베딩 공간에서 어떻게 배치되는지 확인한다.
의미가 비슷한 문장끼리 실제로 가깝게 위치하는가?

실행:
    uv run python scripts/explore/12_semantic_viz.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows 한국어 폰트
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import matplotlib.patheffects as pe
import numpy as np
import umap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcd.embedding.bge_m3 import BGEM3Encoder
from qcd.embedding.sbert import SentenceTransformerEncoder

# ── 테스트 문장 (순수 배치 확인용) ────────────────────────────────────────────────────────────
SENTENCES = [
    # ── distress ──────────────────────────────────────────────────────────
    ("distress", "죽고싶어 살기 싫어"),          # crisis
    ("distress", "죽어버리고 싶다"),
    ("distress", "사라지고 싶어"),

    # ── 반대 의미 쌍 (비교용) ─────────────────────────────────────────────
    ("distress", "살기 싫어"),                   # negative polarity
    ("neutral",  "죽고 싶지 않아 살고 싶어"),    # positive polarity — neutral에 더 가까울까?
    ("neutral",  "살고싶어 힘내자"),

    # ── anger ─────────────────────────────────────────────────────────────
    ("anger", "때리고싶다 어떡해"),
    ("anger", "분노조절장애 자가진단"),
    ("anger", "화병 증상"),
    ("anger", "복수하고 싶어"),

    # ── 경계 케이스 ───────────────────────────────────────────────────────
    ("distress", "죽고싶다"),                    # 1단어 crisis
    ("anger",    "죽여버리고 싶어"),             # 타인 지향 → anger
    ("distress", "나 죽어도 아무도 모를 것 같아"),

    # ── neutral ───────────────────────────────────────────────────────────
    ("neutral", "수능 d-100 공부법"),
    ("neutral", "배달의민족 쿠폰"),
    ("neutral", "오늘 날씨 서울"),
    ("neutral", "아이폰 배터리 교체 비용"),
    ("neutral", "넷플릭스 뭐 볼지 추천"),
]

COLORS = {
    "distress": "#60a5fa",
    "anger":    "#f87171",
    "neutral":  "#4ade80",
}

MODELS = [
    ("BGE-m3-ko",  BGEM3Encoder(model_id="dragonkue/BGE-m3-ko", batch_size=32)),
    ("KURE-v1",    SentenceTransformerEncoder(model_id="nlpai-lab/KURE-v1", batch_size=32)),
]


def plot_panel(ax, coords, labels, texts, title):
    ax.set_facecolor("#0f172a")
    ax.set_title(title, color="white", fontsize=12, pad=8)
    ax.tick_params(colors="#475569")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.set_xticks([])
    ax.set_yticks([])

    for i, (x, y) in enumerate(coords):
        color = COLORS[labels[i]]
        ax.scatter(x, y, c=color, s=80, zorder=3,
                   edgecolors="white", linewidths=0.4, alpha=0.9)
        txt = ax.annotate(
            texts[i], (x, y),
            fontsize=8, color="white",
            xytext=(5, 4), textcoords="offset points",
            zorder=4,
        )
        txt.set_path_effects([
            pe.withStroke(linewidth=1.8, foreground="#0f172a")
        ])


def main() -> None:
    labels = [l for l, _ in SENTENCES]
    texts  = [s for _, s in SENTENCES]

    fig, axes = plt.subplots(1, len(MODELS), figsize=(18, 9))
    fig.patch.set_facecolor("#0f172a")

    for ax, (model_name, encoder) in zip(axes, MODELS):
        print(f"\n[인코딩] {model_name}")
        vecs = encoder.encode(texts)

        print(f"[UMAP] {model_name}")
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=6, min_dist=0.25)
        coords = reducer.fit_transform(vecs)

        plot_panel(ax, coords, labels, texts, model_name)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color=c, label=l) for l, c in COLORS.items()],
        loc="lower center", ncol=3, facecolor="#1e293b",
        labelcolor="white", fontsize=10, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle("의미 근접성 — 두 모델 임베딩 공간 비교", color="white", fontsize=13)
    plt.tight_layout(pad=2)

    out = ROOT / "results" / "semantic_viz.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
