"""
scripts/explore/11_proto_viz.py
프로토타입 문장 + 테스트 쿼리를 UMAP으로 2D 시각화.

사전학습 모델(BGE-m3-ko)로 임베딩 후 각 카테고리가
실제 공간에서 얼마나 분리되어 있는지 확인한다.

실행:
    uv run python scripts/explore/11_proto_viz.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import umap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcd.embedding.bge_m3 import BGEM3Encoder

# ── 프로토타입 & 쿼리 (08_prototype_test.py와 동일) ──────────────────────────
PROTOTYPES = {
    "distress": [
        "너무 힘들어 어떡해", "우울증 자가 진단", "아무것도 하기 싫을 때",
        "불안해서 잠 못 잘 때", "이유 없이 눈물나는 이유", "학교 가기 싫어 어떻게 해",
        "공황발작 증상", "삶이 의미없을 때", "자책 멈추는 법", "우울할 때 어떻게 해야 해",
    ],
    "anger": [
        "화가 너무 날 때 어떻게 해", "분노 조절 안 될 때", "짜증나서 미칠 것 같아",
        "누군가 때리고 싶을 때", "열받아서 물건 던지고 싶어", "욱하는 성격 고치는 법",
        "억울해서 화가 풀리지 않아", "참을 수 없이 화날 때", "화풀이 할 곳",
        "폭력적인 충동 생길 때",
    ],
    "neutral": [
        "오늘 점심 메뉴 추천", "수학 이차방정식 푸는 법", "친구 생일 선물 추천 10대",
        "오늘 서울 날씨", "롤 티어 올리는 법", "한국사 공부 방법", "홈트 루틴 초보",
        "맛있는 라면 끓이는 법", "유튜브 구독자 늘리는 법", "방학 알차게 보내는 법",
    ],
}

QUERIES = {
    "distress": [
        "매일 눈물나는 이유", "공황장애 증상 테스트", "살아있는 게 힘들 때",
        "학교 다니기 너무 힘들어", "세상에서 사라지고 싶어",
    ],
    "anger": [
        "화가 가라앉지 않을 때", "사람 때리고 싶다는 생각",
        "빡쳐서 아무것도 못 하겠어", "복수하는 법",
    ],
    "neutral": [
        "인스타 팔로워 늘리는 법", "고등학교 내신 올리는 방법",
        "오늘 급식 뭐야", "넷플릭스 뭐 볼지 추천",
    ],
}

# ── 색상 & 마커 ───────────────────────────────────────────────────────────────
COLORS = {"distress": "#3b82f6", "anger": "#ef4444", "neutral": "#22c55e"}
# 프로토타입 = 동그라미(●), 쿼리 = 별표(★), 평균벡터 = 다이아몬드(◆)
PROTO_MARKER = "o"
QUERY_MARKER = "*"
MEAN_MARKER  = "D"


def main() -> None:
    print("[모델 로딩] dragonkue/BGE-m3-ko")
    encoder = BGEM3Encoder(model_id="dragonkue/BGE-m3-ko", batch_size=32)

    # ── 모든 텍스트 임베딩 ────────────────────────────────────────────────────
    texts, labels, kinds = [], [], []

    for cat, sentences in PROTOTYPES.items():
        texts.extend(sentences)
        labels.extend([cat] * len(sentences))
        kinds.extend(["prototype"] * len(sentences))

    for cat, sentences in QUERIES.items():
        texts.extend(sentences)
        labels.extend([cat] * len(sentences))
        kinds.extend(["query"] * len(sentences))

    print(f"[인코딩] 총 {len(texts)}개 문장")
    vecs = encoder.encode(texts)                   # (N, D) — 이미 L2 정규화

    # 카테고리별 평균 벡터(프로토타입 벡터) 계산
    mean_texts, mean_labels, mean_kinds = [], [], []
    mean_vecs_list = []
    for cat, sentences in PROTOTYPES.items():
        cat_vecs = encoder.encode(sentences)
        mv = cat_vecs.mean(axis=0)
        mv /= np.linalg.norm(mv) + 1e-9
        mean_vecs_list.append(mv)
        mean_texts.append(f"[평균] {cat}")
        mean_labels.append(cat)
        mean_kinds.append("mean")

    all_vecs = np.vstack([vecs, np.array(mean_vecs_list)])
    all_labels = labels + mean_labels
    all_kinds = kinds + mean_kinds
    all_texts = texts + mean_texts

    # ── UMAP 2D 투영 ─────────────────────────────────────────────────────────
    print("[UMAP] 2D 투영 중...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=8, min_dist=0.3)
    coords = reducer.fit_transform(all_vecs)

    # ── 시각화 ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    for i, (x, y) in enumerate(coords):
        cat    = all_labels[i]
        kind   = all_kinds[i]
        color  = COLORS[cat]

        if kind == "prototype":
            ax.scatter(x, y, c=color, s=80, marker=PROTO_MARKER,
                       alpha=0.7, edgecolors="white", linewidths=0.4, zorder=3)
        elif kind == "query":
            ax.scatter(x, y, c=color, s=120, marker=QUERY_MARKER,
                       alpha=0.9, edgecolors="white", linewidths=0.4, zorder=4)
        else:  # mean
            ax.scatter(x, y, c=color, s=220, marker=MEAN_MARKER,
                       edgecolors="white", linewidths=1.5, zorder=5)
            ax.annotate(cat, (x, y), fontsize=8, color="white",
                        xytext=(6, 6), textcoords="offset points")

    # 범례
    legend_elems = [
        mpatches.Patch(color=COLORS["distress"], label="distress"),
        mpatches.Patch(color=COLORS["anger"],    label="anger"),
        mpatches.Patch(color=COLORS["neutral"],  label="neutral"),
        plt.scatter([], [], c="gray", s=80,  marker=PROTO_MARKER, label="prototype 문장"),
        plt.scatter([], [], c="gray", s=120, marker=QUERY_MARKER, label="test query"),
        plt.scatter([], [], c="gray", s=220, marker=MEAN_MARKER,  label="평균 벡터"),
    ]
    ax.legend(handles=legend_elems, loc="upper left",
              facecolor="#1e293b", labelcolor="white", fontsize=9)

    ax.set_title("프로토타입 & 쿼리 임베딩 공간 (BGE-m3-ko, 사전학습)",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    out = ROOT / "results" / "proto_viz.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
