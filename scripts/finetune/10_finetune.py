"""
scripts/finetune/10_finetune.py
한국어 감성대화 데이터로 임베딩 모델 파인튜닝.

학습 방식: MultipleNegativesRankingLoss
  (anchor, positive) 쌍만 입력 → 배치 내 다른 샘플을 자동으로 negative로 활용
  TripletLoss 대비 인코딩 회수 2/3로 감소, 품질도 대등이상

사용 예:
  python scripts/finetune/10_finetune.py --model bge_m3_ko
  python scripts/finetune/10_finetune.py --model kure_v1
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_FILE = Path("data/finetune_processed/ko_labeled.jsonl")
OUTPUT_BASE = Path("models/finetuned")

MODELS: dict[str, str] = {
    "bge_m3_ko": "dragonkue/BGE-m3-ko",
    "kure_v1":   "nlpai-lab/KURE-v1",
}

TRAIN_RATIO = 0.9
BATCH_SIZE  = 16       # VRAM 압박 해소 — 16GB GPU에서 최적
EPOCHS      = 3
WARMUP_RATIO = 0.1
MAX_SEQ_LEN  = 128
SEED        = 42


# ──────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────
def load_jsonl(path: Path) -> dict[str, list[str]]:
    """카테고리별 텍스트 리스트로 읽기."""
    buckets: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            buckets[item["label"]].append(item["text"])
    print("\n[데이터 로딩]")
    for cat, texts in buckets.items():
        print(f"  {cat:>8}: {len(texts):,}개")
    return dict(buckets)


# ──────────────────────────────────────────────
# Triplet 생성
# ──────────────────────────────────────────────
def make_pairs(
    buckets: dict[str, list[str]],
    seed: int,
) -> list[InputExample]:
    """(anchor, positive) 쌍 생성 — MNRL이 배치 내 다른 샘플을 negative로 활용."""
    rng = random.Random(seed)
    examples: list[InputExample] = []

    for anchor_cat, anchors in buckets.items():
        shuffled = anchors.copy()
        rng.shuffle(shuffled)
        for anchor in shuffled:
            pos = rng.choice(anchors)
            while pos == anchor and len(anchors) > 1:
                pos = rng.choice(anchors)
            examples.append(InputExample(texts=[anchor, pos]))

    rng.shuffle(examples)
    return examples


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────
def finetune(model_key: str) -> None:
    model_id = MODELS[model_key]
    output_dir = OUTPUT_BASE / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"모델: {model_id}")
    print(f"저장 경로: {output_dir}")
    print(f"{'='*50}")

    # 데이터 준비
    buckets = load_jsonl(DATA_FILE)

    # train/eval 분할 (카테고리별로 분할)
    rng = random.Random(SEED)
    train_buckets: dict[str, list[str]] = {}
    for cat, texts in buckets.items():
        shuffled = texts.copy()
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * TRAIN_RATIO)
        train_buckets[cat] = shuffled[:cut]

    print(f"\n[Pair 생성] (MNRL 방식)")
    train_examples = make_pairs(train_buckets, SEED)
    print(f"  Pair 수: {len(train_examples):,}")

    # 모델 로딩
    print(f"\n[모델 로딩] {model_id}")
    model = SentenceTransformer(model_id)
    model.max_seq_length = MAX_SEQ_LEN   # 8192 → 128: 패딩 제거로 속도 급상승

    # DataLoader + Loss
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    loss_fn = losses.MultipleNegativesRankingLoss(model=model)

    warmup_steps = int(len(train_loader) * EPOCHS * WARMUP_RATIO)

    # 학습
    print(f"\n[학습 시작]  epochs={EPOCHS}  batch={BATCH_SIZE}  warmup={warmup_steps}steps")
    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
        use_amp=True,       # fp16 mixed precision → 속도 2배, VRAM 절반
    )

    print(f"\n✅ 파인튜닝 완료 → {output_dir}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="임베딩 모델 파인튜닝")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        required=True,
        help="파인튜닝할 모델 키 (bge_m3_ko | kure_v1)",
    )
    args = parser.parse_args()
    finetune(args.model)


if __name__ == "__main__":
    main()
