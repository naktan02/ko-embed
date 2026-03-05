"""
scripts/finetune/10_finetune.py
Hydra 설정 기반 임베딩 모델 파인튜닝.

학습 방식: MultipleNegativesRankingLoss
  (anchor, positive) 쌍만 입력 → 배치 내 다른 샘플을 자동으로 negative로 활용

실행 예:
  # ourafla 단독
  uv run python scripts/finetune/10_finetune.py model=mxbai_large +data=ourafla

  # ourafla + cssrs 합산
  uv run python scripts/finetune/10_finetune.py model=mxbai_large +data=combined

  # 다른 모델
  uv run python scripts/finetune/10_finetune.py model=bge_base_en +data=ourafla
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

TRAIN_RATIO  = 0.9
EPOCHS       = 3
WARMUP_RATIO = 0.1
MAX_SEQ_LEN  = 128
SEED         = 42
OUTPUT_BASE  = Path("models/finetuned")


def load_jsonl(path: Path) -> dict[str, list[str]]:
    """카테고리별 텍스트 리스트로 읽기."""
    buckets: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            buckets[item["label"]].append(item["text"])
    print("\n[데이터 로딩]")
    for cat, texts in sorted(buckets.items()):
        print(f"  {cat:<12}: {len(texts):,}개")
    return dict(buckets)


def make_pairs(buckets: dict[str, list[str]], seed: int) -> list[InputExample]:
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


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_name: str = cfg.model.get("name", "model")
    model_id:   str = cfg.model.get("model_id", "")
    batch_size: int = cfg.model.get("batch_size", 16)
    data_name:  str = cfg.data.get("name", "data")
    data_path       = Path(cfg.data.get("jsonl", ""))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = OUTPUT_BASE / f"{model_name}_{data_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  모델  : {model_id}")
    print(f"  데이터: {data_path}  ({data_name})")
    print(f"  저장  : {output_dir}")
    print(f"  디바이스: {device}  |  batch={batch_size}")
    print(f"{'='*55}")

    # 데이터 로딩 & train 분할
    buckets = load_jsonl(data_path)
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
    model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ_LEN

    # DataLoader + Loss
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    loss_fn = losses.MultipleNegativesRankingLoss(model=model)
    warmup_steps = int(len(train_loader) * EPOCHS * WARMUP_RATIO)

    print(f"\n[학습 시작]  epochs={EPOCHS}  batch={batch_size}  warmup={warmup_steps}steps")
    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
        use_amp=(device == "cuda"),
    )

    print(f"\n✅ 파인튜닝 완료 → {output_dir}")


if __name__ == "__main__":
    main()
