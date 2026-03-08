"""
scripts/finetune/10_finetune.py
LoRA 기반 임베딩 모델 파인튜닝.

학습 방식: LoRA + MultipleNegativesRankingLoss
  - 전체 레이어 동결, attention query/value 레이어에만 소규모 어댑터 학습
  - Full fine-tuning 붕괴 없음, 원본 모델 능력 보존
  - 완료 후 어댑터를 base 모델에 병합(merge) → 별도 의존성 없이 추론 가능

실행 예:
  uv run python scripts/finetune/10_finetune.py model=mxbai_large +data=ourafla
  uv run python scripts/finetune/10_finetune.py model=mxbai_large +data=combined
  uv run python scripts/finetune/10_finetune.py model=bge_base_en +data=ourafla
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from datasets import Dataset
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)

# ── LoRA 설정 ──────────────────────────────────────────────────────────────────
LORA_R           = 16      # rank: 낮을수록 파라미터 적음 (8~32 권장)
LORA_ALPHA       = 32      # scaling = lora_alpha / r
LORA_DROPOUT     = 0.1
LORA_TARGET      = ["query", "value"]  # attention Q, V 레이어만 학습

# ── 학습 설정 ──────────────────────────────────────────────────────────────────
TRAIN_RATIO   = 0.9
EPOCHS        = 3          # LoRA는 붕괴 위험 없어서 3 epoch 가능
LEARNING_RATE = 1e-4       # LoRA는 full보다 높은 lr 가능 (어댑터만 학습)
WARMUP_RATIO  = 0.1
MAX_SEQ_LEN   = 256
SEED          = 42
OUTPUT_BASE   = Path("models/finetuned")


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


def make_pair_dataset(buckets: dict[str, list[str]], seed: int) -> Dataset:
    """(anchor, positive) 쌍 → datasets.Dataset 생성."""
    rng = random.Random(seed)
    anchors, positives = [], []
    for cat, texts in buckets.items():
        shuffled = texts.copy()
        rng.shuffle(shuffled)
        for anchor in shuffled:
            pos = rng.choice(texts)
            while pos == anchor and len(texts) > 1:
                pos = rng.choice(texts)
            anchors.append(anchor)
            positives.append(pos)
    combined = list(zip(anchors, positives))
    rng.shuffle(combined)
    anchors, positives = zip(*combined)
    return Dataset.from_dict({"anchor": list(anchors), "positive": list(positives)})


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
    print(f"  LoRA  : r={LORA_R}  alpha={LORA_ALPHA}  target={LORA_TARGET}")
    print(f"{'='*55}")

    # ── 데이터 준비 ──────────────────────────────────────────────────────────────
    buckets = load_jsonl(data_path)
    rng = random.Random(SEED)
    train_buckets: dict[str, list[str]] = {}
    for cat, texts in buckets.items():
        shuffled = texts.copy()
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * TRAIN_RATIO)
        train_buckets[cat] = shuffled[:cut]

    print(f"\n[Dataset 생성]")
    train_dataset = make_pair_dataset(train_buckets, SEED)
    print(f"  쌍 수: {len(train_dataset):,}")

    # ── 모델 로딩 + LoRA 적용 ─────────────────────────────────────────────────
    print(f"\n[모델 로딩] {model_id}")
    model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ_LEN

    # LoRA 설정 — FEATURE_EXTRACTION = 임베딩 모델용 TaskType
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET,
        bias="none",
    )
    # SentenceTransformer 내부 transformer 모델에 LoRA 적용
    model[0].auto_model = get_peft_model(model[0].auto_model, lora_cfg)
    model[0].auto_model.print_trainable_parameters()

    # ── Loss ─────────────────────────────────────────────────────────────────────
    loss = losses.MultipleNegativesRankingLoss(model)

    # ── TrainingArguments ─────────────────────────────────────────────────────────
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device == "cuda" and not use_bf16

    warmup_steps = int((len(train_dataset) // batch_size) * EPOCHS * WARMUP_RATIO)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=200,
        save_strategy="epoch",
        seed=SEED,
    )

    print(f"\n[학습 시작]")
    print(f"  epochs={EPOCHS}  batch={batch_size}  lr={LEARNING_RATE}")
    print(f"  bf16={use_bf16}  fp16={use_fp16}  warmup={warmup_steps}steps")

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    # ── LoRA 어댑터를 base 모델에 병합 후 저장 ──────────────────────────────────
    # merge_and_unload: LoRA 가중치를 base에 흡수 → 추론 시 peft 불필요
    print("\n[LoRA 병합] 어댑터를 base 모델에 흡수 중...")
    model[0].auto_model = model[0].auto_model.merge_and_unload()

    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    print(f"\n  파인튜닝 완료 → {final_path}")


if __name__ == "__main__":
    main()
