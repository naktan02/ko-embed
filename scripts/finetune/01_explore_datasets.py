"""
scripts/finetune/01_explore_datasets.py
Phase 1: 3개 데이터셋 다운로드 및 기초 탐색.

사용법:
    uv run python scripts/finetune/01_explore_datasets.py

새 데이터셋 추가 시:
    1. src/qcd/loaders.py 에 로더 클래스 추가
    2. 아래 DATASET_SOURCES 딕셔너리에 항목 추가
"""

import json
import urllib.request
from pathlib import Path

import kagglehub
from datasets import load_dataset
from kagglehub import KaggleDatasetAdapter

from qcd.loaders import LOADERS

# ── 데이터셋 다운로드 소스 정의 ────────────────────────────────────────────────
# 새 데이터셋 추가 시 여기에만 항목을 추가하면 됩니다.
DATASET_SOURCES: dict[str, dict] = {
    "ourafla": {
        "type":    "huggingface",
        "hf_repo": "ourafla/Mental-Health_Text-Classification_Dataset",
        "raw_dir": Path("data/finetune_raw/ourafla"),
    },
    "depressionemo": {
        "type":       "github_json",
        "base_url":   "https://raw.githubusercontent.com/abuBakarSiddiqurRahman/DepressionEmo/main",
        "files":      ["train.json", "val.json", "test.json"],
        "raw_dir":    Path("data/finetune_raw/depressionemo"),
    },
    "cssrs": {
        "type":       "kaggle",
        "kaggle_id":  "av9ash/labelled-reddit-suicidewatch-posts-cssr-s",
        "raw_dir":    Path("data/finetune_raw/cssrs"),
    },
}


# ── 다운로드 함수 ──────────────────────────────────────────────────────────────

def _download_huggingface(name: str, cfg: dict) -> None:
    raw_dir: Path = cfg["raw_dir"]
    if raw_dir.exists() and any(raw_dir.glob("*.csv")):
        print(f"[{name}] 이미 다운로드됨, 건너뜀")
        return

    print(f"[{name}] HuggingFace 다운로드 중: {cfg['hf_repo']}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(cfg["hf_repo"])
    for split_name, split_ds in ds.items():
        out = raw_dir / f"{split_name}.csv"
        split_ds.to_csv(str(out))
        print(f"  {split_name}: {len(split_ds):,}행 → {out}")


def _download_github_json(name: str, cfg: dict) -> None:
    raw_dir: Path = cfg["raw_dir"]
    if raw_dir.exists() and any(raw_dir.glob("*.json")):
        print(f"[{name}] 이미 다운로드됨, 건너뜀")
        return

    print(f"[{name}] GitHub 다운로드 중: {cfg['base_url']}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname in cfg["files"]:
        url = f"{cfg['base_url']}/{fname}"
        dest = raw_dir / fname
        urllib.request.urlretrieve(url, str(dest))
        data = json.load(open(dest, encoding="utf-8"))
        print(f"  {fname}: {len(data):,}행 → {dest}")


def _download_kaggle(name: str, cfg: dict) -> None:
    raw_dir: Path = cfg["raw_dir"]
    if raw_dir.exists() and any(raw_dir.glob("*.csv")):
        print(f"[{name}] 이미 다운로드됨, 건너뜀")
        return

    print(f"[{name}] Kaggle 다운로드 중: {cfg['kaggle_id']}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        cfg["kaggle_id"],
        "",
    )
    out = raw_dir / "data.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"  전체: {len(df):,}행 → {out}")


_DOWNLOADERS = {
    "huggingface": _download_huggingface,
    "github_json": _download_github_json,
    "kaggle":      _download_kaggle,
}


def download_all() -> None:
    """DATASET_SOURCES에 정의된 모든 데이터셋을 다운로드."""
    for name, cfg in DATASET_SOURCES.items():
        _DOWNLOADERS[cfg["type"]](name, cfg)
    print()


# ── 탐색 함수 ──────────────────────────────────────────────────────────────────

def _collect_files(raw_dir: Path, source_type: str) -> list[Path]:
    """데이터셋 폴더에서 로더가 읽을 수 있는 파일 목록 반환."""
    ext = "*.json" if source_type == "github_json" else "*.csv"
    return sorted(raw_dir.glob(ext))


def _print_stats(name: str, rows: list[dict]) -> None:
    """로더에서 반환된 레코드 리스트의 통계를 출력 (raw 포맷 기준)."""
    if not rows:
        print("  ⚠️ 로드된 데이터 없음")
        return

    print(f"  총 {len(rows):,}행 | 평균 텍스트 {sum(len(r['text']) for r in rows) / len(rows):.0f}자")

    # 데이터셋별 원본 레이블/감정 분포 출력
    source = rows[0].get("source", name)

    if source == "ourafla":
        dist: dict[str, int] = {}
        for r in rows:
            k = r.get("original_label", "?")
            dist[k] = dist.get(k, 0) + 1
        print(f"  원본 레이블 분포: {dict(sorted(dist.items()))}")

    elif source == "depressionemo":
        dist = {}
        for r in rows:
            for emo in r.get("emotions", []):
                dist[emo] = dist.get(emo, 0) + 1
        sorted_dist = dict(sorted(dist.items(), key=lambda x: -x[1]))
        print(f"  감정 분포 (멀티레이블): {sorted_dist}")

    elif source == "cssrs":
        dist = {}
        for r in rows:
            k = r.get("cssrs_level", "?")
            dist[k] = dist.get(k, 0) + 1
        print(f"  C-SSRS 레벨 분포: {dict(sorted(dist.items()))}")

    # 샘플 5개
    print("  샘플:")
    for r in rows[:5]:
        preview = r["text"][:100].replace("\n", " ")
        if source == "ourafla":
            tag = r.get("original_label", "?")
        elif source == "depressionemo":
            tag = str(r.get("emotions", []))
        else:
            tag = f"level={r.get('cssrs_level', '?')}"
        print(f"    [{tag}] {preview}")


def explore_all() -> None:
    """LOADERS에 등록된 파인튜닝 로더로 각 데이터셋 탐색."""
    FINETUNE_LOADERS = ("ourafla", "depressionemo", "cssrs")
    sep = "=" * 64

    print(sep)
    print("Phase 1 — 데이터셋 기초 탐색")
    print(sep)

    for name in FINETUNE_LOADERS:
        loader_cls = LOADERS.get(name)
        src_cfg = DATASET_SOURCES.get(name)
        if loader_cls is None or src_cfg is None:
            continue

        loader = loader_cls()
        files = _collect_files(src_cfg["raw_dir"], src_cfg["type"])

        print(f"\n📁 {name} ({len(files)}개 파일)")

        all_rows: list[dict] = []
        for fpath in files:
            rows = loader.load(fpath)
            all_rows.extend(rows)
            print(f"  └ {fpath.name}: {len(rows):,}행")

        _print_stats(name, all_rows)


if __name__ == "__main__":
    download_all()
    explore_all()
