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
        # feature_engineered.csv는 컬럼 수가 달라 스키마 충돌 → 원하는 파일만 지정
        "hf_files": {
            "train": "mental_heath_unbanlanced.csv",
            "test":  "mental_health_combined_test.csv",
        },
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
    if "hf_files" in cfg:
        # 파일별 스키마가 달라 load_dataset()이 실패하는 경우 개별 파일 지정
        data_files = {
            split: f"hf://datasets/{cfg['hf_repo']}/{fname}"
            for split, fname in cfg["hf_files"].items()
        }
        ds = load_dataset("csv", data_files=data_files)
    else:
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


def _print_stats(name: str, rows: list[dict], loader) -> None:
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

    # 샘플 30개 — loader.get_label()로 라벨별 균등 추출
    SAMPLE_TOTAL = 30

    groups: dict[str, list[dict]] = {}
    for r in rows:
        k = loader.get_label(r)
        groups.setdefault(k, []).append(r)

    n_per = max(1, SAMPLE_TOTAL // len(groups))
    samples = [r for grp in groups.values() for r in grp[:n_per]]

    print(f"  샘플 (라벨별 최대 {n_per}개):")
    for r in samples:
        preview = r["text"][:200].replace("\n", " ")
        print(f"    [{loader.get_label(r)}] {preview}")


def explore_all(targets: list[str] | None = None) -> None:
    """LOADERS에 등록된 파인튜닝 로더로 각 데이터셋 탐색.

    targets가 None이면 DATASET_SOURCES의 모든 키를 탐색.
    """
    names = targets if targets is not None else list(DATASET_SOURCES.keys())
    sep = "=" * 64

    print(sep)
    print("Phase 1 — 데이터셋 기초 탐색")
    print(sep)

    for name in names:
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

        _print_stats(name, all_rows, loader)


if __name__ == "__main__":
    import sys
    # 인자 없으면 전체 실행, 있으면 해당 데이터셋만
    # 예) uv run python scripts/finetune/01_explore_datasets.py ourafla cssrs
    targets = sys.argv[1:] or list(DATASET_SOURCES.keys())
    unknown = [t for t in targets if t not in DATASET_SOURCES]
    if unknown:
        print(f"[오류] 알 수 없는 데이터셋: {unknown}")
        print(f"  사용 가능: {list(DATASET_SOURCES.keys())}")
        sys.exit(1)

    for name in targets:
        cfg = DATASET_SOURCES[name]
        _DOWNLOADERS[cfg["type"]](name, cfg)
    print()
    explore_all(targets)
