"""
3개 데이터셋 다운로드 및 기초 탐색 스크립트.
- ourafla/Mental-Health_Text-Classification_Dataset (HuggingFace)
- nikhileswarkomati/suicide-watch (Kaggle) → C-SSRS와 동일 출처
- DepressionEmo (GitHub)

사용법:
  python scripts/finetune/01_download_datasets.py
"""

from pathlib import Path
from datasets import load_dataset
import json
import csv
import subprocess
import urllib.request
import zipfile

DATA_DIR = Path("data/finetune_raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 1. ourafla 4-Class (HuggingFace)
# ──────────────────────────────────────────────
def download_ourafla():
    """Suicidal / Depression / Anxiety / Normal 4분류 데이터셋"""
    out_dir = DATA_DIR / "ourafla"
    if out_dir.exists() and any(out_dir.glob("*.csv")):
        print("[ourafla] 이미 다운로드됨, 건너뜀")
        return

    print("[ourafla] HuggingFace에서 다운로드 중...")
    ds = load_dataset("ourafla/Mental-Health_Text-Classification_Dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_ds in ds.items():
        path = out_dir / f"{split_name}.csv"
        split_ds.to_csv(str(path))
        print(f"  {split_name}: {len(split_ds)}행 → {path}")

    print("[ourafla] 완료\n")


# ──────────────────────────────────────────────
# 2. DepressionEmo (GitHub)
# ──────────────────────────────────────────────
def download_depressionemo():
    """8가지 우울 감정 멀티레이블 데이터셋 (GitHub)"""
    out_dir = DATA_DIR / "depressionemo"
    if out_dir.exists() and any(out_dir.glob("*.json")):
        print("[DepressionEmo] 이미 다운로드됨, 건너뜀")
        return

    print("[DepressionEmo] GitHub에서 다운로드 중...")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/abuBakarSiddiqurRahman/DepressionEmo/main"
    files = ["train.json", "val.json", "test.json"]

    for fname in files:
        url = f"{base_url}/{fname}"
        dest = out_dir / fname
        try:
            urllib.request.urlretrieve(url, str(dest))
            # 행 수 확인
            with open(dest, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  {fname}: {len(data)}행 → {dest}")
        except Exception as e:
            print(f"  ⚠️ {fname} 다운로드 실패: {e}")
            # 대체 경로 시도 (Dataset/ 하위)
            url2 = f"{base_url}/Dataset/{fname}"
            try:
                urllib.request.urlretrieve(url2, str(dest))
                with open(dest, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"  {fname}: {len(data)}행 → {dest} (Dataset/ 경로)")
            except Exception as e2:
                print(f"  ❌ {fname} 다운로드 최종 실패: {e2}")

    print("[DepressionEmo] 완료\n")


# ──────────────────────────────────────────────
# 3. C-SSRS Reddit SuicideWatch (Kaggle)
# ──────────────────────────────────────────────
def download_cssrs():
    """
    C-SSRS 7단계 레이블 데이터.
    Kaggle 데이터는 수동 다운로드가 필요할 수 있음.
    kaggle API가 설정되어 있으면 자동 다운로드 시도.
    """
    out_dir = DATA_DIR / "cssrs"
    if out_dir.exists() and any(out_dir.glob("*.csv")):
        print("[C-SSRS] 이미 다운로드됨, 건너뜀")
        return

    print("[C-SSRS] Kaggle에서 다운로드 시도...")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "av9ash/labelled-reddit-suicidewatch-posts-cssr-s",
                "-p", str(out_dir), "--unzip"
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  → {out_dir} 에 다운로드 완료")
    except FileNotFoundError:
        print("  ⚠️ kaggle CLI가 없습니다.")
        print("  수동 다운로드 필요:")
        print("  https://www.kaggle.com/datasets/av9ash/labelled-reddit-suicidewatch-posts-cssr-s")
        print(f"  다운로드 후 {out_dir} 폴더에 CSV 파일을 넣어주세요.")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️ kaggle 다운로드 실패: {e.stderr}")
        print("  수동 다운로드 필요:")
        print("  https://www.kaggle.com/datasets/av9ash/labelled-reddit-suicidewatch-posts-cssr-s")
        print(f"  다운로드 후 {out_dir} 폴더에 CSV 파일을 넣어주세요.")

    print("[C-SSRS] 완료\n")


# ──────────────────────────────────────────────
# 4. 기초 탐색
# ──────────────────────────────────────────────
def explore_datasets():
    """다운로드된 데이터셋의 기본 통계 출력"""
    print("=" * 60)
    print("데이터셋 기초 탐색")
    print("=" * 60)

    # ourafla
    ourafla_dir = DATA_DIR / "ourafla"
    for csv_path in sorted(ourafla_dir.glob("*.csv")):
        print(f"\n📁 ourafla/{csv_path.name}")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"  행 수: {len(rows)}")
        if rows:
            print(f"  컬럼: {list(rows[0].keys())}")
            # 레이블 분포
            labels = {}
            text_col = None
            for col in rows[0].keys():
                if "status" in col.lower() or "label" in col.lower():
                    for r in rows:
                        lbl = r.get(col, "unknown")
                        labels[lbl] = labels.get(lbl, 0) + 1
                if "text" in col.lower() and text_col is None:
                    text_col = col
            if labels:
                print(f"  레이블 분포: {labels}")
            # 샘플 3개
            if text_col:
                print(f"  샘플 ('{text_col}' 컬럼):")
                for r in rows[:3]:
                    txt = r[text_col][:100] + ("..." if len(r[text_col]) > 100 else "")
                    lbl = r.get("status", r.get("label", "?"))
                    print(f"    [{lbl}] {txt}")

    # DepressionEmo
    demodir = DATA_DIR / "depressionemo"
    for json_path in sorted(demodir.glob("*.json")):
        print(f"\n📁 depressionemo/{json_path.name}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  행 수: {len(data)}")
        if data:
            first = data[0]
            print(f"  키: {list(first.keys())}")
            # 감정 분포
            emotion_counts = {}
            for item in data:
                for emo in item.get("emotions", []):
                    emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
            if emotion_counts:
                sorted_emo = sorted(emotion_counts.items(), key=lambda x: -x[1])
                print(f"  감정 분포: {dict(sorted_emo)}")
            # 샘플 2개
            print(f"  샘플:")
            for item in data[:2]:
                txt = item.get("text", item.get("post", ""))[:100]
                emo = item.get("emotions", [])
                print(f"    {emo} → {txt}...")

    # C-SSRS
    cssrs_dir = DATA_DIR / "cssrs"
    for csv_path in sorted(cssrs_dir.glob("*.csv")):
        print(f"\n📁 cssrs/{csv_path.name}")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"  행 수: {len(rows)}")
        if rows:
            print(f"  컬럼: {list(rows[0].keys())}")
            # 레벨 분포
            levels = {}
            for r in rows:
                for col in r.keys():
                    if "level" in col.lower() or "score" in col.lower() or "cssrs" in col.lower():
                        lbl = r[col]
                        levels[lbl] = levels.get(lbl, 0) + 1
                        break
            if levels:
                print(f"  레벨 분포: {dict(sorted(levels.items()))}")
            # 샘플 2개
            print(f"  샘플:")
            for r in rows[:2]:
                txt_col = next((c for c in r.keys() if "text" in c.lower() or "post" in c.lower() or "title" in c.lower()), list(r.keys())[0])
                txt = r[txt_col][:100]
                print(f"    {txt}...")


if __name__ == "__main__":
    download_ourafla()
    download_depressionemo()
    download_cssrs()

    print("\n")
    explore_datasets()
