"""
src/qcd/loaders.py
데이터셋별 로더 모음.

새 데이터셋을 추가할 때:
  1. 이 파일에 클래스 하나 추가
     - LABEL_MAP  : {원본레이블: 정규화레이블} 딕셔너리
     - SOURCE     : 출처 태그 문자열
     - load()     : Path → list[dict] 구현
  2. scripts/explore/00_preprocess.py의 LOADERS 딕셔너리에 한 줄 추가

카테고리 목록(CATEGORIES)은 LABEL_MAP 값에서 자동 생성됩니다.
→ scoring.categories를 별도 config에 중복 관리할 필요 없음.
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path


class BaseLoader(ABC):
    """모든 로더의 공통 인터페이스.

    서브클래스는 반드시 LABEL_MAP과 SOURCE를 정의해야 합니다.
    CATEGORIES는 LABEL_MAP 값에서 자동 생성됩니다.
    """

    LABEL_MAP: dict[str, str] = {}  # 서브클래스에서 정의
    SOURCE: str = ""               # 서브클래스에서 정의

    @classmethod
    @property
    def CATEGORIES(cls) -> list[str]:
        """LABEL_MAP 값에서 중복 제거 후 삽입 순서 유지한 카테고리 목록."""
        return list(dict.fromkeys(cls.LABEL_MAP.values()))

    @abstractmethod
    def load(self, path: Path) -> list[dict]:
        """파일을 읽어 표준 레코드 리스트로 반환.

        각 레코드는 최소한 다음 키를 포함해야 합니다:
          - text   (str)  : 임베딩할 텍스트
          - label  (str)  : 정규화된 카테고리 레이블
          - source (str)  : 데이터 출처 태그
        """


class TalksetsLoader(BaseLoader):
    """AIHub 비윤리 발화 talksets JSON 로더.

    JSON 구조:
      [ { "id": "...", "sentences": [
            { "text": "...", "types": ["HATE", ...],
              "is_immoral": true, "intensity": 2.0 }
          ] } ]

    types가 복수 레이블인 경우 첫 번째를 primary label로 사용.
    원본 types 목록과 is_immoral, intensity도 함께 저장.
    """

    LABEL_MAP: dict[str, str] = {
        "DISCRIMINATION": "discrimination",
        "HATE": "hate",
        "CENSURE": "censure",
        "VIOLENCE": "violence",
        "CRIME": "crime",
        "SEXUAL": "sexual",
        "ABUSE": "abuse",
        "IMMORAL_NONE": "neutral",
    }
    SOURCE = "talksets"

    def load(self, path: Path) -> list[dict]:
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  JSON 파싱 중... ({size_mb:.1f} MB)", flush=True)

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        rows: list[dict] = []
        skipped_label = 0
        skipped_empty = 0

        for conversation in data:
            for sent in conversation.get("sentences", []):
                text = sent.get("text", "").strip()
                if not text:
                    skipped_empty += 1
                    continue

                types: list[str] = sent.get("types", [])
                if not types:
                    skipped_label += 1
                    continue

                label = self.LABEL_MAP.get(types[0])
                if label is None:
                    skipped_label += 1
                    continue

                rows.append({
                    "text": text,
                    "label": label,
                    "types": types,
                    "is_immoral": sent.get("is_immoral", False),
                    "intensity": sent.get("intensity", 0.0),
                    "source": self.SOURCE,
                })

        if skipped_label:
            print(f"  [경고] 알 수 없는 레이블로 건너뜀: {skipped_label}개", file=sys.stderr)
        if skipped_empty:
            print(f"  [경고] 빈 텍스트 건너뜀: {skipped_empty}개", file=sys.stderr)

        return rows


# ── 로더 레지스트리 ────────────────────────────────────────────────────────────
# 새 데이터셋 추가 시: 로더 클래스를 위에 작성 후 여기에 한 줄만 추가
LOADERS: dict[str, type[BaseLoader]] = {
    "talksets": TalksetsLoader,
    # "aihub558": Aihub558Loader,
}


# ── 향후 추가 예시 ──────────────────────────────────────────────────────────
# class Aihub558Loader(BaseLoader):
#     LABEL_MAP = {"위기": "distress", "자해": "self_harm", ...}
#     SOURCE = "aihub558"
#     CATEGORIES = list(dict.fromkeys(LABEL_MAP.values()))  # 자동 생성
#
#     def load(self, path: Path) -> list[dict]:
#         import pandas as pd
#         df = pd.read_excel(path)
#         rows = []
#         for _, row in df.iterrows():
#             label = self.LABEL_MAP.get(str(row["분류"]).strip())
#             if label:
#                 rows.append({"text": str(row["검색어"]).strip(),
#                              "label": label, "source": self.SOURCE})
#         return rows
