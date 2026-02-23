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

    def get_label(self, row: dict) -> str:
        """샘플링·표시용 라벨 문자열 반환.

        _print_stats의 라벨별 그룹화에 사용됩니다.
        서브클래스에서 오버라이드하여 데이터셋별 필드를 반환하세요.
        """
        return row.get("label", row.get("source", "?"))


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


# ── 파인튜닝용 데이터셋 로더 ──────────────────────────────────────────────────

class OuraflaLoader(BaseLoader):
    """ourafla 4-Class Mental Health 데이터셋 로더.

    load() → 원본 레이블 그대로 반환 (raw I/O)
    Phase 2에서 apply_label_map(records, LABEL_MAP)으로 카테고리 변환

    CSV 컬럼: text, status
    원본 레이블: Suicidal / Depression / Anxiety / Normal
    """

    # Phase 2 참조용 매핑 — load()에서는 사용하지 않음
    LABEL_MAP: dict[str, str] = {
        "Suicidal":   "suicidal",
        "Depression": "depression",
        "Anxiety":    "anxiety",
        "Normal":     "normal",
    }
    SOURCE = "ourafla"

    def load(self, path: Path) -> list[dict]:
        """원본 status 값을 original_label로 그대로 저장."""
        import csv

        rows: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = row.get("text", "").strip()
                original_label = row.get("status", "").strip()
                if text and original_label:
                    rows.append({
                        "text":           text,
                        "original_label": original_label,
                        "source":         self.SOURCE,
                    })
        return rows

    def get_label(self, row: dict) -> str:
        return row.get("original_label", "?")


class DepressionEmoLoader(BaseLoader):
    """DepressionEmo 8감정 멀티레이블 로더.

    load() → emotions 리스트를 그대로 반환 (raw I/O)
    Phase 2에서 apply_emotion_map(records, EMOTION_MAP, PRIORITY)으로 카테고리 변환

    JSON 컬럼: text, emotions (list)
    """

    # 감정 → 카테고리 (None = 제외)
    # ⚠️ 모두 임시값 — 데이터 샘플 확인 후 Phase 2에서 확정
    EMOTION_MAP: dict[str, str | None] = {
        # suicide intent는 이견 없이 suicidal
        "suicide intent":             "suicidal",

        # 아래는 탐색 후 결정: hopelessness는 자살 위험 요인이기도 함
        # "hopelessness": "suicidal",  # 대안: 자살 생각과 연결성 높음
        "hopelessness":               "depression",

        "emptiness":                  "depression",
        "worthlessness":              "depression",
        "sadness":                    "depression",
        "loneliness":                 "depression",
        "brain dysfunction (forget)": "depression",

        # anger: 세 가지 대안 중 데이터 보고 결정
        # "anger": "anxiety",    # 대안 1: 공격적 긴장 → anxiety
        # "anger": "depression", # 대안 2: 우울의 일환
        "anger":                      None,  # 현재: 제외
    }
    # 멀티레이블 충돌 시 우선순위 — Phase 2에서 apply_emotion_map()이 사용
    PRIORITY: list[str] = ["suicidal", "depression"]

    # LABEL_MAP은 BaseLoader 인터페이스 호환용 (카테고리 목록 참조)
    LABEL_MAP: dict[str, str] = {v: v for v in EMOTION_MAP.values() if v}
    SOURCE = "depressionemo"

    def load(self, path: Path) -> list[dict]:
        """emotions 리스트를 그대로 저장 — 카테고리 매핑 없음."""
        data = json.load(open(path, encoding="utf-8"))
        rows: list[dict] = []
        for item in data:
            text = item.get("text", "").strip()
            emotions: list[str] = item.get("emotions", [])
            if text and emotions:
                rows.append({
                    "text":    text,
                    "emotions": emotions,
                    "source":  self.SOURCE,
                })
        return rows

    def get_label(self, row: dict) -> str:
        # 그룹화 키: 감정 조합 (정렬해서 순서 무관하게)
        return str(sorted(row.get("emotions", [])))


class CSSRSLoader(BaseLoader):
    """C-SSRS 7단계 레이블 Reddit SuicideWatch 로더.

    load() → 원본 cssrs_level(0~6)을 그대로 반환 (raw I/O)
    Phase 2에서 데이터 확인 후 LABEL_MAP 채우고 apply_label_map() 사용

    CSV 컬럼: 실제 컬럼명은 탐색 후 확인 (자동 탐지)
    """

    # Phase 2에서 데이터 확인 후 아래 매핑을 검토·수정하여 주석 해제할 것
    # LABEL_MAP: dict[str, str] = {
    #     "0": "normal",
    #     "1": "depression",   # 수동적 자살 생각, 비특이적
    #     "2": "depression",   # 수동적 자살 생각, 비특이적
    #     "3": "suicidal",     # 방법 있음
    #     "4": "suicidal",     # 계획 있음
    #     "5": "suicidal",     # 의도 있음
    #     "6": "suicidal",     # 시도
    # }
    LABEL_MAP: dict[str, str] = {}  # 탐색 전 — 비워둠
    SOURCE = "cssrs"

    # 레벨/텍스트 컬럼 탐지 우선순위 키워드
    _LEVEL_KEYWORDS = ["cssrs", "level", "score", "label", "class"]
    _TEXT_KEYWORDS  = ["text", "post", "body", "content", "title"]

    def _detect_col(self, headers: list[str], keywords: list[str]) -> str | None:
        for kw in keywords:
            for h in headers:
                if kw in h.lower():
                    return h
        return headers[0] if headers else None

    def load(self, path: Path) -> list[dict]:
        """Phase 1: 원본 레벨 그대로 반환 (카테고리 매핑 없음)."""
        import csv

        rows: list[dict] = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            level_col = self._detect_col(headers, self._LEVEL_KEYWORDS)
            text_col  = self._detect_col(headers, self._TEXT_KEYWORDS)

            for row in reader:
                text = row.get(text_col or "", "").strip()
                level_raw = str(row.get(level_col or "", "")).strip()
                if text and level_raw:
                    rows.append({
                        "text":        text,
                        "cssrs_level": level_raw,  # 원본 레벨 그대로
                        "source":      self.SOURCE,
                    })
        return rows

    def get_label(self, row: dict) -> str:
        return str(row.get("cssrs_level", "?"))


# ── 로더 레지스트리 ────────────────────────────────────────────────────────────
# 새 데이터셋 추가 시: 로더 클래스를 위에 작성 후 여기에 한 줄만 추가
LOADERS: dict[str, type[BaseLoader]] = {
    "talksets":      TalksetsLoader,
    "ourafla":       OuraflaLoader,
    "depressionemo": DepressionEmoLoader,
    "cssrs":         CSSRSLoader,
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
