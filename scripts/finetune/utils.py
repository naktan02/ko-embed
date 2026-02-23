"""
scripts/finetune/utils.py
Phase 2 카테고리 매핑 유틸 함수.

로더(loaders.py)는 원본 데이터를 그대로 읽어오고,
카테고리 변환은 이 파일의 함수가 담당한다.

사용 예:
    from qcd.loaders import LOADERS
    from scripts.finetune.utils import apply_label_map, apply_emotion_map

    loader = LOADERS["ourafla"]()
    raw = loader.load(path)
    mapped = apply_label_map(raw, loader.LABEL_MAP, key="original_label")
"""

from __future__ import annotations


def apply_label_map(
    records: list[dict],
    label_map: dict[str, str],
    key: str = "original_label",
) -> list[dict]:
    """records의 key 값을 label_map으로 변환해 label 필드를 추가한다.

    매핑에 없는 값은 제외된다. (None → 제외)

    Args:
        records:   load()가 반환한 레코드 리스트
        label_map: {원본값: 카테고리} 딕셔너리
        key:       원본 레이블이 담긴 필드명 (기본값 "original_label")

    Returns:
        label 필드가 추가된 레코드 리스트 (매핑 불가 레코드 제외)
    """
    result: list[dict] = []
    for r in records:
        label = label_map.get(r.get(key, ""))
        if label:
            result.append({**r, "label": label})
    return result


def apply_emotion_map(
    records: list[dict],
    emotion_map: dict[str, str | None],
    priority: list[str],
) -> list[dict]:
    """DepressionEmo 멀티레이블 감정 리스트를 단일 카테고리로 변환한다.

    멀티레이블 충돌 시 priority 순서대로 우선 적용.
    모든 감정이 None(제외)이면 해당 레코드를 건너뜀.

    Args:
        records:     load()가 반환한 레코드 리스트 (emotions 필드 포함)
        emotion_map: {감정명: 카테고리 또는 None} 딕셔너리
        priority:    충돌 시 우선순위 카테고리 목록 (앞이 높은 우선순위)

    Returns:
        label 필드가 추가된 레코드 리스트
    """
    result: list[dict] = []
    for r in records:
        emotions: list[str] = r.get("emotions", [])
        cats = {emotion_map.get(e) for e in emotions} - {None}
        label = next((p for p in priority if p in cats), None)
        if label:
            result.append({**r, "label": label})
    return result
