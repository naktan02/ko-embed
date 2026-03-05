"""BaseEncoder – abstract contract for all embedding models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseEncoder(ABC):
    """모든 임베딩 모델이 구현해야 하는 인터페이스."""

    @abstractmethod
    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """텍스트 리스트를 임베딩 행렬로 변환.

        Args:
            texts: 인코딩할 문자열 리스트
            **kwargs: 모델별 추가 인자

        Returns:
            shape (N, D) numpy 배열, L2-정규화 완료
        """

    def encode_file(self, path: Path, **kwargs) -> np.ndarray:
        """텍스트 파일(한 줄 = 한 쿼리)을 읽고 임베딩."""
        texts = Path(path).read_text(encoding="utf-8").splitlines()
        texts = [t.strip() for t in texts if t.strip()]
        return self.encode(texts, **kwargs)
