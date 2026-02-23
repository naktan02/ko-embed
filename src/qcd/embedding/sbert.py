"""sentence-transformers 기반 범용 인코더."""

from __future__ import annotations

import numpy as np

from qcd.embedding.base import BaseEncoder


class SentenceTransformerEncoder(BaseEncoder):
    """sentence-transformers 라이브러리로 모든 HuggingFace 모델을 로드하는 인코더.

    KURE-v1, SRoBERTa 처럼 FlagEmbedding이 필요 없는 모델에 사용합니다.
    """

    def __init__(
        self,
        model_id: str,
        max_length: int = 512,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize_embeddings
        self._model = None  # lazy load

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(self.model_id)
        self._model.max_seq_length = self.max_length
        print(f"  [SentenceTransformer] {self.model_id} 로드 완료")

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        self._load()
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        return np.array(vecs, dtype=np.float32)
