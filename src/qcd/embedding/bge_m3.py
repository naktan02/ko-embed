"""BGE-M3 encoder (dense embeddings via FlagEmbedding / sentence-transformers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
# import yaml  <-- 이제 필요 없습니다.

from qcd.embedding.base import BaseEncoder


class BGEM3Encoder(BaseEncoder):
    """BAAI/bge-m3 를 사용하는 dense encoder."""

    def __init__(
        self, 
        model_id: str, 
        max_length: int = 8192, 
        batch_size: int = 32, 
        normalize_embeddings: bool = True,
        **kwargs  # 미래 확장성 및 Hydra의 유연한 인자 전달을 위해 유지
    ):
        """Hydra instantiate에 의해 인자들이 직접 주입됩니다."""
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize_embeddings
        self._model = None  # lazy load

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            # FlagEmbedding 우선 시도 (BGE-M3 full feature)
            from FlagEmbedding import BGEM3FlagModel  # type: ignore

            self._model = BGEM3FlagModel(
                self.model_id,
                use_fp16=False,
            )
            self._backend = "flag"
        except ImportError:
            # fallback: sentence-transformers
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_id)
            self._backend = "sbert"

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        self._load()
        if self._backend == "flag":
            out = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vecs = np.array(out["dense_vecs"], dtype=np.float32)
        else:
            vecs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=True,
            )
            vecs = np.array(vecs, dtype=np.float32)

        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-9, None)
        return vecs