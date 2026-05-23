"""
src/qcd/translation/nllb.py
============================
facebook/nllb-200-distilled-600M 기반 번역기.
transformers pipeline API 대신 model/tokenizer 직접 로드 방식 사용.

사용 예:
    from qcd.translation.nllb import NLLBTranslator
    tr = NLLBTranslator()          # 기본: ko → en (GPU 자동)
    print(tr.translate("죽고 싶어"))
    print(tr.translate_batch(["불안해", "우울해"]))
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class NLLBTranslator:
    """NLLB-200 기반 단방향 번역기 (기본: 한국어 → 영어)."""

    LANG_KO = "kor_Hang"
    LANG_EN = "eng_Latn"

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "kor_Hang",
        tgt_lang: str = "eng_Latn",
        max_length: int = 512,
    ) -> None:
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._max_length = max_length
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"  [NLLB] {model_id} 로드 중 ({src_lang} → {tgt_lang}, device={self._device})...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang=src_lang)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self._device)
        self._model.eval()

        # 번역 대상 언어의 토큰 ID
        self._tgt_lang_id = self._tokenizer.convert_tokens_to_ids(tgt_lang)
        print("  [NLLB] 로드 완료")

    def translate(self, text: str) -> str:
        """단일 문장 번역."""
        return self.translate_batch([text])[0]

    def translate_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[str]:
        """배치 번역. 순서 보존."""
        results: list[str] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            ).to(self._device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tgt_lang_id,
                    max_length=self._max_length,
                )

            decoded = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(decoded)

        return results
