"""
06_sts_eval.py
KLUE-STS ë²¤ì¹˜ë§ˆí�¬ë¡??„ë² ??ëª¨ë�¸???˜ë???? ì‚¬???±ëŠ¥???‰ê??©ë‹ˆ??

?‰ê? ë°©ì‹�:
  - KLUE-STS validation split (519?? ?¬ìš©
  - ê°?ë¬¸ìž¥?�ì�„ ?„ë² ????ì½”ì‚¬??? ì‚¬??ê³„ì‚°
  - ?¬ëžŒ??ë¶™ì�¸ ? ì‚¬???•ë‹µ(0~5??ê³?Spearman ?�ê?ê³„ìˆ˜ ë¹„êµ�
  - Spearman ?ê°€ 1??ê°€ê¹Œìš¸?˜ë¡� "?˜ë? ê°€ê¹Œìš´ ê±?ê°€ê¹�ê²Œ, ë¨?ê±?ë©€ê²? ??ë°°ì¹˜??ê²?

?¤í–‰ ?ˆì‹œ:
  uv run python scripts/explore/06_sts_eval.py
  uv run python scripts/explore/06_sts_eval.py model=bge_m3_ko
  uv run python scripts/explore/06_sts_eval.py model=bge_m3_korean

ëª¨ë“  ëª¨ë�¸ ?œë²ˆ??ë¹„êµ�:
  python scripts/explore/06_sts_eval.py  (model=bge_m3)
  python scripts/explore/06_sts_eval.py model=bge_m3_ko
  python scripts/explore/06_sts_eval.py model=bge_m3_korean
"""

from __future__ import annotations

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """?‰ë³„ ì½”ì‚¬??? ì‚¬?? a, b: (N, D)"""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a * b).sum(axis=1)  # (N,)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from datasets import load_dataset
    from scipy.stats import pearsonr, spearmanr

    model_name: str = cfg.model.get("name", "model")

    # ?€?€ KLUE-STS ë¡œë“œ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    print("[?°ì�´?? KLUE-STS validation ë¡œë“œ ì¤?..")
    ds = load_dataset("klue", "sts", split="validation")
    print(f"  {len(ds)}ê°?ë¬¸ìž¥??ë¡œë“œ ?„ë£Œ")

    # ë¬¸ìž¥??& ?•ë‹µ ? ì‚¬??ì¶”ì¶œ
    sents_a: list[str] = [ex["sentence1"] for ex in ds]
    sents_b: list[str] = [ex["sentence2"] for ex in ds]
    # ?•ë‹µ ?ˆì�´ë¸? real_label (0~5???¤ìˆ˜)
    gold_scores = np.array([ex["labels"]["real-label"] for ex in ds], dtype=np.float32)

    # ?€?€ ëª¨ë�¸ ë¡œë“œ & ?¸ì½”???€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    print(f"\n[ëª¨ë�¸] {model_name} ë¡œë“œ ì¤?..")
    encoder = instantiate(cfg.model)

    print(f"[?¸ì½”?? sentence1 ({len(sents_a)}ê°?...")
    vecs_a = encoder.encode(sents_a)  # (N, D)
    print(f"[?¸ì½”?? sentence2 ({len(sents_b)}ê°?...")
    vecs_b = encoder.encode(sents_b)  # (N, D)

    # ?€?€ ì½”ì‚¬??? ì‚¬??& ?�ê?ê³„ìˆ˜ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    pred_sims = cosine_sim(vecs_a, vecs_b)  # (N,)

    spearman_r, spearman_p = spearmanr(gold_scores, pred_sims)
    pearson_r,  pearson_p  = pearsonr(gold_scores,  pred_sims)

    # ?€?€ êµ¬ê°„ë³?ë¶„ì„� (??�Œ/ì¤‘ê°„/?’ì�Œ ? ì‚¬??êµ¬ê°„?�ì„œ ?¼ë§ˆ???˜í•˜?? ?€?€?€?€?€?€?€?€?€?€?€?€?€
    bins = [(0, 1, "??�Œ (0~1)"), (1, 3, "ì¤‘ê°„ (1~3)"), (3, 5, "?’ì�Œ (3~5)")]

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  KLUE-STS ?‰ê? ê²°ê³¼  |  ëª¨ë�¸: {model_name}")
    print(sep)
    print(f"  Spearman ?   : {spearman_r:.4f}  (p={spearman_p:.2e})")
    print(f"  Pearson  r   : {pearson_r:.4f}  (p={pearson_p:.2e})")
    print(f"\n  ? ?´ì„�:")
    print(f"    0.85??: ìµœê³  ?˜ì? (SOTA ?œêµ­??ëª¨ë�¸)")
    print(f"    0.80??: ?°ìˆ˜")
    print(f"    0.70??: ?‘í˜¸")
    print(f"    0.60??: ê°œì„  ?„ìš”")
    print(sep)

    print(f"\n  ? ì‚¬??êµ¬ê°„ë³?ë¶„ì„� (êµ¬ê°„ ??Spearman ?):")
    for lo, hi, label in bins:
        mask = (gold_scores >= lo) & (gold_scores < hi)
        n = int(mask.sum())
        if n >= 5:
            try:
                r, _ = spearmanr(gold_scores[mask], pred_sims[mask])
                print(f"    {label:12s}: ?={r:.3f}  (n={n})")
            except Exception:
                print(f"    {label:12s}: ê³„ì‚° ?¤íŒ¨ (n={n})")
        else:
            print(f"    {label:12s}: ?˜í”Œ ë¶€ì¡?(n={n})")

    # ?€?€ ?˜í”Œ ì¶œë ¥: ??ë§žì¶˜ ì¼€?´ìŠ¤ & ?€ë¦?ì¼€?´ìŠ¤ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    errors = np.abs(pred_sims - gold_scores / 5.0)  # ?•ê·œ?????ˆë??¤ì°¨

    print(f"\n  [ëª¨ë�¸????ë§žì¶˜ ?? (?¤ì°¨ ??�Œ top-5)")
    best_idx = np.argsort(errors)[:5]
    for idx in best_idx:
        print(f"    ?•ë‹µ={gold_scores[idx]:.1f}  ?ˆì¸¡sim={pred_sims[idx]:.3f}")
        print(f"    A: {sents_a[idx][:45]}")
        print(f"    B: {sents_b[idx][:45]}")
        print()

    print(f"  [ëª¨ë�¸???€ë¦??? (?¤ì°¨ ?’ì�Œ top-5)")
    worst_idx = np.argsort(errors)[-5:][::-1]
    for idx in worst_idx:
        print(f"    ?•ë‹µ={gold_scores[idx]:.1f}  ?ˆì¸¡sim={pred_sims[idx]:.3f}")
        print(f"    A: {sents_a[idx][:45]}")
        print(f"    B: {sents_b[idx][:45]}")
        print()


if __name__ == "__main__":
    main()
