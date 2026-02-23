"""
07_bench.py
mteb ?¼ì�´ë¸ŒëŸ¬ë¦¬ë¡œ ?œêµ­??Retrieval/STS ë²¤ì¹˜ë§ˆí�¬ë¥??¤í–‰?©ë‹ˆ??
SentenceTransformerë¥?ì§�ì ‘ mteb???„ë‹¬?˜ì—¬ ?¸í™˜??ë¬¸ì œë¥?ë°©ì??©ë‹ˆ??

?‰ê? ?œìŠ¤??
  [Retrieval]  nDCG@10 ê¸°ì? ??ì¿¼ë¦¬ ??ê´€??ë¬¸ì„œ ê²€???¥ë ¥
    - Ko-StrategyQA       : ?œêµ­??ë©€?°í™‰ QA (9,843ê°?
    - AutoRAGRetrieval    : ê¸ˆìœµ/ê³µê³µ/?˜ë£Œ/ë²•ë¥ /ì»¤ë¨¸??ë¬¸ì„œ ê²€??(834ê°?
    - PublicHealthQA      : ?˜ë£Œ/ê³µì¤‘ë³´ê±´ ë¬¸ì„œ ê²€??(1,774ê°?
    - XPQARetrieval       : ?¤ì–‘???„ë©”??ë¬¸ì„œ ê²€??(81,710ê°?

  [STS]  Spearman ? ê¸°ì? ??ë¬¸ìž¥ ???˜ë? ? ì‚¬??
    - KorSTS              : ?œêµ­??STS (KLUE-STS?€ ?¤ë¥¸ ì¶œì²˜, 1,376??

?œì™¸ (ì½”í�¼???ˆë¬´ ??:
  - MIRACLRetrieval (106M), MrTidyRetrieval (58M)
  - MultiLongDocRetrieval (497K), BelebeleRetrieval (522K)

?¤í–‰ ?ˆì‹œ:
  uv run python scripts/explore/07_bench.py model=bge_m3
  uv run python scripts/explore/07_bench.py model=bge_m3_ko
  uv run python scripts/explore/07_bench.py model=bge_m3_korean
  uv run python scripts/explore/07_bench.py model=kure_v1
"""

from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import mteb
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=DeprecationWarning)

TASKS = [
    # ?€?€ Retrieval (nDCG@10) ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    "Ko-StrategyQA",        # ë©€?°í™‰ QA ê²€?? corpus 9,843ê°?
    "AutoRAGRetrieval",     # 5ê°??„ë©”??ë¬¸ì„œ ê²€?? corpus 834ê°?
    "PublicHealthQA",       # ?˜ë£Œ/ê³µì¤‘ë³´ê±´, corpus 1,774ê°?
    "XPQARetrieval",        # ?¤ì–‘???„ë©”?? corpus 81,710ê°?
    # ?€?€ STS (Spearman ?) ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    "KorSTS",               # ?œêµ­??STS, 1,376??
]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import pandas as pd

    model_name: str = cfg.model.get("name", "model")
    model_id: str = cfg.model.get("model_id", "")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    mteb_out = out_dir / f"mteb_{model_name}"

    print(f"\n[ëª¨ë�¸] {model_name}  ({model_id}) ë¡œë“œ ì¤?..")
    # SentenceTransformerë¥?ì§�ì ‘ ?¬ìš© ??mtebê³??„ì „ ?¸í™˜
    st_model = SentenceTransformer(model_id, trust_remote_code=True)
    print(f"  ?„ë² ??ì°¨ì›�: {st_model.get_sentence_embedding_dimension()}")

    # ?€?€ ?œìŠ¤?¬ë³„ ?¤í–‰ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    summary: list[dict] = []

    for task_name in TASKS:
        print(f"\n{'='*55}")
        print(f"  ?œìŠ¤?? {task_name}")
        print(f"{'='*55}")
        try:
            task_list = mteb.get_tasks(tasks=[task_name], languages=["kor"])
            if not task_list:
                print(f"  [?¤í‚µ] ?œìŠ¤?¬ë? ì°¾ì�„ ???†ì�Œ: {task_name}")
                continue

            evaluation = mteb.MTEB(tasks=task_list)
            results = evaluation.run(
                st_model,
                output_folder=str(mteb_out),
                overwrite_results=False,
                verbosity=1,
            )

            # ì£¼ìš” ì§€??ì¶”ì¶œ
            for res in results:
                for split in ("test", "dev", "validation"):
                    if split in res.scores and res.scores[split]:
                        s = res.scores[split][0]
                        main_score = s.get("main_score", None)
                        if main_score is not None:
                            ndcg = s.get("ndcg_at_10", None)
                            r10  = s.get("recall_at_10", None)
                            spearman = s.get("cos_sim", {}).get("spearman", None) if isinstance(s.get("cos_sim"), dict) else None
                            print(f"  ??[{split}] main_score={main_score:.4f}", end="")
                            if ndcg:  print(f"  nDCG@10={ndcg:.4f}", end="")
                            if r10:   print(f"  R@10={r10:.4f}",      end="")
                            print()
                            summary.append({
                                "task":        task_name,
                                "type":        task_list[0].metadata.type,
                                "split":       split,
                                "main_score":  round(main_score, 4),
                                "ndcg_at_10":  round(ndcg, 4) if ndcg else None,
                                "recall_at_10": round(r10, 4) if r10 else None,
                            })
                            break

        except Exception as e:
            print(f"  [?¤ë¥˜] {e}")
            summary.append({"task": task_name, "error": str(e)[:120]})

    # ?€?€ ê²°ê³¼ ?€???€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    df = pd.DataFrame(summary)
    csv_path = out_dir / f"mteb_{model_name}_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  MTEB ê²°ê³¼  |  ëª¨ë�¸: {model_name}")
    print(sep)
    for row in summary:
        if "main_score" in row:
            extra = f"  nDCG@10={row['ndcg_at_10']}" if row.get("ndcg_at_10") else ""
            print(f"  {row['task']:<30} {row['main_score']:.4f}{extra}")
        else:
            print(f"  {row['task']:<30} ERROR  {row.get('error','')[:50]}")
    print(sep)
    print(f"\n[?€?? {csv_path}")


if __name__ == "__main__":
    main()
