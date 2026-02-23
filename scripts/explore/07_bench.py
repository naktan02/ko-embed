"""
07_bench.py
mteb 라이브러리로 한국어 Retrieval/STS 벤치마크를 실행합니다.
SentenceTransformer를 직접 mteb에 전달하여 인환성 문제를 방지합니다.

평가 태스크:
  [Retrieval]  nDCG@10 기준 — 쿼리 관련 문서 검색 능력
    - Ko-StrategyQA       : 한국어 멀티홉 QA (corpus 9,843개)
    - AutoRAGRetrieval    : 금융/공공/의료/법률/커머스 문서 검색 (corpus 834개)
    - PublicHealthQA      : 의료/공중보건 문서 검색 (corpus 1,774개)
    - XPQARetrieval       : 다양한 도메인 문서 검색 (corpus 81,710개)

  [STS]  Spearman ρ 기준 — 문장 간 의미 유사도
    - KorSTS              : 한국어 STS (KLUE-STS와 다른 출처, 1,376쌍)

제외 (코퍼스 너무 큼):
  - MIRACLRetrieval (106M), MrTidyRetrieval (58M)
  - MultiLongDocRetrieval (497K), BelebeleRetrieval (522K)

실행 예시:
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
    # ── Retrieval (nDCG@10) ───────────────────────────────────────────────────
    "Ko-StrategyQA",        # 멀티홉 QA 검색, corpus 9,843개
    "AutoRAGRetrieval",     # 5개 도메인 문서 검색, corpus 834개
    "PublicHealthQA",       # 의료/공중보건, corpus 1,774개
    "XPQARetrieval",        # 다양한 도메인, corpus 81,710개
    # ── STS (Spearman ρ) ──────────────────────────────────────────────────────
    "KorSTS",               # 한국어 STS, 1,376쌍
]


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import pandas as pd

    model_name: str = cfg.model.get("name", "model")
    model_id: str = cfg.model.get("model_id", "")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    mteb_out = out_dir / f"mteb_{model_name}"

    print(f"\n[모델] {model_name}  ({model_id}) 로드 중..")
    # SentenceTransformer를 직접 사용 → mteb과 완전 호환
    st_model = SentenceTransformer(model_id, trust_remote_code=True)
    print(f"  임베딩 차원: {st_model.get_sentence_embedding_dimension()}")

    # ── 태스크별 실행 ──────────────────────────────────────────────────────────
    summary: list[dict] = []

    for task_name in TASKS:
        print(f"\n{'='*55}")
        print(f"  태스크: {task_name}")
        print(f"{'='*55}")
        try:
            task_list = mteb.get_tasks(tasks=[task_name], languages=["kor"])
            if not task_list:
                print(f"  [스킵] 태스크를 찾을 수 없음: {task_name}")
                continue

            evaluation = mteb.MTEB(tasks=task_list)
            results = evaluation.run(
                st_model,
                output_folder=str(mteb_out),
                overwrite_results=False,
                verbosity=1,
            )

            # 주요 지표 추출
            for res in results:
                for split in ("test", "dev", "validation"):
                    if split in res.scores and res.scores[split]:
                        s = res.scores[split][0]
                        main_score = s.get("main_score", None)
                        if main_score is not None:
                            ndcg = s.get("ndcg_at_10", None)
                            r10  = s.get("recall_at_10", None)
                            spearman = s.get("cos_sim", {}).get("spearman", None) if isinstance(s.get("cos_sim"), dict) else None
                            print(f"  ✓ [{split}] main_score={main_score:.4f}", end="")
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
            print(f"  [오류] {e}")
            summary.append({"task": task_name, "error": str(e)[:120]})

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(summary)
    csv_path = out_dir / f"mteb_{model_name}_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  MTEB 결과  |  모델: {model_name}")
    print(sep)
    for row in summary:
        if "main_score" in row:
            extra = f"  nDCG@10={row['ndcg_at_10']}" if row.get("ndcg_at_10") else ""
            print(f"  {row['task']:<30} {row['main_score']:.4f}{extra}")
        else:
            print(f"  {row['task']:<30} ERROR  {row.get('error','')[:50]}")
    print(sep)
    print(f"\n[저장] {csv_path}")


if __name__ == "__main__":
    main()
