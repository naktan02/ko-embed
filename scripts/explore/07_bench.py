"""
07_bench.py
mteb 라이브러리로 영어 Retrieval/STS 벤치마크를 실행합니다.

태스크 그룹:
  [retrieval]  nDCG@10 수치 기준 — 쿼리 관련 문서 검색 능력
    - SciFact             : 과학 논문 팩트 체크, corpus 5,183개
    - NFCorpus            : 의학/영양학 문서 검색
    - FiQA2018            : 금융 QA 검색
    - TRECCOVID           : 의료/COVID 문서 검색 (도메인 유사성 ↑)

  [sts]  Spearman ρ 기준 — 문장 간 의미 유사도 (파인튜닝 후 회귀 체크용)
    - STSBenchmark        : 영어 STS 표준 (8,628쌍)
    - SICK-R              : 문장 유사도 평가 (10,000쌍)

테스트 모델:
  model=bge_large_en     : BAAI/bge-large-en-v1.5  (335M)
  model=mxbai_large      : mixedbread-ai/mxbai-embed-large-v1 (335M)
  model=bge_base_en      : BAAI/bge-base-en-v1.5  (110M)
  model=bge_m3           : BAAI/bge-m3  (568M, 다국어 baseline)
  model=mpnet_base       : sentence-transformers/all-mpnet-base-v2 (110M)

실행 예:
  # 전체 태스크
  python scripts/explore/07_bench.py model=bge_large_en

  # STS만 (파인튜닝 후 회귀 체크)
  python scripts/explore/07_bench.py model=bge_large_en +group=sts

  # Retrieval만
  python scripts/explore/07_bench.py model=bge_large_en +group=retrieval

  # 파인튜닝된 모델 로컬 경로 지정
  python scripts/explore/07_bench.py model=bge_large_en model.model_id=models/finetuned/bge_large_en +group=sts
"""

from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import mteb
import torch
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=DeprecationWarning)

# 태스크 그룹 정의 — `+group=<태스크>` 오버라이드로 선택 실행
TASK_GROUPS: dict[str, list[str]] = {
    "retrieval": [
        "SciFact",              # 과학 논문 팩트 체크, corpus 5,183개
        "NFCorpus",             # 의학/영양학 문서 검색
        "FiQA2018",             # 금융 QA 검색
        "TRECCOVID",            # 의료/COVID 문서 검색 (도메인 유사성 ↑)
    ],
    "sts": [
        "STSBenchmark",         # 영어 STS 표준, 8,628쌍
        "SICK-R",               # 문장 유사도 평가, 10,000쌍
    ],
}
TASK_GROUPS["all"] = TASK_GROUPS["retrieval"] + TASK_GROUPS["sts"]


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import pandas as pd

    model_name: str = cfg.model.get("name", "model")
    model_id: str = cfg.model.get("model_id", "")
    group: str = cfg.get("group", "all")          # +group=sts | retrieval | all
    tasks = TASK_GROUPS.get(group, TASK_GROUPS["all"])

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    mteb_out = out_dir / f"mteb_{model_name}"

    batch_size: int = cfg.model.get("batch_size", 16)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[모델] {model_name}  ({model_id}) 로드 중..")
    print(f"[그룹] {group}  →  {tasks}")
    print(f"[디바이스] {device}  |  batch_size={batch_size}")
    # SentenceTransformer를 직접 사용 → mteb과 완전 호환
    st_model = SentenceTransformer(model_id, trust_remote_code=True, device=device)
    print(f"  임베딩 차원: {st_model.get_sentence_embedding_dimension()}")

    # ── 태스크별 실행 ──────────────────────────────────────────────────────────
    summary: list[dict] = []

    for task_name in tasks:
        print(f"\n{'='*55}")
        print(f"  태스크: {task_name}")
        print(f"{'='*55}")
        try:
            task_list = mteb.get_tasks(tasks=[task_name], languages=["eng"])
            if not task_list:
                print(f"  [스킵] 태스크를 찾을 수 없음: {task_name}")
                continue

            evaluation = mteb.MTEB(tasks=task_list)
            results = evaluation.run(
                st_model,
                output_folder=str(mteb_out),
                overwrite_results=False,
                verbosity=1,
                encode_kwargs={"batch_size": batch_size},
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
