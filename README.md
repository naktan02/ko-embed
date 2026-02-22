# ko-embed

한국어 검색 쿼리 임베딩 파이프라인 — 청소년 검색 쿼리 변화 감지 연구

## 빠른 시작

```bash
uv sync
uv run python scripts/00_make_sample_data.py
uv run python scripts/01_embed.py
uv run python scripts/02_score.py
uv run python scripts/03_umap.py
```

## 구조

```
configs/     모델·실험 설정 YAML
data/        raw → interim → processed
artifacts/   임베딩, 시각화 결과
scripts/     단계별 실행 스크립트 (00~04)
src/qcd/     패키지 소스 (embedding / scoring / detection / viz)
tests/       연기 테스트
```
