# Cross-Lingual Semantic Embedding Benchmark

Research pipeline for evaluating Korean, English, and multilingual sentence embedding models across retrieval, semantic textual similarity, NLI, and query representation tasks.

This repository compares multilingual, Korean-specialized, and English sentence embedding models, analyzes their behavior on Korean and English benchmark datasets, and provides reusable scripts for preprocessing, embedding, scoring, visualization, and model comparison.

## Project Motivation

Semantic search and retrieval systems depend heavily on the quality of sentence embeddings. However, English, multilingual, and Korean-specialized models often behave differently across retrieval, STS, NLI, and downstream classification-style tasks.

This project investigates:

- how well embedding models preserve semantic similarity across Korean and English datasets;
- whether retrieval-oriented models transfer to Korean query understanding and cross-lingual evaluation settings;
- which models are suitable as baselines for later fine-tuning;
- how embedding spaces can be visualized and analyzed for semantic structure.

## Key Features

- End-to-end embedding evaluation pipeline using `uv`, Hydra configs, and reusable Python modules.
- Model wrappers for SentenceTransformers and BGE-style embedding models.
- Benchmark scripts for Korean and English retrieval, STS, and NLI-style evaluation.
- Prototype-based scoring and baseline comparison utilities.
- UMAP and semantic visualization scripts for embedding-space analysis.
- Reproducible result summaries under `results/`.

## Evaluated Models

The project currently includes configurations for Korean-specialized, multilingual, and English-focused models:

- `BAAI/bge-m3`
- `dragonkue/BGE-m3-ko`
- `upskyy/bge-m3-korean`
- `nlpai-lab/KURE-v1`
- `Qwen/Qwen3-Embedding-0.6B`
- `sentence-transformers/all-mpnet-base-v2`
- `sentence-transformers/stsb-roberta-base-v2`
- `mixedbread-ai/mxbai-embed-large-v1`
- `BAAI/bge-base-en-v1.5`
- `BAAI/bge-large-en-v1.5`

## Benchmark Summary

### Korean Retrieval

Retrieval tasks are evaluated mainly with `nDCG@10`.

| Model | Ko-StrategyQA | AutoRAG | PublicHealthQA | XPQA | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BAAI/bge-m3` | 0.7941 | 0.8301 | 0.8041 | 0.3609 | 0.6973 |
| `dragonkue/BGE-m3-ko` | 0.7955 | **0.8738** | 0.8155 | 0.3814 | 0.7166 |
| `upskyy/bge-m3-korean` | 0.7527 | 0.7285 | 0.7756 | 0.3166 | 0.6434 |
| `nlpai-lab/KURE-v1` | **0.7998** | 0.8708 | **0.8193** | **0.3815** | **0.7179** |

### Semantic Textual Similarity

STS tasks are evaluated with Spearman correlation.

| Model | KLUE-STS | KorSTS |
| --- | ---: | ---: |
| `BAAI/bge-m3` | 0.8773 | 0.8026 |
| `dragonkue/BGE-m3-ko` | **0.8869** | 0.8159 |
| `upskyy/bge-m3-korean` | 0.8782 | **0.8334** |
| `nlpai-lab/KURE-v1` | - | 0.8125 |

### Prototype-Based Query Scoring

Prototype scoring was tested on a Korean query dataset with 70,593 samples.

| Model | Accuracy | Macro F1 | Weighted F1 | Silhouette |
| --- | ---: | ---: | ---: | ---: |
| `BAAI/bge-m3` | 38.6% | 0.247 | 0.440 | -0.022 |
| `dragonkue/BGE-m3-ko` | **39.4%** | **0.253** | **0.447** | **-0.022** |
| `upskyy/bge-m3-korean` | 34.9% | 0.222 | 0.403 | -0.032 |

These results suggest that pretrained embedding models are strong for semantic similarity and retrieval, but prototype-based category separation still requires task-specific fine-tuning.

## Repository Structure

```text
configs/            Hydra configuration files for data and model settings
data/               Raw, intermediate, and processed datasets
artifacts/          Generated embeddings, plots, and intermediate outputs
results/            Benchmark summaries and evaluation outputs
scripts/
  explore/          Preprocessing, embedding, scoring, visualization, and search scripts
  eval/             STS, NLI, retrieval, and prototype evaluation scripts
  finetune/         Dataset construction and fine-tuning scripts
src/qcd/            Reusable package code for embedding, scoring, detection, IO, and visualization
tests/              Smoke tests and basic validation
```

## Quick Start

Install dependencies:

```bash
uv sync
```

Run the main exploration pipeline:

```bash
uv run python scripts/explore/00_preprocess.py
uv run python scripts/explore/01_embed.py
uv run python scripts/explore/02_score.py
uv run python scripts/explore/03_umap.py
```

Run model comparison:

```bash
uv run python scripts/explore/04_compare_models.py
```

Run benchmark evaluation:

```bash
uv run python scripts/explore/07_bench.py
```

## Configuration

Models and datasets are controlled through YAML files:

```text
configs/config.yaml
configs/model/*.yaml
configs/data/*.yaml
```

Example model override:

```bash
uv run python scripts/explore/01_embed.py model=bge_m3_ko
```

## Research Notes

Current findings:

- `nlpai-lab/KURE-v1` and `dragonkue/BGE-m3-ko` are competitive for Korean retrieval.
- `upskyy/bge-m3-korean` performs strongly on KorSTS.
- General pretrained embeddings do not cleanly separate custom query categories without fine-tuning.
- Fine-tuning with task-specific contrastive data is the next step for improving category-level separation.

## Recommended Repository Metadata

Recommended repository name:

```text
cross-lingual-semantic-embedding-benchmark
```

Recommended GitHub description:

```text
Benchmarking and analysis pipeline for Korean, English, and multilingual sentence embeddings across retrieval, STS, NLI, and query understanding tasks.
```
