# RAG Project

Retrieval-Augmented Generation built with **LlamaIndex**, **Qdrant**, **BGE-M3** embeddings,
**LlamaParse**, and **BGE reranking**. Retrieval is **hybrid**: dense BGE-M3 vectors fused with
a lexical **BM25** sparse leg inside Qdrant. The generation LLM is **NVIDIA Nemotron 3 Super 120B (free)**
(`nvidia/nemotron-3-super-120b-a12b:free`) via OpenRouter's API.

## Architecture

The system is three independently runnable stages that share **one configuration source**
(`config.py`) and **one Qdrant collection**, so ingest and query can never drift out of sync.

| Stage                        | Entry point     | Responsibility                                                          |
| ---------------------------- | --------------- | ----------------------------------------------------------------------- |
| **Ingest** (offline)   | `ingest.py`   | Parse PDFs → chunk → embed dense + sparse → upsert into Qdrant       |
| **Query** (online)     | `query.py`    | Hybrid retrieve → fuse → rerank → generate grounded answer + sources |
| **Evaluate** (offline) | `evaluate.py` | Run the query pipeline over a testset → score with RAGAS + abstention  |

### Components

| Layer            | Component                             | Technology                          | Role                                                          |
| ---------------- | ------------------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| Parsing          | LlamaParse                            | cloud.llamaindex.ai                 | PDF → structured markdown                                    |
| Chunking         | MarkdownNodeParser / SentenceSplitter | LlamaIndex                          | structure-aware (or fixed-size) chunks                        |
| Dense embedding  | BGE-M3                                | sentence-transformers (local)       | 1024-dim semantic vectors                                     |
| Sparse embedding | Qdrant/bm25                           | fastembed (local, ONNX)             | lexical term weights                                          |
| Vector store     | Qdrant                                | Docker (`docker-compose.yml`)     | one collection,`text-dense` + `text-sparse` named vectors |
| Retrieval        | Hybrid search                         | LlamaIndex`QdrantVectorStore`     | dense + sparse queried together, alpha-weighted fusion        |
| Reranking        | BGE-reranker-v2-m3                    | sentence-transformers cross-encoder | reorder fused candidates, keep top-N                          |
| Generation       | Nemotron 3 Super 120B                 | OpenRouter (or Google)              | answer grounded in the top-N passages                         |
| Configuration    | `config.py`                         | python-dotenv                       | env vars + LLM / embedding / reranker / store factories       |
| Evaluation       | RAGAS 0.4.3 + Ollama judge            | `evaluate.py`                     | faithfulness, relevancy, precision, recall + abstention       |

Dense and sparse embedders, the reranker, and the vector store are all created by factories in
`config.py`; `get_vector_store()` is the single source of truth for the store config, and
`get_llm()` is the only provider-aware function.

### Data flow

```
                    ┌───────────────────────────────┐
  PDFs ────────────▶│  LlamaParse (markdown output) │   cloud.llamaindex.ai
                    └──────────────┬────────────────┘
                                   ▼
              MarkdownNodeParser / SentenceSplitter   (chunking)
                                   ▼
       ┌───────────────┬───────────┴───────────┐
       ▼               ▼                       ▼
 BGE-M3 dense    Qdrant/bm25 sparse      Qdrant (Docker)
 1024-dim, local  lexical, CPU/ONNX       one collection,
       │               │                  two named vectors
       │               │                       ▲
       │        Question ─── BGE-M3 + BM25 ─────┤
       │               │                       │
       ▼               ▼                       │
   dense top-K     sparse top-K                │
       └───────▶ relative score fusion ◀───────┘
                   (alpha-weighted)
                          ▼
                   fused top-K
                          ▼
           BGE-reranker-v2-m3 (cross-encoder)
                          ▼
                 Nemotron 3 Super 120B LLM    openrouter.ai
                          ▼
                   Answer + sources
```

Dense retrieval handles paraphrase and conceptual similarity; the BM25 leg catches the exact
tokens dense vectors routinely miss — part numbers, error codes, proper names, acronyms,
citations. `HYBRID_ALPHA=1.0` collapses back to pure dense, `0.0` to pure BM25.

## Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Qdrant)
- API keys:
  - **OpenRouter** (LLM): https://openrouter.ai/keys → Create key
  - **LlamaParse**: https://cloud.llamaindex.ai → API keys

## Setup

```powershell
# 1. Create venv and install deps (torch + sentence-transformers are large)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Start Qdrant (keep Docker Desktop running)
docker compose up -d

# 3. Configure secrets
Copy-Item .env.example .env
# then edit .env: OPENROUTER_API_KEY (and LLAMA_CLOUD_API_KEY)
```

## Ingest

Put PDFs in `data/`, then:

```powershell
python ingest.py            # adds to existing collection
python ingest.py --reset    # wipe collection first (re-index)
```

First run downloads `BAAI/bge-m3` (~2.5 GB) locally — models are cached on disk afterwards.
`Qdrant/bm25` is tokenizer-only (~10 MB, ships with `fastembed`), so the sparse leg adds no
model download and no GPU work.

> Collections indexed **before** hybrid was enabled have no sparse vector configured, so the
> sparse leg cannot be added to them retroactively — re-index with `python ingest.py --reset`.

## Query

```powershell
python query.py "What does the document say about ...?"
```

or run interactively:

```powershell
python query.py
```

Each answer prints its retrieved sources with rerank scores, file name, section, and a preview.

## Metadata filtering

Filters are applied as a Qdrant `Filter` **before** retrieval, so they apply to the dense
*and* the sparse leg, not after fusion. Both legs are filtered, then fused, then reranked.

```powershell
python query.py --file spec.pdf "What is the retry policy?"
python query.py --filter "header_path~=/Setup" "How do I configure the device?"
python query.py --file a.pdf --filter "page>3" "What changed in 2024?"
```

| Flag                     | Meaning                                                                         |
| ------------------------ | ------------------------------------------------------------------------------- |
| `--file NAME`          | Shorthand for`file_name=NAME`. Repeatable.                                    |
| `--filter EXPR`        | Any payload field. Repeatable. Multiple filters are combined with**AND**. |
| `--no-default-filters` | Ignore`DEFAULT_FILTERS` from `.env` for this run.                           |

Expression syntax:

| Operator                                   | Meaning                             |
| ------------------------------------------ | ----------------------------------- |
| `key=value`                              | exact match                         |
| `key!=value`                             | not equal                           |
| `key~=value`                             | full-text contains                  |
| `key>value` `key>=` `key<` `key<=` | numeric range (value cast to float) |

Filterable fields are whatever ends up in the node payload. With the default pipeline that is:

| Field           | Example value      | Notes                                                                        |
| --------------- | ------------------ | ---------------------------------------------------------------------------- |
| `file_name`   | `spec.pdf`       | set from the PDF filename at ingest                                          |
| `header_path` | `/Setup/Config/` | only with`CHUNKER=markdown`; note the leading **and trailing** `/` |
| `document_id` | `doc-abc123`     | LlamaIndex-assigned                                                          |

`header_path` carries surrounding slashes, so match it exactly — `--filter "header_path=/Setup/"`
— or use `~=` to match on the text alone: `--filter "header_path~=/Setup"`.

Add fields to `FILTERED_FIELDS` in `.env` to get a Qdrant payload index for them. Indexes are
created on startup and are safe to add to an existing collection; filters still work without
them, but Qdrant falls back to scanning the collection.

Set persistent filters in `.env`:

```
DEFAULT_FILTERS=file_name=spec.pdf;header_path~=/Setup
```

Entries are `;`-separated and AND-ed together.

## Configuration (`.env`)

| Variable                | Description                                                      | Default                                                           |
| ----------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------- |
| `OPENROUTER_API_KEY`  | Key from openrouter.ai/keys (for`LLM_PROVIDER=openrouter`)     | –                                                                |
| `GEMINI_API_KEY`      | Key from aistudio.google.com/apikey (for`LLM_PROVIDER=google`) | –                                                                |
| `LLM_PROVIDER`        | Generator backend:`openrouter` or `google`                   | `openrouter`                                                    |
| `LLM_MODEL`           | Generator model id for the selected provider                     | `nvidia/nemotron-3-super-120b-a12b:free`                        |
| `LLM_FALLBACK_MODELS` | Models tried in order when the generator fails                   | –                                                                |
| `LLM_MAX_TOKENS`      | Max completion tokens                                            | `16384`                                                         |
| `LLM_CONTEXT_WINDOW`  | Context window reported to LlamaIndex                            | `256000`                                                        |
| `LLAMA_CLOUD_API_KEY` | LlamaParse key                                                   | –                                                                |
| `QDRANT_URL`          | Qdrant REST endpoint                                             | `http://localhost:6333`                                         |
| `QDRANT_COLLECTION`   | Collection name                                                  | `documents`                                                     |
| `EMBED_MODEL`         | Local embedding model (sentence-transformers)                    | `BAAI/bge-m3`                                                   |
| `RERANK_MODEL`        | Local reranker                                                   | `BAAI/bge-reranker-v2-m3`                                       |
| `CHUNKER`             | `markdown` (structure-aware) or `sentence`                   | `markdown`                                                      |
| `CHUNK_SIZE`          | Token chunk size (`sentence` mode)                             | `512`                                                           |
| `CHUNK_OVERLAP`       | Overlap between chunks                                           | `50`                                                            |
| `TOP_K`               | Candidates fetched from each leg before fusion and rerank        | `20`                                                            |
| `RERANK_TOP_N`        | Passages kept after reranking and sent to the LLM                | `3`                                                             |
| `LLM_TEMPERATURE`     | Sampling temperature                                             | `0.2`                                                           |
| `ENABLE_HYBRID`       | `0` restores dense-only retrieval                              | `1`                                                             |
| `SPARSE_MODEL`        | Sparse (lexical) leg run through`fastembed`                    | `Qdrant/bm25`                                                   |
| `HYBRID_ALPHA`        | Fusion weight,`0` = pure BM25, `1` = pure dense              | `0.5`                                                           |
| `SPARSE_TOP_K`        | Candidates pulled from the sparse leg (defaults to`TOP_K`)     | `20`                                                            |
| `FILTERED_FIELDS`     | Payload fields to index for metadata filtering                   | `file_name,header_path,document_id`                             |
| `DEFAULT_FILTERS`     | Filters applied to every query (`;`-separated, AND-ed)         | –                                                                |
| `EVAL_JUDGE_PROVIDER` | Judge backend:`ollama` or `openrouter`                       | `ollama`                                                        |
| `EVAL_JUDGE_MODEL`    | Judge model id on the selected provider                          | `gpt-oss:120b-cloud`                                            |
| `EVAL_METRICS`        | Which RAGAS metrics to compute                                   | faithfulness, answer_relevancy, context_precision, context_recall |
| `EVAL_TESTSET_FILE`   | Testset path                                                     | `Ragas_testset_with_sources.json`                               |

### How the two legs are fused

`QdrantVectorStore` issues both queries in one batched `query_batch_points` call — the same
embedding is reused for the dense leg, and the raw question string is tokenized for BM25.
Results are min-max normalized to `[0, 1]` per leg and combined as
`(1 - alpha) * sparse + alpha * dense`, so `alpha` is directly comparable across runs. The
fused top-K then goes through the BGE cross-encoder reranker unchanged.

Swap `SPARSE_MODEL` to `prithivida/Splade_PP_en_v1` for learned sparse expansion
(synonym-aware, English-only) — no code change needed.

### Picking a different LLM

`LLM_PROVIDER` in `.env` selects the backend:

| Provider                 | Needs                                                         | Model ids                                                                                          |
| ------------------------ | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `openrouter` (default) | `OPENROUTER_API_KEY`                                        | any OpenRouter slug:`nvidia/nemotron-3-super-120b-a12b:free`, `google/gemma-4-31b-it:free`, … |
| `google`               | `GEMINI_API_KEY` (free, https://aistudio.google.com/apikey) | `gemma-4-31b-it`, `gemma-4-26b-a4b-it`, `gemini-3.8-flash`, `gemini-3.5-flash-lite`        |

`google` uses Gemini's OpenAI-compatible endpoint, so no extra package is needed. When the
primary model fails, `LLM_FALLBACK_MODELS` (comma-separated, ids must match the provider) is
tried in order. Avoid `gemini-2.5-*` — Google retires them on 2026-10-20.

## Evaluate

`evaluate.py` scores the pipeline against `Ragas_testset_with_sources.json` — a RAGAS-format
testset of **59 questions** (51 answerable, 8 unanswerable) with gold contexts, extracted from
the same four source PDFs.

```bash
python evaluate.py                       # full run (59 items)
python evaluate.py --id q001 --id q002   # subset, useful for smoke tests
python evaluate.py --run-dir eval_runs\baseline
```

**Generator** = your configured `LLM_MODEL`. **Judge** = `EVAL_JUDGE_PROVIDER`
(`ollama` by default, local `gpt-oss:120b-cloud`, free and unlimited) or `openrouter`.

Metrics — computed only on the **51 answerable** items:

| Metric                | Measures                                          |
| --------------------- | ------------------------------------------------- |
| `faithfulness`      | the answer is supported by the retrieved contexts |
| `answer_relevancy`  | the answer addresses the question                 |
| `context_precision` | the retrieved contexts are ranked correctly       |
| `context_recall`    | the gold contexts were retrieved                  |

Plus a custom **abstention** measure on the 8 unanswerable items (RAGAS has no equivalent):
`ABSTAINED` vs `ANSWERED` judged from the answer alone.

Artifacts written to `--run-dir`:

- `rag_outputs.json` — generated answer + contexts per question (**resumed on re-run**)
- `ragas_scores.json`, `abstention.json` — raw per-item scores (judge responses are cached on disk in `.ragas_cache`, so re-scoring is free)
- `per_item.csv` — one row per question, for spreadsheets
- `summary.json` — aggregate metrics + run configuration

RAGAS is called directly per metric (`metric.ascore`) rather than through `ragas.evaluate()`:
`evaluate()` skips the newer `ragas.metrics.collections` classes on 0.4.3. The dependency stack
is therefore pinned — see the comment in `requirements.txt`.

## Evaluation results

Baseline run `eval_runs/baseline` — generator `nvidia/nemotron-3-super-120b-a12b:free`,
judge `Ollama:gpt-oss:120b-cloud`, hybrid retrieval (`alpha=0.5`, `top_k=20`, `rerank_top_n=3`).
50 of 59 items were generated before the OpenRouter **free-tier daily cap (50 req/day)** was hit;
the remaining 9 resume on re-run. RAGAS scored 43 answerable items; abstention judged 50.

| Metric                           |  Score |  n |
| -------------------------------- | -----: | -: |
| faithfulness                     | 0.9050 | 43 |
| answer_relevancy                 | 0.7657 | 43 |
| context_precision                | 0.8527 | 43 |
| context_recall                   | 0.8837 | 43 |
| abstains on unanswerable         | 1.0000 |  7 |
| answers when it can (answerable) | 0.8837 | 43 |

**Reading:** retrieval is strong (recall **0.88**, precision **0.85**) and answers are well
grounded (faithfulness **0.91**); the model reliably abstains on unanswerable items (**7/7**). The
weakest metric is **answer_relevancy (0.77)** — some answers are grounded but off-target. The
main failure cluster is multi-document questions and a handful of items where the wrong source
document was retrieved (e.g. `q019` expected two documents, retrieved neither). Highest-leverage
next experiments: raise `RERANK_TOP_N` (3 → 5) for multi-document queries, tune `HYBRID_ALPHA`,
and use metadata `DEFAULT_FILTERS` to steer retrieval to the right document.

> These are single-run, partial numbers (43/51 answerable scored). Re-run the command below after
> the daily quota resets to complete the baseline; finished items and judge calls are cached.

## Notes

- **BGE-M3** produces dense 1024-dim vectors; Qdrant auto-creates the collection with a named
  `text-dense` and `text-sparse` vector when hybrid is on.
- `get_vector_store()` is the single source of truth for the store config — `ingest.py` and
  `query.py` both go through it, so the two legs can never drift out of sync.
- **Multiple LLM backends**: `LLM_PROVIDER` switches between OpenRouter and Google without
  code changes — only `config.get_llm()` is provider-aware.
- Reranker and embedding models run on GPU automatically (CUDA, or Intel XPU when available).

## Project layout

```
RAG_Project/
├── docker-compose.yml   # Qdrant
├── requirements.txt
├── .env    
├── config.py            # env + LLM / embedding / reranker / vector store factories
├── ingest.py            # Parse -> chunk -> embed dense+sparse -> index
├── query.py             # hybrid retrieve -> rerank -> generate
├── evaluate.py          # RAGAS eval over Ragas_testset_with_sources.json
├── Ragas_testset_with_sources.json   # 59 questions with gold contexts
├── eval_runs/           # per-run artifacts (rag_outputs, scores, csv, summary)
└── data/                # drop PDFs here
```
