# RAG Project

Retrieval-Augmented Generation built with **LlamaIndex**, **Qdrant**, **BGE-M3** embeddings,
**LlamaParse**, and **BGE reranking**. Retrieval is **hybrid**: dense BGE-M3 vectors fused with
a lexical **BM25** sparse leg inside Qdrant. The generation LLM is **Z.ai GLM 5.2 (free)**
(`z-ai/glm-5.2:free`) via OpenRouter's API.

## Pipeline

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
                Z.ai GLM 5.2 (free) LLM    openrouter.ai
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

| Flag | Meaning |
|------|---------|
| `--file NAME` | Shorthand for `file_name=NAME`. Repeatable. |
| `--filter EXPR` | Any payload field. Repeatable. Multiple filters are combined with **AND**. |
| `--no-default-filters` | Ignore `DEFAULT_FILTERS` from `.env` for this run. |

Expression syntax:

| Operator | Meaning |
|----------|---------|
| `key=value` | exact match |
| `key!=value` | not equal |
| `key~=value` | full-text contains |
| `key>value` `key>=` `key<` `key<=` | numeric range (value cast to float) |

Filterable fields are whatever ends up in the node payload. With the default pipeline that is:

| Field | Example value | Notes |
|-------|---------------|-------|
| `file_name` | `spec.pdf` | set from the PDF filename at ingest |
| `header_path` | `/Setup/Config/` | only with `CHUNKER=markdown`; note the leading **and trailing** `/` |
| `document_id` | `doc-abc123` | LlamaIndex-assigned |

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

| Variable            | Description                                              | Default |
|---------------------|----------------------------------------------------------|---------|
| `OPENROUTER_API_KEY` | Key from openrouter.ai/keys                            | – |
| `LLM_MODEL`         | Model on the OpenRouter API                              | `z-ai/glm-5.2:free` |
| `LLM_MAX_TOKENS`    | Max completion tokens                                    | `16384` |
| `LLM_CONTEXT_WINDOW`| Context window reported to LlamaIndex                    | `256000` |
| `LLAMA_CLOUD_API_KEY` | LlamaParse key                                        | – |
| `QDRANT_URL`        | Qdrant REST endpoint                                     | `http://localhost:6333` |
| `QDRANT_COLLECTION` | Collection name                                          | `documents` |
| `EMBED_MODEL`       | Local embedding model (sentence-transformers)            | `BAAI/bge-m3` |
| `RERANK_MODEL`      | Local reranker                                            | `BAAI/bge-reranker-v2-m3` |
| `CHUNKER`           | `markdown` (structure-aware) or `sentence`               | `markdown` |
| `CHUNK_SIZE`        | Token chunk size (`sentence` mode)                       | `512` |
| `CHUNK_OVERLAP`     | Overlap between chunks                                   | `50` |
| `TOP_K`             | Candidates fetched from each leg before fusion and rerank | `20` |
| `RERANK_TOP_N`      | Passages kept after reranking and sent to the LLM        | `3` |
| `LLM_TEMPERATURE`   | Sampling temperature                                     | `0.2` |
| `ENABLE_HYBRID`     | `0` restores dense-only retrieval                        | `1` |
| `SPARSE_MODEL`      | Sparse (lexical) leg run through `fastembed`             | `Qdrant/bm25` |
| `HYBRID_ALPHA`      | Fusion weight, `0` = pure BM25, `1` = pure dense         | `0.5` |
| `SPARSE_TOP_K`      | Candidates pulled from the sparse leg (defaults to `TOP_K`) | `20` |
| `FILTERED_FIELDS`   | Payload fields to index for metadata filtering            | `file_name,header_path,document_id` |
| `DEFAULT_FILTERS`   | Filters applied to every query (`;`-separated, AND-ed)   | – |

### How the two legs are fused

`QdrantVectorStore` issues both queries in one batched `query_batch_points` call — the same
embedding is reused for the dense leg, and the raw question string is tokenized for BM25.
Results are min-max normalized to `[0, 1]` per leg and combined as
`(1 - alpha) * sparse + alpha * dense`, so `alpha` is directly comparable across runs. The
fused top-K then goes through the BGE cross-encoder reranker unchanged.

Swap `SPARSE_MODEL` to `prithivida/Splade_PP_en_v1` for learned sparse expansion
(synonym-aware, English-only) — no code change needed.

### Picking a different LLM

The default is **Z.ai GLM 5.2 (free)** (`z-ai/glm-5.2:free`) via OpenRouter's
OpenAI-compatible API (`https://openrouter.ai/api/v1`). To change the model, set
`LLM_MODEL` in `.env` to any OpenRouter model slug — e.g. `meta-llama/llama-3.3-70b-instruct`,
`deepseek/deepseek-chat:free`, or any ID from https://openrouter.ai/models.
Free endpoints are rate-limited; check the model page for limits.

## Notes

- **BGE-M3** produces dense 1024-dim vectors; Qdrant auto-creates the collection with a named
  `text-dense` and `text-sparse` vector when hybrid is on.
- `get_vector_store()` is the single source of truth for the store config — `ingest.py` and
  `query.py` both go through it, so the two legs can never drift out of sync.
- **Offline option**: swap the OpenRouter provider for a local LLM (Ollama) later — only
  `config.get_llm()` needs to change.
- Reranker and embedding models run on GPU automatically (CUDA, or Intel XPU when available).

## Project layout

```
RAG_Project/
├── docker-compose.yml   # Qdrant
├── requirements.txt
├── .env.example         # copy to .env
├── config.py            # env + LLM / embedding / reranker / vector store factories
├── ingest.py            # Parse -> chunk -> embed dense+sparse -> index
├── query.py             # hybrid retrieve -> rerank -> generate
└── data/                # drop PDFs here
```