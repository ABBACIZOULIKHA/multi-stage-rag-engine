import os
from typing import Optional

import torch
from dotenv import load_dotenv
from qdrant_client.http import models as rest

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b:free")
LLM_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "LLM_FALLBACK_MODELS",
        "nvidia/nemotron-3-super-120b-a12b:free,"
        "google/gemma-4-31b-it:free,"
        "dots-studio/dots-3-note-preview:free,"
        "inclusionai/ling-3.0-flash-sante:free",
    ).split(",")
    if m.strip()
]
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "256000"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

CHUNKER = os.getenv("CHUNKER", "markdown")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "20"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))

FILTERED_FIELDS = [
    f.strip()
    for f in os.getenv(
        "FILTERED_FIELDS", "file_name,header_path,document_id"
    ).split(",")
    if f.strip()
]
DEFAULT_FILTERS = [
    f.strip()
    for f in os.getenv("DEFAULT_FILTERS", "").split(";")
    if f.strip()
]

ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))
SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", TOP_K))

DATA_DIR = os.getenv("DATA_DIR", "data")


DEVICE = os.getenv("DEVICE", "").strip()


def get_device() -> str:
    if DEVICE:
        return DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch, "xpu", None) and torch.xpu.is_available():
            return "xpu"
        return "cpu"
    except ImportError:
        return "cpu"


def get_vector_store(client=None, filters=None):
    store_class = get_filtered_store_class()

    if client is None:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_URL)

    kwargs = {}
    if ENABLE_HYBRID:
        kwargs = {"enable_hybrid": True, "fastembed_sparse_model": SPARSE_MODEL}

    store = store_class(
        client=client,
        collection_name=QDRANT_COLLECTION,
        payload_indexes=get_payload_indexes(),
        **kwargs,
    )
    if filters is not None:
        store.set_metadata_filter(filters)
    return store


def get_retriever_kwargs():
    if not ENABLE_HYBRID:
        return {"similarity_top_k": TOP_K}

    return {
        "similarity_top_k": TOP_K,
        "vector_store_query_mode": "hybrid",
        "sparse_top_k": SPARSE_TOP_K,
        "hybrid_top_k": TOP_K,
        "alpha": HYBRID_ALPHA,
    }


_FILTER_OPERATORS = {
    "!=": "ne",
    ">=": "gte",
    "<=": "lte",
    "~=": "text_match",
    "=": "eq",
    ">": "gt",
    "<": "lt",
}
_FILTER_SYMBOLS = {operator: symbol for symbol, operator in _FILTER_OPERATORS.items()}


def get_payload_indexes():
    return [
        {"field_name": field, "field_schema": rest.PayloadSchemaType.KEYWORD}
        for field in FILTERED_FIELDS
    ]


def parse_filter_expression(expression):
    for symbol, operator in _FILTER_OPERATORS.items():
        index = expression.find(symbol)
        if index > 0:
            key = expression[:index].strip()
            value = expression[index + len(symbol) :].strip()
            if key and value:
                return key, operator, value
    raise ValueError(
        f"Invalid filter '{expression}'. Use key=value, key!=value, key>=value, "
        "key<=value, key>value, key<value or key~=value."
    )


def resolve_filter_specs(files=None, expressions=None, include_defaults=True):
    specs = list(DEFAULT_FILTERS) if include_defaults else []
    specs.extend(expressions or [])
    specs.extend(f"file_name={name}" for name in files or [])
    return [parse_filter_expression(spec) for spec in specs]


def describe_filter_specs(files=None, expressions=None, include_defaults=True):
    return ", ".join(
        f"{key}{_FILTER_SYMBOLS[operator]}{value}"
        for key, operator, value in resolve_filter_specs(
            files, expressions, include_defaults
        )
    )


_FILTERED_STORE_CLASS = None


def get_filtered_store_class():
    global _FILTERED_STORE_CLASS

    if _FILTERED_STORE_CLASS is None:
        from llama_index.core.bridge.pydantic import PrivateAttr
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        class MetadataFilteredQdrantStore(QdrantVectorStore):
            _metadata_filter: Optional[rest.Filter] = PrivateAttr(default=None)

            def set_metadata_filter(self, qdrant_filter) -> None:
                self._metadata_filter = qdrant_filter

            def query(self, query, **kwargs):
                if self._metadata_filter is not None:
                    kwargs["qdrant_filters"] = self._metadata_filter
                return super().query(query, **kwargs)

        _FILTERED_STORE_CLASS = MetadataFilteredQdrantStore

    return _FILTERED_STORE_CLASS


def build_metadata_filter(files=None, expressions=None, include_defaults=True):
    conditions = []
    for key, operator, value in resolve_filter_specs(
        files, expressions, include_defaults
    ):
        if operator in ("gt", "gte", "lt", "lte"):
            conditions.append(
                rest.FieldCondition(
                    key=key, range=rest.Range(**{operator: float(value)})
                )
            )
        elif operator == "ne":
            conditions.append(
                rest.FieldCondition(
                    key=key, match=rest.MatchExcept(**{"except": [value]})
                )
            )
        elif operator == "text_match":
            conditions.append(
                rest.FieldCondition(key=key, match=rest.MatchText(text=value))
            )
        else:
            conditions.append(
                rest.FieldCondition(key=key, match=rest.MatchValue(value=value))
            )

    if not conditions:
        return None

    return rest.Filter(must=conditions)


def llm_model_candidates():
    seen = []
    for model in [LLM_MODEL, *LLM_FALLBACK_MODELS]:
        if model and model not in seen:
            seen.append(model)
    return seen


def get_llm(model=None):
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Create a key at "
            "https://openrouter.ai/keys and add "
            "OPENROUTER_API_KEY=... to your .env"
        )
    from llama_index.llms.openrouter import OpenRouter

    return OpenRouter(
        model=model or LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        context_window=LLM_CONTEXT_WINDOW,
        timeout=120,
        max_retries=5,
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "RAG Project"},
        additional_kwargs={
            "extra_body": {
                "provider": {
                    "sort": "throughput",
                    "allow_fallbacks": True,
                }
            }
        },
    )


def get_embed_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device=get_device(),
        embed_batch_size=4,
        model_kwargs={"torch_dtype": torch.float16},
    )


def get_reranker():
    from llama_index.core.postprocessor import SentenceTransformerRerank

    return SentenceTransformerRerank(
        model=RERANK_MODEL,
        top_n=RERANK_TOP_N,
        device=get_device(),
    )