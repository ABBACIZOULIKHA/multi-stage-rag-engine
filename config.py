import os

import torch
from dotenv import load_dotenv

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


def get_vector_store(client=None):
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    if client is None:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_URL)

    kwargs = {}
    if ENABLE_HYBRID:
        kwargs = {"enable_hybrid": True, "fastembed_sparse_model": SPARSE_MODEL}

    return QdrantVectorStore(
        client=client, collection_name=QDRANT_COLLECTION, **kwargs
    )


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