import argparse
import sys

import config
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine

FALLBACK_ERRORS = ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError")


def build_index(llm):
    Settings.embed_model = config.get_embed_model()
    vector_store = config.get_vector_store()
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=Settings.embed_model
    )
    retriever = index.as_retriever(**config.get_retriever_kwargs())
    return RetrieverQueryEngine.from_args(
        retriever, llm=llm, node_postprocessors=[config.get_reranker()]
    )


def query_with_failover(question: str):
    models = config.llm_model_candidates()
    last_error = None

    for i, model in enumerate(models):
        llm = config.get_llm(model)
        Settings.llm = llm
        engine = build_index(llm)
        try:
            return engine.query(question)
        except Exception as exc:
            if not any(kind in type(exc).__name__ for kind in FALLBACK_ERRORS):
                raise
            last_error = exc
            if i + 1 < len(models):
                print(
                    f"[fallback] '{model}' failed ({type(exc).__name__}); "
                    f"trying '{models[i + 1]}' ..."
                )
            else:
                print(f"[fallback] '{model}' failed ({type(exc).__name__}); no models left.")

    raise last_error


def answer(question: str):
    response = query_with_failover(question)

    print("\n" + "=" * 64)
    print("ANSWER")
    print("=" * 64)
    print(response.response)

    print("\n" + "=" * 64)
    print(f"SOURCES ({len(response.source_nodes)})")
    print("=" * 64)
    for i, node in enumerate(response.source_nodes, 1):
        meta = node.node.metadata
        preview = node.node.get_content()[:300].replace("\n", " ")
        print(
            f"{i}. score={node.score:.4f} "
            f"src={meta.get('file_name', '?')} "
            f"page={meta.get('page_label', '?')}"
        )
        print(f"   {preview}")
        print()


def main():
    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]))
        return

    print("Interactive RAG. Type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in ("exit", "quit"):
            break
        if question.strip():
            answer(question)


if __name__ == "__main__":
    main()