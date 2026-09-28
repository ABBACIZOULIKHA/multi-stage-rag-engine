import argparse
import sys

import config
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine

FALLBACK_ERRORS = ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError")


def build_index(llm, filters=None):
    Settings.embed_model = config.get_embed_model()
    vector_store = config.get_vector_store(filters=filters)
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=Settings.embed_model
    )
    retriever = index.as_retriever(**config.get_retriever_kwargs())
    return RetrieverQueryEngine.from_args(
        retriever, llm=llm, node_postprocessors=[config.get_reranker()]
    )


def query_with_failover(question: str, filters=None):
    models = config.llm_model_candidates()
    last_error = None

    for i, model in enumerate(models):
        llm = config.get_llm(model)
        Settings.llm = llm
        engine = build_index(llm, filters=filters)
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


def answer(question: str, filters=None):
    response = query_with_failover(question, filters=filters)

    print("\n" + "=" * 64)
    print("ANSWER")
    print("=" * 64)
    print(response.response)

    print("\n" + "=" * 64)
    print(f"SOURCES ({len(response.source_nodes)})")
    print("=" * 64)
    if not response.source_nodes:
        print("No passages matched. Check your metadata filters, or widen them.")
        return
    for i, node in enumerate(response.source_nodes, 1):
        meta = node.node.metadata
        preview = node.node.get_content()[:300].replace("\n", " ")
        print(
            f"{i}. score={node.score:.4f} "
            f"src={meta.get('file_name', '?')} "
            f"section={meta.get('header_path', '?')}"
        )
        print(f"   {preview}")
        print()


def main():
    argp = argparse.ArgumentParser(description="Query the RAG index")
    argp.add_argument("question", nargs="*", help="Question to ask")
    argp.add_argument(
        "--file",
        action="append",
        dest="files",
        metavar="NAME",
        help="Restrict retrieval to a source file (repeatable)",
    )
    argp.add_argument(
        "--filter",
        action="append",
        dest="filters",
        metavar="EXPR",
        help="Metadata filter, e.g. file_name=spec.pdf, header_path~=/Setup "
        "(repeatable)",
    )
    argp.add_argument(
        "--no-default-filters",
        action="store_true",
        help="Ignore DEFAULT_FILTERS from .env",
    )
    args = argp.parse_args()

    try:
        filters = config.build_metadata_filter(
            files=args.files,
            expressions=args.filters,
            include_defaults=not args.no_default_filters,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        sys.exit(2)

    if filters is not None:
        print(
            "Metadata filters (AND): "
            + config.describe_filter_specs(
                args.files, args.filters, not args.no_default_filters
            )
        )

    if args.question:
        answer(" ".join(args.question), filters=filters)
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
            answer(question, filters=filters)


if __name__ == "__main__":
    main()