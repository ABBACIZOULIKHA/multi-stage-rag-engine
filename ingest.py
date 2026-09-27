import argparse
import sys
from pathlib import Path

import torch
import config
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_parse import LlamaParse
from qdrant_client import QdrantClient

DATA_DIR = Path(config.DATA_DIR)


def get_splitter():
    if config.CHUNKER == "sentence":
        return SentenceSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
    return MarkdownNodeParser()


def parse_documents():
    if not config.LLAMA_CLOUD_API_KEY:
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is not set. Copy .env.example to .env and add your "
            "key from https://cloud.llamaindex.ai"
        )
    pdfs = [p for p in DATA_DIR.rglob("*.pdf")]
    if not pdfs:
        print(f"No PDFs found in '{DATA_DIR}'. Place your PDFs there and retry.")
        sys.exit(1)

    parser = LlamaParse(
        result_type="markdown",
        api_key=config.LLAMA_CLOUD_API_KEY,
        verbose=True,
    )
    documents = []
    for pdf in pdfs:
        print(f"Parsing '{pdf.name}' ...")
        documents.extend(
            parser.load_data(str(pdf), extra_info={"file_name": pdf.name})
        )
    print(f"Parsed {len(pdfs)} PDF(s) into {len(documents)} document(s).")
    return documents


def main():
    argp = argparse.ArgumentParser(description="Ingest PDFs into Qdrant via LlamaIndex")
    argp.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the Qdrant collection before ingesting",
    )
    args = argp.parse_args()

    Settings.embed_model = config.get_embed_model()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

    if args.reset:
        client = QdrantClient(url=config.QDRANT_URL)
        if client.collection_exists(config.QDRANT_COLLECTION):
            client.delete_collection(config.QDRANT_COLLECTION)
            print(f"Deleted collection '{config.QDRANT_COLLECTION}'.")
        else:
            print(f"Collection '{config.QDRANT_COLLECTION}' not present, creating it.")

    vector_store = config.get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    nodes = get_splitter().get_nodes_from_documents(parse_documents())
    print(f"Created {len(nodes)} nodes.")

    VectorStoreIndex(nodes, storage_context=storage_context)
    legs = (
        f"dense '{config.EMBED_MODEL}' + sparse '{config.SPARSE_MODEL}'"
        if config.ENABLE_HYBRID
        else f"dense '{config.EMBED_MODEL}'"
    )
    print(
        f"Done. {len(nodes)} nodes indexed with {legs} "
        f"into Qdrant collection '{config.QDRANT_COLLECTION}'."
    )


if __name__ == "__main__":
    main()