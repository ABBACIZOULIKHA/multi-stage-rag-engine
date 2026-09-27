# data/

Drop your PDF files in this folder. `python ingest.py` parses every `*.pdf` here
(recursively) with LlamaParse and indexes the parsed content into Qdrant.

Subfolders are supported, e.g. `data/reports/q3-2026.pdf`.