"""
ingestion/__init__.py

Exposes the unified ingestion interface for the Blog-to-Podcast Agent.

Usage:
    from ingestion import ingest_source
    ingest_source("https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/")
    ingest_source("./docs/paper.pdf")
    ingest_source("./docs/notes.txt")
    ingest_source("./docs/")  # entire directory
"""

from ingestion.loaders import ingest_source

__all__ = ["ingest_source"]
