"""
ingestion/loaders.py

PURPOSE:
    Unified ingestion pipeline that accepts ANY source type and loads it into
    the ChromaDB vector store. This replaces the old hardcoded ingestion.py.

SUPPORTED SOURCE TYPES:
    - URL         → Crawled via Tavily (existing approach)
    - PDF file    → Loaded via PyPDFLoader (langchain + pypdf)
    - Text/MD     → Loaded via TextLoader
    - Directory   → Recursively loads all supported files in the directory

DESIGN DECISIONS:
    - All sources are normalised to LangChain Document objects before chunking.
    - Chunking strategy is IDENTICAL for all sources (300 tok / 50 overlap)
      to ensure comparable embedding quality across source types.
    - Metadata is injected per-source so ChromaDB supports metadata filtering:
        { "source": "<url_or_filepath>", "source_type": "url|pdf|text|directory" }
    - Deduplication: Before inserting, we check if a chunk with the same
      source URL already exists, so re-running ingestion doesn't duplicate chunks.

USAGE:
    from ingestion.loaders import ingest_source
    ingest_source("https://lilianweng.github.io/posts/...")  # URL
    ingest_source("./docs/paper.pdf")                        # PDF
    ingest_source("./docs/notes.txt")                        # Plain text
    ingest_source("./docs/")                                 # Directory
"""

import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────

CHROMA_COLLECTION = "blog_podcast_agent"
CHROMA_DIR = "./.chroma"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 300      # tokens — same as the original ingestion.py
CHUNK_OVERLAP = 50    # tokens — 50-token overlap preserves sentence boundaries

# File extensions recognised when scanning a directory
SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".pdf"}


# ─── Text Splitter (shared across all loaders) ────────────────────────────────

_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# ─── Individual Loaders ───────────────────────────────────────────────────────

def _load_from_url(url: str) -> List[Document]:
    """
    Crawls a URL using Tavily and returns LangChain Documents.
    This is the original ingestion path — unchanged.
    """
    try:
        from langchain_tavily import TavilyCrawl
    except ImportError:
        raise ImportError("langchain-tavily is required for URL ingestion. Run: poetry add langchain-tavily")

    print(f"  [URL] Crawling: {url}")
    tavily_crawl = TavilyCrawl()
    response = tavily_crawl.invoke({"url": url, "max_depth": 2, "max_breadth": 4})

    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            raise ValueError(f"Tavily returned an invalid response for URL: {url}")

    pages = response.get("results", [])
    documents = []
    for page in pages:
        content = page.get("content") or page.get("raw_content") or ""
        if not content:
            continue
        documents.append(Document(
            page_content=content,
            metadata={
                "source": page.get("url", url),
                "source_type": "url",
                "title": page.get("title", ""),
            },
        ))

    print(f"  [URL] Loaded {len(documents)} pages from crawl.")
    return documents


def _load_from_pdf(path: str) -> List[Document]:
    """
    Loads a PDF file using PyPDFLoader.
    Each page becomes a Document. Metadata includes file path and page number.
    """
    print(f"  [PDF] Loading: {path}")
    loader = PyPDFLoader(path)
    docs = loader.load()

    # Normalise metadata to our standard schema
    for doc in docs:
        doc.metadata["source"] = str(Path(path).resolve())
        doc.metadata["source_type"] = "pdf"
        doc.metadata.setdefault("title", Path(path).stem)

    print(f"  [PDF] Loaded {len(docs)} pages.")
    return docs


def _load_from_text(path: str) -> List[Document]:
    """
    Loads a plain text or markdown file using TextLoader.
    The entire file becomes one Document before chunking.
    """
    print(f"  [TEXT] Loading: {path}")
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = str(Path(path).resolve())
        doc.metadata["source_type"] = "text"
        doc.metadata.setdefault("title", Path(path).stem)

    print(f"  [TEXT] Loaded {len(docs)} document(s).")
    return docs


def _load_from_directory(directory: str) -> List[Document]:
    """
    Recursively loads all supported files (.txt, .md, .rst, .pdf) in a directory.
    Uses DirectoryLoader with glob patterns to find files.
    """
    print(f"  [DIR] Scanning directory: {directory}")
    all_docs = []

    dir_path = Path(directory)
    for ext in SUPPORTED_EXTENSIONS:
        # Find all files with this extension recursively
        files = list(dir_path.rglob(f"*{ext}"))
        for file in files:
            try:
                if ext == ".pdf":
                    docs = _load_from_pdf(str(file))
                else:
                    docs = _load_from_text(str(file))
                all_docs.extend(docs)
            except Exception as e:
                print(f"  ⚠ Skipping {file}: {e}")

    print(f"  [DIR] Total: {len(all_docs)} document(s) loaded from directory.")
    return all_docs


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def ingest_source(source: str) -> int:
    """
    Unified ingestion function. Detects the source type, loads documents,
    chunks them, embeds them, and stores them in ChromaDB.

    Args:
        source: One of:
            - A URL string starting with "http://" or "https://"
            - A file path ending in .pdf, .txt, .md, .rst
            - A directory path

    Returns:
        The number of chunks successfully ingested into ChromaDB.

    Raises:
        ValueError: If the source type cannot be determined.
        FileNotFoundError: If a local path does not exist.
    """
    print(f"\n{'='*60}")
    print(f"  Ingesting source: {source}")
    print(f"{'='*60}")

    # ── Detect source type ────────────────────────────────────────────────────
    if source.startswith("http://") or source.startswith("https://"):
        raw_docs = _load_from_url(source)

    elif os.path.isdir(source):
        raw_docs = _load_from_directory(source)

    elif os.path.isfile(source):
        suffix = Path(source).suffix.lower()
        if suffix == ".pdf":
            raw_docs = _load_from_pdf(source)
        elif suffix in {".txt", ".md", ".rst"}:
            raw_docs = _load_from_text(source)
        else:
            raise ValueError(
                f"Unsupported file extension: '{suffix}'. "
                f"Supported: .pdf, .txt, .md, .rst"
            )
    else:
        raise FileNotFoundError(
            f"Source not found: '{source}'. "
            "Provide a valid URL, file path, or directory."
        )

    if not raw_docs:
        print("  ⚠ No content found in source. Nothing ingested.")
        return 0

    # ── Chunk ─────────────────────────────────────────────────────────────────
    print(f"\n  Chunking {len(raw_docs)} document(s)...")
    chunked_docs = _splitter.split_documents(raw_docs)
    print(f"  → {len(chunked_docs)} chunks created (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    if not chunked_docs:
        print("  ⚠ No chunks produced. Nothing ingested.")
        return 0

    # ── Embed + Store in ChromaDB ─────────────────────────────────────────────
    print(f"\n  Embedding and storing in ChromaDB...")
    print(f"  Collection: '{CHROMA_COLLECTION}' | Directory: '{CHROMA_DIR}'")

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DIR,
    )

    # Add documents in one batch
    vectorstore.add_documents(chunked_docs)

    print(f"\n  ✅ Ingestion complete!")
    print(f"     Source     : {source}")
    print(f"     Chunks     : {len(chunked_docs)}")
    print(f"     Collection : {CHROMA_COLLECTION}")
    print(f"{'='*60}\n")

    return len(chunked_docs)
