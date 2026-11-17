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
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────

CHROMA_COLLECTION = "blog_podcast_agent_v2"  # v2: cosine distance (was L2 in v1)
CHROMA_DIR = "./.chroma"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 300  # tokens — same as the original ingestion.py
CHUNK_OVERLAP = 50  # tokens — 50-token overlap preserves sentence boundaries

import logging

import chromadb as _chromadb

_logger = logging.getLogger(__name__)

# ChromaDB client for the dedup pre-flight check (shared, thread-safe)
_chroma_dedup_client = _chromadb.PersistentClient(path=CHROMA_DIR)

# File extensions recognised when scanning a directory
SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".pdf"}


# ─── Text Splitter (shared across all loaders) ────────────────────────────────

_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# ─── Individual Loaders ───────────────────────────────────────────────────────


def _load_from_url(url: str, user_id: str = "default_user") -> List[Document]:
    """
    Crawls a URL using Tavily and returns LangChain Documents.
    This is the original ingestion path — unchanged.
    """
    try:
        from langchain_tavily import TavilyCrawl
    except ImportError:
        raise ImportError(
            "langchain-tavily is required for URL ingestion. Run: poetry add langchain-tavily"
        )

    print(f"  [URL] Crawling: {url}")
    tavily_crawl = TavilyCrawl()
    response = tavily_crawl.invoke({"url": url, "max_depth": 2, "max_breadth": 4})

    if isinstance(response, str):
        if not response.strip():
            raise ValueError(
                f"The URL could not be crawled — Tavily returned an empty response. "
                f"This usually means the page blocks automated crawlers, requires JavaScript to load, "
                f"or is temporarily unavailable. Please try a different URL or upload the page as a PDF. "
                f"(URL: {url})"
            )
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            # If it's just plain text, treat it as the content itself
            if "error" in response.lower()[:50]:
                raise ValueError(f"Tavily returned an error for URL {url}: {response[:100]}")
            response = {"results": [{"url": url, "content": response}]}

    pages = response.get("results", [])
    documents = []
    for page in pages:
        content = page.get("content") or page.get("raw_content") or ""
        if not content:
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": page.get("url", url),
                    "source_type": "url",
                    "title": page.get("title", ""),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "user_id": user_id,
                },
            )
        )

    if not documents:
        raise ValueError(
            f"The URL was crawled successfully but no readable text content was found. "
            f"This can happen with JavaScript-heavy pages, paywalled content, login-gated pages, "
            f"or pages that only contain images and media. "
            f"Please try a different URL or save the page as a PDF and upload it instead. "
            f"(URL: {url})"
        )

    print(f"  [URL] Loaded {len(documents)} pages from crawl.")
    return documents


def _load_from_pdf(path: str, user_id: str = "default_user") -> List[Document]:
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
        doc.metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()
        doc.metadata["user_id"] = user_id

    print(f"  [PDF] Loaded {len(docs)} pages.")
    return docs


def _load_from_text(path: str, user_id: str = "default_user") -> List[Document]:
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
        doc.metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()
        doc.metadata["user_id"] = user_id

    print(f"  [TEXT] Loaded {len(docs)} document(s).")
    return docs


def _load_from_directory(directory: str, user_id: str = "default_user") -> List[Document]:
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
                    docs = _load_from_pdf(str(file), user_id=user_id)
                else:
                    docs = _load_from_text(str(file), user_id=user_id)
                all_docs.extend(docs)
            except Exception as e:
                print(f"  [WARNING] Skipping {file}: {e}")

    print(f"  [DIR] Total: {len(all_docs)} document(s) loaded from directory.")
    return all_docs


# ─── Main Entry Point ─────────────────────────────────────────────────────────


def ingest_source(source: str, user_id: str = "default_user") -> int:
    """
    Unified ingestion function. Detects the source type, loads documents,
    chunks them, embeds them, and stores them in ChromaDB.

    Args:
        source: One of:
            - A URL string starting with "http://" or "https://"
            - A file path ending in .pdf, .txt, .md, .rst
            - A directory path
        user_id: The unique ID of the user owning these documents.

    Returns:
        The number of chunks successfully ingested into ChromaDB.

    Raises:
        ValueError: If the source type cannot be determined.
        FileNotFoundError: If a local path does not exist.
    """
    print(f"\n{'=' * 60}")
    print(f"  Ingesting source: {source} for user_id: {user_id}")
    print(f"{'=' * 60}")

    # ── Detect source type ────────────────────────────────────────────────────
    if source.startswith("http://") or source.startswith("https://"):
        raw_docs = _load_from_url(source, user_id=user_id)

    elif os.path.isdir(source):
        raw_docs = _load_from_directory(source, user_id=user_id)

    elif os.path.isfile(source):
        suffix = Path(source).suffix.lower()
        if suffix == ".pdf":
            raw_docs = _load_from_pdf(source, user_id=user_id)
        elif suffix in {".txt", ".md", ".rst"}:
            raw_docs = _load_from_text(source, user_id=user_id)
        else:
            raise ValueError(
                f"Unsupported file extension: '{suffix}'. Supported: .pdf, .txt, .md, .rst"
            )
    else:
        raise FileNotFoundError(
            f"Source not found: '{source}'. Provide a valid URL, file path, or directory."
        )

    if not raw_docs:
        # This is a failure, not a silent success — raise so the route returns HTTP 422.
        raise ValueError(
            "No content could be extracted from the provided source. "
            "If this is a URL, the page may block crawlers, require JavaScript, or be behind a paywall. "
            "Try uploading the content as a PDF or plain-text file instead."
        )

    # ── Chunk ─────────────────────────────────────────────────────────────────
    print(f"\n  Chunking {len(raw_docs)} document(s)...")
    chunked_docs = _splitter.split_documents(raw_docs)
    print(f"  -> {len(chunked_docs)} chunks created (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    if not chunked_docs:
        print("  [WARNING] No chunks produced. Nothing ingested.")
        return 0

    # ── Embed + Store in ChromaDB ─────────────────────────────────────────────
    print("\n  Embedding and storing in ChromaDB...")
    print(f"  Collection: '{CHROMA_COLLECTION}' | Directory: '{CHROMA_DIR}'")

    # ── Deduplication pre-flight check ───────────────────────────────────────
    # Before embedding (which is slow and expensive), check if ANY chunk for this
    # source+user_id pair already exists in ChromaDB. If so, skip ingestion and
    # return the existing chunk count. This fulfils the dedup promise in the docstring
    # and prevents duplicate chunks from repeated ingestion calls.
    #
    # WHY NOT upsert: ChromaDB upsert requires stable chunk IDs. We use
    # auto-generated IDs today. The source-level existence check is a simpler,
    # fully correct guard for the common case (re-uploading same file/URL).
    primary_source = raw_docs[0].metadata.get("source", source) if raw_docs else source
    try:
        _col = _chroma_dedup_client.get_or_create_collection(
            CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        existing = _col.get(
            where={"$and": [{"source": {"$eq": primary_source}}, {"user_id": {"$eq": user_id}}]},
            limit=1,
        )
        if existing and existing.get("ids"):
            # Get accurate count for this source
            all_for_source = _col.get(
                where={
                    "$and": [{"source": {"$eq": primary_source}}, {"user_id": {"$eq": user_id}}]
                },
            )
            source_chunk_count = len(all_for_source.get("ids", []))
            _logger.info(
                "Dedup: source '%s' already ingested for user '%s' (%d chunks). Skipping re-ingestion.",
                primary_source,
                user_id,
                source_chunk_count,
            )
            print(f"  [DEDUP] Source already ingested ({source_chunk_count} chunks). Skipping.")
            return source_chunk_count
    except Exception as dedup_exc:
        # Non-critical: if dedup check fails (e.g. collection not yet created), proceed normally
        _logger.debug("Dedup pre-flight check failed (non-critical): %s", dedup_exc)

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DIR,
        # #8 Phase 1: Use cosine distance (industry standard for normalized text embeddings).
        # nomic-embed-text produces unit-norm vectors → cosine = canonical choice.
        # Creates a NEW collection; cannot change metric on existing collection.
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Add documents in one batch
    vectorstore.add_documents(chunked_docs)

    print("\n  [SUCCESS] Ingestion complete!")
    print(f"     Source     : {source}")
    print(f"     Chunks     : {len(chunked_docs)}")
    print(f"     Collection : {CHROMA_COLLECTION}")
    print(f"{'=' * 60}\n")

    return len(chunked_docs)
