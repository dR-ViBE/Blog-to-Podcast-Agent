"""
graph/tools/vectorstore_tool.py

PURPOSE:
    LangChain @tool that lets the Retriever Agent search the local ChromaDB
    vector store. Supports compound metadata filtering (source URL + source type).

METADATA FILTERING (Phase 1 enhancement):
    ChromaDB supports compound `where` filters using $and / $or operators.
    We build a compound filter when both source_filter AND source_type_filter
    are provided, allowing queries like:
      "Only search PDF files from this specific domain"

    Single filters work as before:
      source_filter="https://lilianweng.github.io"  → filters by URL prefix
      source_type_filter="pdf"                       → filters to PDF content only

    Combined filters use ChromaDB's $and operator:
      {"$and": [{"source_type": {"$eq": "pdf"}}, {"source": {"$contains": "..."}}]}
"""

import logging
import os
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

# Create the persistent client at module level (main thread) to prevent
# Rust bindings threading crashes when instantiated inside LangGraph threads.
_chroma_client = chromadb.PersistentClient(path="./.chroma")


def _build_chroma_filter(
    source_filter: str = "",
    source_type_filter: str = "",
) -> Optional[dict]:
    """
    Build a ChromaDB metadata filter dict from optional filter parameters.

    ChromaDB filter syntax:
      Single condition: {"field": {"$operator": "value"}}
      AND compound:     {"$and": [condition1, condition2]}

    Supported operators: $eq, $ne, $contains, $not_contains, $in, $nin

    Args:
        source_filter:      URL prefix or file path substring to match against "source" metadata.
        source_type_filter: Exact source type to match: "url" | "pdf" | "text" | "directory".

    Returns:
        A ChromaDB-compatible filter dict, or None if no filters provided.
    """
    conditions = []

    if source_filter and source_filter.strip():
        conditions.append({"source": {"$contains": source_filter.strip()}})

    if source_type_filter and source_type_filter.strip():
        # source_type is an exact match field (e.g. "pdf", "url", "text")
        conditions.append({"source_type": {"$eq": source_type_filter.strip()}})

    if not conditions:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        # Multiple conditions → compound AND filter
        return {"$and": conditions}


@tool
def search_vectorstore(
    query: str,
    source_filter: str = "",
    source_type_filter: str = "",
) -> str:
    """
    Search the local ChromaDB vector store for content relevant to the query.

    Use this tool when you need to retrieve information from documents that
    have already been ingested into the knowledge base (blog posts, PDFs,
    text files). This is your PRIMARY tool — always try this first.

    Args:
        query: A specific, targeted search query. Be precise — vague queries
               return low-quality results. For best results, make one focused
               query per concept rather than one broad query for everything.
        source_filter: Optional. A URL prefix or file path substring to restrict
                       search to a specific source.
                       Example: "https://lilianweng.github.io" to only search that blog.
                       Leave as empty string "" to search all ingested sources.
        source_type_filter: Optional. Restrict search to a specific content type.
                            One of: "url", "pdf", "text", "directory".
                            Example: "pdf" to only search ingested PDF documents.
                            Leave as empty string "" to search all content types.

    Returns:
        The top relevant text chunks from the vector store, concatenated as a
        single string separated by "---". Returns a message if nothing is found.
    """
    chroma_filter = _build_chroma_filter(source_filter, source_type_filter)

    if chroma_filter:
        logger.debug("VectorStore search | query=%r | filter=%s", query[:60], chroma_filter)

    vectorstore = Chroma(
        collection_name="blog_podcast_agent",
        client=_chroma_client,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    )

    docs = vectorstore.similarity_search(
        query=query,
        k=2,
        filter=chroma_filter,
    )

    if not docs:
        filter_desc = ""
        if chroma_filter:
            filter_desc = f" (with filter: {chroma_filter})"
        return f"No results found in the vector store for query: '{query}'{filter_desc}"

    chunks = [
        f"[Source: {doc.metadata.get('source', 'unknown')} | Type: {doc.metadata.get('source_type', 'unknown')} | Ingested: {doc.metadata.get('ingested_at', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    ]
    return "\n---\n".join(chunks)
