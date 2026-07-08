"""
graph/tools/vectorstore_tool.py

PURPOSE:
    LangChain @tool that lets the Retriever Agent search the local ChromaDB
    vector store. The LLM calls this tool when local ingested content is
    sufficient to answer the episode's information needs.

WHY A TOOL (not just a function call):
    When wrapped with @tool, LangChain exposes this function to the LLM as a
    callable action with a name and docstring. The Retriever Agent's LLM reads
    the docstring to decide WHEN and HOW to call this tool. The docstring is
    the LLM's instruction manual — it must be precise and clear.

TOOL CONTRACT:
    Input : A search query string (semantic search)
            An optional source_filter string (metadata scoping)
    Output: A list of the most relevant text chunks from ChromaDB as a string
"""

import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings


@tool
def search_vectorstore(query: str, source_filter: Optional[str] = None) -> str:
    """
    Search the local ChromaDB vector store for content relevant to the query.

    Use this tool when you need to retrieve information from documents that
    have already been ingested into the knowledge base (blog posts, PDFs,
    text files). This is your PRIMARY tool — always try this first.

    Args:
        query: A specific, targeted search query. Be precise — vague queries
               return low-quality results. For best results, make one focused
               query per concept rather than one broad query for everything.
        source_filter: Optional. A URL prefix or file path to restrict the
                       search to a specific source. Example:
                       "https://lilianweng.github.io" to only search that blog.
                       Leave as None to search all ingested sources.

    Returns:
        The top relevant text chunks from the vector store, concatenated as a
        single string separated by "---". Returns a message if nothing is found.
    """
    chroma_filter: Optional[dict] = None
    if source_filter:
        chroma_filter = {"source": {"$contains": source_filter}}

    vectorstore = Chroma(
        collection_name="blog_podcast_agent",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="./.chroma",
    )

    docs = vectorstore.similarity_search(
        query=query,
        k=4,
        filter=chroma_filter,
    )

    if not docs:
        return f"No results found in the vector store for query: '{query}'"

    chunks = [f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs]
    return "\n---\n".join(chunks)
