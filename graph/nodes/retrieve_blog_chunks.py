import json
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_ollama import OllamaEmbeddings

from graph.chains.retriever_chain import retriever_chain
from graph.state import GraphState


def retrieve_blog_chunks(state: GraphState, config: RunnableConfig = None) -> Dict:
    """
    LangGraph node: Retriever Agent.

    Executes a multi-step intelligent retrieval pipeline:
      1. Reads the Planner's episode_outline from state
      2. Uses retriever_chain (LLM) to generate a RetrievalStrategy with 3-5 sub-queries
      3. Runs each sub-query against ChromaDB (optionally filtered by source_filter)
      4. Deduplicates chunks by content hash
      5. Reranks with BAAI/bge-reranker-base cross-encoder
      6. Returns the top-6 most relevant chunks

    Metadata Filtering:
      When state["source_filter"] is set, ChromaDB's `where` clause is used
      to restrict retrieval to chunks whose "source" metadata field starts with
      (or exactly matches) the provided filter value.

      Example: source_filter="https://lilianweng.github.io"
      → Only chunks crawled from that domain are candidates.
      → Prevents cross-contamination in multi-source knowledge bases.
    """
    query = state.get("query", "")
    episode_outline = state.get("episode_outline", {})
    source_filter = state.get("source_filter")  # Optional: None means search all

    if not query:
        raise ValueError("retrieve_blog_chunks requires 'query' in state")

    # ── 1. Generate Retrieval Strategy ────────────────────────────────────────
    strategy = retriever_chain.invoke({"outline": json.dumps(episode_outline, indent=2)})

    # ── 2. Vectorstore Setup ──────────────────────────────────────────────────
    vectorstore = Chroma(
        collection_name="blog_podcast_agent",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="./.chroma",
    )

    # ── 3. Build ChromaDB metadata filter (if source_filter is provided) ──────
    # ChromaDB `where` clause filters on document metadata.
    # We use $contains (substring match) so a domain prefix like
    # "https://lilianweng.github.io" matches any URL under that domain.
    chroma_filter: Optional[dict] = None
    if source_filter:
        chroma_filter = {"source": {"$contains": source_filter}}

    # ── 4. Multi-Query Parallel Retrieval ─────────────────────────────────────
    all_docs: List[Document] = []
    search_queries = strategy.queries if strategy.queries else [query]

    for q in search_queries:
        if chroma_filter:
            # Metadata-filtered search: only chunks from the specified source
            docs = vectorstore.similarity_search(query=q, k=4, filter=chroma_filter)
        else:
            # Unfiltered search: all ingested sources are candidates
            docs = vectorstore.similarity_search(query=q, k=4)
        all_docs.extend(docs)

    # ── 5. Deduplication ──────────────────────────────────────────────────────
    unique_docs: List[Document] = []
    seen_contents: set = set()
    for doc in all_docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            unique_docs.append(doc)

    # ── 6. Reranking ──────────────────────────────────────────────────────────
    from sentence_transformers import CrossEncoder

    if not unique_docs:
        final_docs = []
    else:
        model = CrossEncoder("BAAI/bge-reranker-base")
        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = model.predict(pairs)

        scored_docs = list(zip(unique_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, score in scored_docs[:6]]

    return {
        "documents": final_docs,
        "retrieval_strategy": strategy.model_dump(),
    }
