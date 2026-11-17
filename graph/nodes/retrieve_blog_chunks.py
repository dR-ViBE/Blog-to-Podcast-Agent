"""
graph/nodes/retrieve_blog_chunks.py

PURPOSE:
    LangGraph node for the Retriever Agent — a TRUE tool-use agent.

AFTER (Tool-Use Agent):
    The LLM autonomously decides which tools to call and in what order:
      - search_vectorstore : search local ChromaDB (primary)
      - search_web         : live Tavily web search (supplemental, Editor-approved only)

ARCHITECTURE CHANGE (Fix #3 — Route-After-Retrieval Gate):
    BEFORE: Any empty initial retrieval silently set web_search_required=True,
            offering the internet as a fallback regardless of why retrieval was empty.

    AFTER:  Two distinct retrieval contexts:

      1. INITIAL pass (context_gaps not set, generation_count == 0):
         - web_search_approved is NOT given to the agent (vectorstore-only search)
         - If final_docs is empty → set retrieval_failure_reason (fail-fast signal)
         - web_search_required is NEVER set on the initial pass
         - route_after_retrieval sees retrieval_failure_reason → routes to END
           with a user-facing "no content" error (not a web search dialog)

      2. SUPPLEMENTAL pass (context_gaps set by Editor's request_more_context):
         - web_search is available if user previously approved it
         - If final_docs is empty after supplemental search → web_search_required=True
         - This is the ONLY path where the web search approval dialog appears
         - Web search supplements the user's existing content, not replaces it

HOW create_react_agent WORKS:
    LangGraph's create_react_agent implements the ReAct (Reason + Act) loop:
      1. THINK : LLM reasons about what to do next given current observations
      2. ACT   : LLM calls one of the available tools with specific arguments
      3. OBSERVE: Tool result is added to the LLM's context
      4. REPEAT : Until the LLM decides it has enough information
      5. RETURN : LLM produces final output from accumulated tool results

RERANKING (still applied after agent):
    All documents gathered by the agent's tool calls are collected, deduplicated,
    and reranked using the BAAI/bge-reranker-base cross-encoder — giving us the
    best of both worlds: autonomous tool selection + precise relevance scoring.
"""

import json
import logging
from typing import Dict, List, Optional

import chromadb
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from graph.state import GraphState
from graph.tools import search_vectorstore, search_web
from graph.utils.node_metrics import NodeTimer

logger = logging.getLogger(__name__)

# ─── ChromaDB Client for user-doc count check ────────────────────────────────
# Used to distinguish "user has zero documents" from "query is off-topic".
# Same client as vectorstore_tool.py (PersistentClient is safe to share).
_chroma_client_check = chromadb.PersistentClient(path="./.chroma")


# ─── Lazy CrossEncoder Singleton ─────────────────────────────────────────────
# WHY LAZY (not module-level):
#   Loading at import time triggers a HuggingFace download during API startup
#   which conflicts with the httpx async lifecycle and causes:
#     RuntimeError: Cannot send a request, as the client has been closed.
#
# WHY SINGLETON (not per-call):
#   We still only want ONE model loaded in RAM. The getter caches after first
#   call. CrossEncoder.predict() is stateless and thread-safe for inference.
#
# WHY IMPORT IS INSIDE THE GETTER (not at module top):
#   sentence-transformers pulls in torch + transformers (~2 GB). Moving the
#   import inside the lazy getter means:
#     1. `from graph.nodes.retrieve_blog_chunks import retrieve_blog_chunks`
#        works in CI/test environments where sentence-transformers is not yet
#        installed (import-time safety).
#     2. If sentence-transformers IS installed, the first actual reranking call
#        will load the model — same runtime behaviour as before.
_CROSS_ENCODER = None  # type: ignore[assignment]  # actual type: Optional[CrossEncoder]


def _get_cross_encoder():
    """Return the cached CrossEncoder, loading it on first call.

    The import is performed here (not at module top) so that the module can
    be safely imported in CI environments where sentence-transformers is not
    installed. The model is only downloaded on the first inference call.
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        # Deferred import: keeps sentence-transformers (+ torch) out of the
        # module-level import graph so the API can start without it.
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        logger.info("Loading CrossEncoder model BAAI/bge-reranker-base (first call)...")
        _CROSS_ENCODER = CrossEncoder("BAAI/bge-reranker-base")
        logger.info("CrossEncoder model loaded successfully.")
    return _CROSS_ENCODER


# ─── Retriever Agent Setup ────────────────────────────────────────────────────
# The LLM is given both tools. Its docstrings tell it when to use each one.
# openai/gpt-oss-20b supports tool calling via Groq's API.
_retriever_llm = ChatGroq(model="openai/gpt-oss-20b", max_tokens=2500).with_fallbacks(
    [ChatGroq(model="openai/gpt-oss-120b", max_tokens=2500)]
)
_retriever_tools = [search_vectorstore, search_web]

# create_react_agent returns a compiled LangGraph sub-graph.
# We give it a system prompt that explains its role clearly.
_RETRIEVER_SYSTEM_PROMPT = """You are the Retriever Agent for a podcast production pipeline.

Your job is to gather ALL the information needed to produce an excellent podcast episode on the given topic.
You have been given an episode outline that tells you exactly what information is needed.

STRATEGY:
1. Read the episode outline carefully. Identify the key concepts and talking points.
2. Use search_vectorstore to find relevant content. Be highly efficient. Make 1 or 2 targeted queries at most.
3. Only use search_web if local vectorstore results are completely empty or missing crucial facts.
4. Stop searching immediately once you have a basic overview of the talking points. Do not perform excessive searches.

RULES:
- Always try search_vectorstore first.
- Only use search_web if local results are clearly insufficient.
- Minimize tool calls: do not call tools more than 2 times total.
- Stop when you have enough content to cover the outline points.
- Do not search for the same thing twice.
"""

_retriever_agent = create_react_agent(
    model=_retriever_llm,
    tools=_retriever_tools,
    prompt=_RETRIEVER_SYSTEM_PROMPT,
)


def _extract_docs_from_agent_messages(messages: list) -> List[Document]:
    """
    Extracts text content from ToolMessage results produced by the agent's
    tool calls and converts them into LangChain Document objects.

    The agent's message history contains a mix of AI messages (reasoning),
    ToolMessages (tool results), and a final AI message (summary). We extract
    all ToolMessage content since that's the actual retrieved text.
    """
    documents = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content
            if not content or "No results found" in content or "failed" in content.lower():
                continue
            # Each tool result may contain multiple chunks separated by "---"
            chunks = content.split("\n---\n")
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk:
                    # Parse source metadata if present
                    source = "unknown"
                    if chunk.startswith("[Source:"):
                        end_bracket = chunk.find("]")
                        if end_bracket > 0:
                            source = chunk[len("[Source:") : end_bracket].strip()
                            chunk = chunk[end_bracket + 1 :].strip()
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": source, "retriever": "agent"},
                        )
                    )
    return documents


def retrieve_blog_chunks(state: GraphState, config: RunnableConfig = None) -> Dict:
    """
    LangGraph node: Retriever Agent (Tool-Use).

    The agent autonomously decides which tools to call based on the episode
    outline and what it observes from each tool's response.

    If the Editor requested more context (editor_action == "request_more_context"),
    the agent is given the context_gaps to focus its search.

    Args:
        state: GraphState with query, episode_outline, source_filter,
               and optionally context_gaps (from Editor's request_more_context).
        config: RunnableConfig for LangSmith tracing.

    Returns:
        Dict with "documents" (List[Document]), "retrieval_strategy" (dict),
                  and "web_search_required" (bool).
    """
    query = state.get("query", "")
    episode_outline = state.get("episode_outline", {})
    source_filter = state.get("source_filter")
    context_gaps = state.get("context_gaps", "")  # Set by Editor if requesting more context
    web_search_approved = state.get("web_search_approved", False)

    if not query:
        raise ValueError("retrieve_blog_chunks requires 'query' in state")

    # ── Build the agent's input message ───────────────────────────────────────
    outline_str = json.dumps(episode_outline, indent=2) if episode_outline else f"Topic: {query}"

    if context_gaps:
        # Editor sent the agent back to get more context on specific gaps
        agent_input = (
            f"EPISODE OUTLINE:\n{outline_str}\n\n"
            f"CONTEXT GAPS (the Editor found these topics insufficiently covered):\n{context_gaps}\n\n"
            f"SOURCE FILTER: {source_filter or 'None (search all sources)'}\n\n"
            f"Focus your search on filling the context gaps identified above."
        )
    else:
        agent_input = (
            f"EPISODE OUTLINE:\n{outline_str}\n\n"
            f"SOURCE FILTER: {source_filter or 'None (search all sources)'}\n\n"
            f"Retrieve all information needed to write this podcast episode."
        )

    # ── Dynamically compile React Agent based on web search approval ─────────
    # If the user has not approved web search, the agent does not receive search_web tool.
    tools = [search_vectorstore]
    if web_search_approved:
        tools.append(search_web)
        agent_prompt = _RETRIEVER_SYSTEM_PROMPT
    else:
        agent_prompt = (
            "You are the Retriever Agent for a podcast production pipeline.\n\n"
            "Your job is to gather information to produce a podcast episode on the given topic.\n"
            "You ONLY have access to the `search_vectorstore` tool. "
            "Under no circumstances do you have `search_web` or any other tool. "
            "Do NOT attempt to call `search_web` or write tool calls for it. If `search_vectorstore` returns "
            "insufficient or empty results, simply stop searching."
        )

    dynamic_agent = create_react_agent(
        model=_retriever_llm,
        tools=tools,
        prompt=agent_prompt,
    )

    # ── Invoke the ReAct agent ─────────────────────────────────────────────────
    # The agent will call tools autonomously until it decides it's done.
    # We set a strict recursion_limit to prevent infinite tool-calling loops that blow up context size and TPM rate limits.
    # Phase 2 (#10): NodeTimer records retrieve latency to Prometheus.
    with NodeTimer("retrieve"):
        agent_result = dynamic_agent.invoke(
            {"messages": [("user", agent_input)]}, {"recursion_limit": 4}
        )
    all_messages = agent_result.get("messages", [])

    # ── Extract documents from tool call results ───────────────────────────────
    raw_docs = _extract_docs_from_agent_messages(all_messages)

    # ── Deduplication ──────────────────────────────────────────────────────────
    unique_docs: List[Document] = []
    seen: set = set()
    for doc in raw_docs:
        h = hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            unique_docs.append(doc)

    # ── Reranking (cross-encoder) ──────────────────────────────────────────────
    # Uses the module-level _CROSS_ENCODER instance (loaded once at import time).
    # Even though the agent selected tools intelligently, we still rerank to
    # ensure the Writer receives the most relevant chunks first.
    if not unique_docs:
        final_docs = []
    else:
        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = _get_cross_encoder().predict(pairs)
        scored = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)

        # Apply a strict relevance threshold to discard irrelevant documents
        # (e.g. generic healthcare articles retrieved for an excavator query)
        RELEVANCE_THRESHOLD = 0.0005
        filtered_scored = [(doc, score) for doc, score in scored if score > RELEVANCE_THRESHOLD]
        final_docs = [doc for doc, _ in filtered_scored[:4]]

    # ── Build retrieval strategy summary for state/LangSmith ──────────────────
    tool_calls_made = [
        msg.tool_calls[0]["name"] if hasattr(msg, "tool_calls") and msg.tool_calls else None
        for msg in all_messages
    ]
    tools_used = list({t for t in tool_calls_made if t})

    retrieval_strategy = {
        "tools_used": tools_used,
        "total_raw_docs": len(raw_docs),
        "unique_docs": len(unique_docs),
        "final_docs": len(final_docs),
        "context_gaps_addressed": bool(context_gaps),
    }

    # ── Set retrieval outcome signals ──────────────────────────────────────────
    #
    # Two fundamentally different cases when final_docs is empty:
    #
    # Case A — INITIAL retrieval pass (context_gaps not set):
    #   The user just submitted a query. We found nothing in their documents.
    #   This should FAIL FAST with an informative message, NOT offer web search.
    #   We set retrieval_failure_reason so route_after_retrieval can route to END.
    #   We try to distinguish WHY nothing was found:
    #     - "no_documents_ingested" : user has uploaded nothing at all for this user_id
    #     - "no_matching_content"   : user has docs but query is off-topic / unrelated
    #
    # Case B — SUPPLEMENTAL retrieval pass (context_gaps set by Editor):
    #   The Editor identified gaps in an existing script and sent retriever back.
    #   Web search is legitimate here — it supplements existing user content.
    #   We set web_search_required=True if no docs found, triggering HITL dialog.

    web_search_required = False
    retrieval_failure_reason = None

    if not final_docs:
        if context_gaps:
            # Case B: Supplemental pass — editor found context gaps.
            # Web search is acceptable as a supplement to existing source material.
            if not web_search_approved:
                web_search_required = True
                logger.info(
                    "Supplemental retrieval empty | web_search_required=True | context_gaps=%r",
                    context_gaps[:100] if context_gaps else "",
                )
        else:
            # Case A: Initial retrieval pass — fail fast.
            # Determine WHY there are no results so we can give a precise message.
            try:
                collection = _chroma_client_check.get_collection("blog_podcast_agent_v2")
                # Count docs for this specific user_id
                user_docs = collection.get(
                    where={"user_id": {"$eq": state.get("user_id", "default_user")}},
                    limit=1,
                )
                has_any_docs = bool(user_docs and user_docs.get("ids"))
            except Exception as chroma_exc:
                logger.debug("ChromaDB count check failed (non-critical): %s", chroma_exc)
                has_any_docs = False  # assume no docs if check fails

            if has_any_docs:
                retrieval_failure_reason = "no_matching_content"
                logger.info(
                    "Initial retrieval empty — user has docs but query is off-topic | "
                    "query=%r | user_id=%r",
                    query,
                    state.get("user_id", "default_user"),
                )
            else:
                retrieval_failure_reason = "no_documents_ingested"
                logger.info(
                    "Initial retrieval empty — user has NO documents ingested | user_id=%r",
                    state.get("user_id", "default_user"),
                )

    return {
        "documents": final_docs,
        "retrieval_strategy": retrieval_strategy,
        "web_search_required": web_search_required,
        "retrieval_failure_reason": retrieval_failure_reason,
    }
