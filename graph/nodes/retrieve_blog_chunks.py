"""
graph/nodes/retrieve_blog_chunks.py

PURPOSE:
    LangGraph node for the Retriever Agent — now a TRUE tool-use agent.

BEFORE (Pipeline):
    Hardcoded sequence: retriever_chain → ChromaDB search → dedup → rerank.
    The LLM generated sub-queries but the execution path was fixed in code.

AFTER (Tool-Use Agent):
    The LLM autonomously decides which tools to call and in what order:
      - search_vectorstore : search local ChromaDB (primary)
      - search_web         : live Tavily web search (fallback)

    The LLM observes what each tool returns and decides:
      "The vector store has enough context on this topic." → stop.
      "The vector store is thin on recent developments." → also call search_web.
      "I need more specifics on concept X." → call search_vectorstore again with a narrower query.

    This is the key difference from a pipeline: the LLM controls the flow,
    not hardcoded Python edges.

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
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from graph.tools import search_vectorstore, search_web
from graph.state import GraphState


# ─── Retriever Agent Setup ────────────────────────────────────────────────────
# The LLM is given both tools. Its docstrings tell it when to use each one.
# llama-3.1-8b-instant supports tool calling via Groq's API.
_retriever_llm = ChatGroq(model="llama-3.1-8b-instant")
_retriever_tools = [search_vectorstore, search_web]

# create_react_agent returns a compiled LangGraph sub-graph.
# We give it a system prompt that explains its role clearly.
_RETRIEVER_SYSTEM_PROMPT = """You are the Retriever Agent for a podcast production pipeline.

Your job is to gather ALL the information needed to produce an excellent podcast episode on the given topic.
You have been given an episode outline that tells you exactly what information is needed.

STRATEGY:
1. Read the episode outline carefully. Identify the key concepts, examples, and talking points.
2. Use search_vectorstore to find relevant content for each key concept.
   Make multiple targeted queries — one per concept is better than one broad query.
3. If the vector store results are sparse or missing key information, use search_web as a supplement.
4. Keep searching until you have solid content for ALL the talking points in the outline.

RULES:
- Always try search_vectorstore first.
- Only use search_web if local results are clearly insufficient.
- Stop when you have enough content to cover all talking points in the outline.
- Do not search for the same thing twice with the same query.
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
                            source = chunk[len("[Source:"):end_bracket].strip()
                            chunk = chunk[end_bracket + 1:].strip()
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": source, "retriever": "agent"},
                    ))
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
        Dict with "documents" (List[Document]) and "retrieval_strategy" (dict).
    """
    query = state.get("query", "")
    episode_outline = state.get("episode_outline", {})
    source_filter = state.get("source_filter")
    context_gaps = state.get("context_gaps", "")  # Set by Editor if requesting more context

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

    # ── Invoke the ReAct agent ─────────────────────────────────────────────────
    # The agent will call tools autonomously until it decides it's done.
    agent_result = _retriever_agent.invoke({"messages": [("user", agent_input)]})
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

    # ── Reranking (cross-encoder — same as before) ─────────────────────────────
    # Even though the agent selected the tools intelligently, we still rerank
    # to ensure the Writer receives the most relevant chunks first.
    from sentence_transformers import CrossEncoder

    if not unique_docs:
        final_docs = []
    else:
        model = CrossEncoder("BAAI/bge-reranker-base")
        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = model.predict(pairs)
        scored = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, _ in scored[:8]]  # top 8 (agent may gather more)

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

    return {
        "documents": final_docs,
        "retrieval_strategy": retrieval_strategy,
    }
