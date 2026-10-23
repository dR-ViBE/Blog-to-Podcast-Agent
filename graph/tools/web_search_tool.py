"""
graph/tools/web_search_tool.py

PURPOSE:
    LangChain @tool that lets the Retriever Agent perform a live Tavily web
    search when the local vector store doesn't have enough information.

WHY THIS TOOL EXISTS:
    The local ChromaDB only contains documents that were explicitly ingested.
    If a user asks about a topic that's too thin in the local corpus — or asks
    about recent developments that postdate the ingestion — the Retriever Agent
    can autonomously decide to supplement with live web data.

    This is the key agentic behavior: the LLM observes that local results are
    insufficient and DECIDES to call a different tool. That decision is not
    hardcoded — the LLM makes it based on what it found (or didn't find) from
    the vector store.

TOOL CONTRACT:
    Input : A search query string
    Output: Summarised search results from the live web as a string

COST:
    Tavily free tier: 1,000 searches/month. Each call to this tool = 1 search.
    The LLM is instructed (via docstring) to use this as a FALLBACK, not primary.
"""

from langchain_core.tools import tool
from langchain_tavily import TavilySearch


# Initialise once at module load — not inside the function, to avoid
# re-creating the client on every tool call.
_tavily_search = TavilySearch(max_results=2)


@tool
def search_web(query: str) -> str:
    """
    Search the live web using Tavily for up-to-date information on a topic.

    Use this tool ONLY when:
    1. The vector store search returned no results or very sparse results, OR
    2. The episode outline requires information about recent events or
       developments that may not be in the local knowledge base.

    Do NOT use this as your primary tool — always try search_vectorstore first.
    Each web search consumes API quota, so use it selectively and purposefully.

    Args:
        query: A specific, targeted search query about the topic you need
               more information on. Be precise to get relevant results.

    Returns:
        A summary of the most relevant web search results as a string.
        Each result includes the title, URL, and content snippet.
    """
    try:
        results = _tavily_search.invoke({"query": query})

        if not results:
            return f"No web results found for: '{query}'"

        # Format results as a readable string for the LLM
        formatted = []
        for r in results:
            if isinstance(r, dict):
                title = r.get("title", "No title")
                url = r.get("url", "")
                content = r.get("content", "No content")
                formatted.append(f"[{title}]({url})\n{content}")
            else:
                formatted.append(str(r))

        return "\n---\n".join(formatted)

    except Exception as e:
        return f"Web search failed for '{query}': {str(e)}. Try a different query or use the vector store."
