# graph/nodes/generate_podcast_script.py
#
# LANGSMITH INTEGRATION:
#   LangGraph automatically passes a `RunnableConfig` to every node function
#   as the second argument. This config carries the metadata and tags that were
#   set in services.py when we called langgraph_app.invoke(state, config=...).
#
#   By accepting `config` here and forwarding it to `script_generation_chain.invoke()`,
#   we ensure that in LangSmith the chain run appears as a CHILD of this node's
#   run, and it carries additional node-specific metadata like:
#     - which node is currently running ("current_node": "generate_script")
#     - which attempt number this is ("generation_attempt": N)
#
#   If we did NOT forward the config, the chain would still be traced (because
#   LANGCHAIN_TRACING_V2=true enables global tracing), but it would lose the
#   parent-child relationship with the node and the extra metadata.

from typing import Dict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig  # Standard LangChain config type

from graph.chains.podcast_script_chain import script_generation_chain
from graph.state import GraphState


def generate_podcast_script(state: GraphState, config: RunnableConfig = None) -> Dict:
    """
    LangGraph node: Generates a podcast script from retrieved document chunks.

    Args:
        state:  The current LangGraph state dictionary. Contains documents,
                improvement_suggestions, and generation_count.
        config: The RunnableConfig propagated by LangGraph from the graph's
                invoke() call. Contains LangSmith metadata and tags.
                Defaults to None if no config was passed at graph invocation.

    Returns:
        Dict with updated "script" and "generation_count" fields.
    """
    # Read documents from state
    documents = state.get("documents", [])

    # Combine all document contents into a single context string
    context = "\n\n".join(
        doc.page_content for doc in documents if isinstance(doc, Document)
    )

    # Read and increment generation count
    generation_count = state.get("generation_count", 0) + 1

    improvement_suggestions = state.get("improvement_suggestions", "")

    generation_input = {
        "context": context,
        "improvement_suggestions": improvement_suggestions,
    }

    # -----------------------------------------------------------------------
    # BUILD NODE-SPECIFIC CONFIG FOR LANGSMITH
    #
    # We merge the incoming config (from services.py) with node-specific
    # metadata so this chain call appears with full context in LangSmith.
    #
    # `RunnableConfig` constructor accepts the fields we want to update.
    # We spread the parent metadata first (using .get()), then overlay
    # our node-specific keys so they don't overwrite the parent metadata.
    # -----------------------------------------------------------------------
    parent_metadata = config.get("metadata", {}) if config else {}

    node_config = RunnableConfig(
        # Keep the run_name readable in LangSmith nested view
        run_name=f"Generate Script (attempt {generation_count})",

        # Forward parent tags and add a node-specific tag
        tags=(config.get("tags", []) if config else []) + ["generate-script"],

        # Merge parent metadata with node-specific fields
        metadata={
            **parent_metadata,                           # inherit query, topic, env, etc.
            "current_node": "generate_script",           # which node is running
            "generation_attempt": generation_count,      # attempt number (1, 2, 3...)
            "has_improvement_suggestions": bool(improvement_suggestions),
            "context_length_chars": len(context),        # size of retrieved content
        },
    )

    # Invoke the chain with the enriched config
    # LangSmith will now show:
    #   graph run → generate_script node → this chain (with all metadata above)
    script = script_generation_chain.invoke(generation_input, config=node_config)

    return {
        "script": script,
        "generation_count": generation_count,
    }
