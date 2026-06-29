# graph/nodes/plan_episode.py
#
# PURPOSE:
#   LangGraph node that runs the Planner Agent.
#   This node sits between "retrieve" and "generate_script" in the graph.
#
# WHAT IT DOES:
#   1. Reads retrieved documents and user query from state
#   2. Concatenates documents into a single context string
#   3. Invokes the planner chain to produce a structured EpisodeOutline
#   4. Converts the outline to a dict and stores it in state
#
# WHY THE OUTLINE IS STORED AS A DICT:
#   LangGraph's TypedDict state requires all values to be JSON-serializable.
#   Pydantic objects are not directly JSON-serializable in TypedDict context.
#   Calling .model_dump() converts the EpisodeOutline to a plain dict,
#   which LangGraph can store, serialize, and pass between nodes.
#
# WHEN THIS NODE RUNS:
#   ONCE per pipeline execution — before the first script generation.
#   On improvement loops (suggest_improvements → generate_script), the
#   graph edge goes directly to generate_script, skipping this node.
#   The outline is already in state from the first run.

from typing import Dict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from graph.chains.planner_chain import planner_chain
from graph.state import GraphState


def plan_episode(state: GraphState, config: RunnableConfig = None) -> Dict:
    """
    LangGraph node: Analyzes retrieved blog content and produces a structured
    episode outline for the Writer Agent to follow.

    This is the Planner Agent — it decides WHAT to talk about and in WHAT ORDER,
    but does NOT write the actual script.

    Args:
        state:  Current LangGraph state. Must contain:
                  - "documents": list of retrieved Document objects
                  - "query": the user's search query string
        config: RunnableConfig propagated by LangGraph from the graph's
                invoke() call. Carries LangSmith metadata and tags.

    Returns:
        Dict with "episode_outline" key containing the outline as a dict.
        This dict is merged into GraphState by LangGraph automatically.
    """

    # ── Read inputs from state ───────────────────────────────────────────────
    documents = state.get("documents", [])
    query = state.get("query", "")

    # Concatenate all document contents into a single context string.
    # This is the same approach used by generate_podcast_script, ensuring
    # the planner sees the exact same content the writer will see later.
    context = "\n\n".join(
        doc.page_content for doc in documents if isinstance(doc, Document)
    )

    # ── Build LangSmith metadata ─────────────────────────────────────────────
    # Same pattern as all other nodes — merge parent metadata with
    # node-specific fields so the planner appears with full context in traces.
    parent_metadata = config.get("metadata", {}) if config else {}

    node_config = RunnableConfig(
        run_name="Plan Episode Outline",
        tags=(config.get("tags", []) if config else []) + ["plan-episode"],
        metadata={
            **parent_metadata,
            "current_node": "plan_episode",
            "context_length_chars": len(context),
            "num_documents": len(documents),
        },
    )

    # ── Invoke the planner chain ─────────────────────────────────────────────
    # The chain returns a validated EpisodeOutline Pydantic object.
    # We pass both the user query (for angle/focus) and the context (for content).
    outline = planner_chain.invoke(
        {"query": query, "context": context},
        config=node_config,
    )

    # ── Convert Pydantic model to dict for state storage ─────────────────────
    # .model_dump() produces a plain Python dict that LangGraph can serialize.
    # Example output:
    #   {
    #     "episode_title": "AI Agents: Your New Digital Coworkers",
    #     "hook": "What if your next coworker didn't need coffee breaks?",
    #     "key_talking_points": [
    #       {"topic": "What are AI Agents?", "key_insight": "...", "estimated_duration_seconds": 60},
    #       ...
    #     ],
    #     "target_audience": "Tech-curious professionals...",
    #     "tone_guidance": "Conversational and upbeat...",
    #     "total_estimated_duration_seconds": 240,
    #     "sign_off_suggestion": "Challenge listeners to try building..."
    #   }
    outline_dict = outline.model_dump()

    return {"episode_outline": outline_dict}
