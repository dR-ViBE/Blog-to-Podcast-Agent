# graph/nodes/plan_episode.py
#
# PURPOSE:
#   LangGraph node that runs the Planner Agent.
#   This node sits AFTER "retrieve" and BEFORE "generate_script" in the graph.
#
# CHANGE (Architecture Fix #2 — Retrieve-first):
#   BEFORE: plan_episode was the ENTRY POINT. It received only the user query and
#           planned from LLM general knowledge — with no awareness of what was in
#           the database. This produced aspirational outlines the retriever couldn't fulfil.
#
#   AFTER:  plan_episode runs AFTER retrieve_blog_chunks. It receives the RETRIEVED
#           DOCUMENTS from state and passes them as context to the planner chain.
#           The planner now plans ONLY from actual source material.
#           This guarantees every outline reflects content the writer actually has access to.
#
# WHAT IT DOES:
#   1. Reads retrieved documents and user query from state
#   2. Concatenates documents into a single context string
#   3. Invokes the planner chain with both query AND context
#   4. Converts the outline to a dict and stores it in state
#
# WHY THE OUTLINE IS STORED AS A DICT:
#   LangGraph's TypedDict state requires all values to be JSON-serializable.
#   Pydantic objects are not directly JSON-serializable in TypedDict context.
#   Calling .model_dump() converts the EpisodeOutline to a plain dict,
#   which LangGraph can store, serialize, and pass between nodes.
#
# WHEN THIS NODE RUNS:
#   ONCE per pipeline execution — after the first successful retrieval.
#   On improvement loops (suggest_improvements → generate_script), the
#   graph edge goes directly to generate_script, skipping this node.
#   The outline is already in state from the first run.

from typing import Dict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from graph.chains.planner_chain import planner_chain
from graph.state import GraphState
from graph.utils.node_metrics import NodeTimer



def plan_episode(state: GraphState, config: RunnableConfig = None) -> Dict:
    """
    LangGraph node: Analyzes retrieved blog content and produces a structured
    episode outline for the Writer Agent to follow.

    This is the Planner Agent — it decides WHAT to talk about and in WHAT ORDER,
    but does NOT write the actual script.

    Key change: now runs AFTER retrieval (not before). Receives retrieved documents
    from state so it can plan exclusively from actual source material.

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
    query = state.get("query", "")
    documents = state.get("documents", [])

    # ── Build context string from retrieved documents ─────────────────────────
    # Concatenate all retrieved chunks into a single text block for the planner.
    # The planner uses this as the SOLE source of truth for planning talking points.
    context = "\n\n".join(
        doc.page_content for doc in documents if isinstance(doc, Document) and doc.page_content.strip()
    )
    if not context:
        # Defensive fallback: if documents are present but all empty, signal the topic
        context = f"(No readable content available for query: {query})"

    # ── Build LangSmith metadata ─────────────────────────────────────────────
    parent_metadata = config.get("metadata", {}) if config else {}

    node_config = RunnableConfig(
        run_name="Plan Episode Outline",
        tags=(config.get("tags", []) if config else []) + ["plan-episode"],
        metadata={
            **parent_metadata,
            "current_node": "plan_episode",
            "context_length_chars": len(context),
            "document_count": len(documents),
        },
    )

    # ── Invoke the planner chain ─────────────────────────────────────────────
    # The chain now receives BOTH query (focus/angle) and context (source material).
    # Phase 2 (#10): NodeTimer records plan_episode latency to Prometheus.
    with NodeTimer("plan_episode"):
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

