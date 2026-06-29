# graph/nodes/generate_podcast_script.py
#
# PURPOSE:
#   LangGraph node that runs the Writer Agent.
#   The writer generates a podcast script by following:
#     1. The Planner's episode outline (structure, order, pacing)
#     2. The retrieved blog content (details, examples, quotes)
#     3. Editor improvement suggestions (if this is a retry attempt)
#
# WHAT CHANGED (PLANNER INTEGRATION):
#   BEFORE: The writer only received raw blog chunks and had to figure out
#           structure entirely on its own.
#   AFTER:  The writer receives a pre-planned outline that dictates episode
#           title, hook, talking point order, timing, tone, and sign-off.
#           This produces higher-quality, more structured scripts.
#
# LANGSMITH INTEGRATION:
#   Same pattern as before — accepts RunnableConfig from LangGraph and
#   forwards it with node-specific metadata to the chain call.

from typing import Dict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from graph.chains.podcast_script_chain import script_generation_chain
from graph.state import GraphState


def _format_outline_for_prompt(outline: dict) -> str:
    """
    Converts the Planner's outline dict into a human-readable string
    that the Writer LLM can easily follow.

    WHY THIS EXISTS:
      The outline is stored in state as a dict (from Pydantic .model_dump()).
      The LLM needs a clean, readable text format — not raw JSON.
      This function formats it into clearly labelled sections.

    Args:
        outline: The episode_outline dict from GraphState.

    Returns:
        A formatted string ready to be injected into the prompt template.
        Returns "(No outline available)" if the outline is empty/None.
    """
    if not outline:
        return "(No outline available)"

    # Build the formatted text block section by section
    parts = []

    parts.append(f"EPISODE TITLE: {outline.get('episode_title', 'N/A')}")
    parts.append(f"HOOK (opening line): {outline.get('hook', 'N/A')}")
    parts.append(f"TARGET AUDIENCE: {outline.get('target_audience', 'N/A')}")
    parts.append(f"TONE GUIDANCE: {outline.get('tone_guidance', 'N/A')}")
    parts.append(
        f"TARGET DURATION: {outline.get('total_estimated_duration_seconds', 'N/A')} seconds"
    )

    # Format each talking point with its index, topic, key insight, and duration
    talking_points = outline.get("key_talking_points", [])
    if talking_points:
        parts.append("\nTALKING POINTS (follow this order):")
        for i, tp in enumerate(talking_points, start=1):
            parts.append(
                f"  {i}. {tp.get('topic', 'N/A')}\n"
                f"     Key Insight: {tp.get('key_insight', 'N/A')}\n"
                f"     Duration: ~{tp.get('estimated_duration_seconds', 'N/A')} seconds"
            )

    parts.append(f"\nSIGN-OFF SUGGESTION: {outline.get('sign_off_suggestion', 'N/A')}")

    return "\n".join(parts)


def generate_podcast_script(state: GraphState, config: RunnableConfig = None) -> Dict:
    """
    LangGraph node: Generates a podcast script from the Planner's outline
    and retrieved document chunks.

    Args:
        state:  The current LangGraph state dictionary. Contains:
                  - documents: retrieved blog chunks
                  - episode_outline: the Planner's structured plan (dict)
                  - improvement_suggestions: editor feedback (empty on first attempt)
                  - generation_count: current attempt number
        config: RunnableConfig propagated by LangGraph for LangSmith tracing.

    Returns:
        Dict with updated "script" and "generation_count" fields.
    """
    # ── Read inputs from state ───────────────────────────────────────────────
    documents = state.get("documents", [])
    episode_outline = state.get("episode_outline", {})
    improvement_suggestions = state.get("improvement_suggestions", "")

    # Combine all document contents into a single context string
    context = "\n\n".join(
        doc.page_content for doc in documents if isinstance(doc, Document)
    )

    # Read and increment generation count
    generation_count = state.get("generation_count", 0) + 1

    # ── Format the outline for the prompt ────────────────────────────────────
    # Convert the dict into a clean, readable text block
    outline_text = _format_outline_for_prompt(episode_outline)

    # ── Build the chain input ────────────────────────────────────────────────
    # The updated prompt expects three variables:
    #   1. episode_outline: the formatted outline text
    #   2. context: the raw blog content
    #   3. improvement_suggestions: editor feedback (may be empty string)
    generation_input = {
        "episode_outline": outline_text,
        "context": context,
        "improvement_suggestions": improvement_suggestions or "(None — this is the first draft)",
    }

    # ── Build LangSmith metadata ─────────────────────────────────────────────
    parent_metadata = config.get("metadata", {}) if config else {}

    node_config = RunnableConfig(
        run_name=f"Generate Script (attempt {generation_count})",
        tags=(config.get("tags", []) if config else []) + ["generate-script"],
        metadata={
            **parent_metadata,
            "current_node": "generate_script",
            "generation_attempt": generation_count,
            "has_improvement_suggestions": bool(improvement_suggestions),
            "has_episode_outline": bool(episode_outline),
            "context_length_chars": len(context),
        },
    )

    # ── Invoke the writer chain ──────────────────────────────────────────────
    script = script_generation_chain.invoke(generation_input, config=node_config)

    return {
        "script": script,
        "generation_count": generation_count,
    }
