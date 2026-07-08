# graph/nodes/suggest_improvements.py
#
# LANGSMITH INTEGRATION:
#   Same config-propagation pattern. When this node runs it means the grader
#   rejected the script. The metadata we attach here makes it easy to see
#   in LangSmith EXACTLY why the node ran and what the rejection reason was —
#   without needing to open each trace and read the prompt input manually.

from typing import Dict

from langchain_core.runnables import RunnableConfig

from graph.chains.improvements_chain import suggest_improvements_chain
from graph.state import GraphState


def suggest_imporvements(state: GraphState, config: RunnableConfig = None) -> Dict:
    """
    LangGraph node: Generates editor-style improvement instructions for a
    rejected podcast script.

    Args:
        state:  Current LangGraph state. Contains "script", "script_evaluation",
                and "generation_count".
        config: RunnableConfig propagated by LangGraph. Carries LangSmith metadata.

    Returns:
        Dict with "improvement_suggestions" (str) field.
    """
    script = state.get("script", "")
    script_evaluation = state.get("script_evaluation", "")
    generation_count = state.get("generation_count", 0)

    # Guard: if either input is missing, return empty suggestions
    if not script or not script_evaluation:
        return {"improvement_suggestions": ""}

    # -----------------------------------------------------------------------
    # BUILD NODE-SPECIFIC CONFIG FOR LANGSMITH
    #
    # Attaching the rejection reason here is especially useful: in LangSmith
    # you can immediately see WHY this node ran without opening the trace input.
    # -----------------------------------------------------------------------
    parent_metadata = config.get("metadata", {}) if config else {}

    node_config = RunnableConfig(
        run_name=f"Suggest Improvements (attempt {generation_count})",
        tags=(config.get("tags", []) if config else []) + ["suggest-improvements"],
        metadata={
            **parent_metadata,
            "current_node": "suggest_improvements",
            "generation_attempt": generation_count,
            # Truncate to 500 chars for LangSmith metadata (long strings clutter the UI)
            "rejection_reason_preview": (script_evaluation or "")[:500],
        },
    )

    # Invoke the improvements chain, forwarding the enriched config
    suggestions = suggest_improvements_chain.invoke(
        {"script": script, "script_evaluation": script_evaluation},
        config=node_config,
    )

    return {"improvement_suggestions": suggestions}
