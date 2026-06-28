# graph/nodes/grade_script.py
#
# LANGSMITH INTEGRATION:
#   Same pattern as generate_podcast_script.py — we accept the propagated
#   RunnableConfig from LangGraph and forward it (with extra grading-specific
#   metadata) to the script_grader chain call.
#
#   In LangSmith this produces:
#     graph run
#       └── grade_script (node)
#             └── ChatPromptTemplate + ChatGroq (structured output) + GradeScript
#                 ← all tagged with "grade-script" and current generation count

from typing import Dict

from langchain_core.runnables import RunnableConfig

from graph.state import GraphState
from graph.chains.script_grader_chain import script_grader


def grade_script(state: GraphState, config: RunnableConfig = None) -> Dict:
    """
    LangGraph node: Grades the generated podcast script for quality.

    Uses a structured-output LLM chain (script_grader) to evaluate whether
    the script meets length, structure, format, and tone requirements.

    Args:
        state:  Current LangGraph state. Contains "script" and "generation_count".
        config: RunnableConfig propagated by LangGraph. Carries LangSmith
                metadata from services.py.

    Returns:
        Dict with "is_acceptable" (bool) and "script_evaluation" (str) fields.
    """
    script = state.get("script", "")
    generation_count = state.get("generation_count", 0)

    # -----------------------------------------------------------------------
    # BUILD NODE-SPECIFIC CONFIG FOR LANGSMITH
    # -----------------------------------------------------------------------
    parent_metadata = config.get("metadata", {}) if config else {}

    node_config = RunnableConfig(
        run_name=f"Grade Script (attempt {generation_count})",
        tags=(config.get("tags", []) if config else []) + ["grade-script"],
        metadata={
            **parent_metadata,
            "current_node": "grade_script",
            "generation_attempt": generation_count,
            "script_length_chars": len(script),  # Useful for debugging quality issues
            # Word count approximation — helpful for understanding grader decisions
            "script_word_count_approx": len(script.split()) if script else 0,
        },
    )

    # Invoke the grader chain, forwarding the enriched config
    result = script_grader.invoke({"script": script}, config=node_config)

    return {
        "is_acceptable": result.is_acceptable,
        "script_evaluation": result.reason,
    }
