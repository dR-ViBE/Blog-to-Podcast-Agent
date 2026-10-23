# graph/nodes/grade_script.py
#
# UPGRADE: Now reads the triadic Editor action (accept / revise_script / request_more_context)
# and writes the appropriate routing fields back to state.
#
# KEY CHANGE:
#   Before: returned {"is_acceptable": bool, "script_evaluation": str}
#   After:  returns  {"is_acceptable": bool,       # still kept for backward compat
#                     "editor_action": str,         # the actual 3-way routing key
#                     "editor_notes": str,          # detailed notes for writer/retriever
#                     "context_gaps": Optional[str] # for Retriever if request_more_context
#                    }
#
# The decide_next_step conditional reads "editor_action" to route the graph.

from typing import Dict

from langchain_core.runnables import RunnableConfig

from graph.chains.script_grader_chain import script_grader
from graph.state import GraphState


def grade_script(state: GraphState, config: RunnableConfig = None) -> Dict:
    """
    LangGraph node: Editor Agent — grades the script and decides what happens next.

    Three possible decisions (set in editor_action):
      - "accept"               → decide_next_step routes to GENERATE_AUDIO
      - "revise_script"        → decide_next_step routes to SUGGEST_IMPROVEMENTS
      - "request_more_context" → decide_next_step routes back to RETRIEVE

    Args:
        state:  Contains "script" and "generation_count".
        config: RunnableConfig for LangSmith tracing.

    Returns:
        Dict with editor_action, editor_notes, context_gaps, is_acceptable,
        and script_evaluation fields.
    """
    script = state.get("script", "")
    generation_count = state.get("generation_count", 0)

    parent_metadata = config.get("metadata", {}) if config else {}

    node_config = RunnableConfig(
        run_name=f"Grade Script (attempt {generation_count})",
        tags=(config.get("tags", []) if config else []) + ["grade-script"],
        metadata={
            **parent_metadata,
            "current_node": "grade_script",
            "generation_attempt": generation_count,
            "script_length_chars": len(script),
            "script_word_count_approx": len(script.split()) if script else 0,
        },
    )

    result = script_grader.invoke({"script": script}, config=node_config)

    return {
        # Primary routing field — decide_next_step reads this
        "editor_action": result.action,
        "editor_notes": result.notes,
        "context_gaps": result.context_gaps,  # populated only for request_more_context

        # Kept for backward compatibility with existing state fields and tests
        "is_acceptable": result.action == "accept",
        "script_evaluation": result.notes,
    }
