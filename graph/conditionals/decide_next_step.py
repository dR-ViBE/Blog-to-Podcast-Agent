from langgraph.graph import END

from graph.consts import GENERATE_AUDIO, SUGGEST_IMPROVEMENTS, RETRIEVE
from graph.state import GraphState


def decide_next_step(state: GraphState):
    action = state.get("editor_action")
    if not action:
        action = "accept" if state.get("is_acceptable", False) else "revise_script"

    if action == "accept":
        return GENERATE_AUDIO

    max_iterations = state.get("max_generations", 1)
    current_generation_count = state.get("generation_count", 0)

    if current_generation_count < max_iterations:
        if action == "request_more_context":
            return RETRIEVE
        else:
            return SUGGEST_IMPROVEMENTS
    else:
        return END

