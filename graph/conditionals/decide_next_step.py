from langgraph.graph import END
from graph.state import GraphState
from graph.consts import GENERATE_AUDIO, SUGGEST_IMPROVEMENTS


def decide_next_step(state: GraphState):
    editor_action = state.get("editor_action")
    status = state.get("is_acceptable", False)
    
    is_approved = (editor_action == "accept") if editor_action else status
    
    max_iterations = state.get("max_generations", 1)
    current_generation_count = state.get("generation_count", 0)
    
    if is_approved:
        return GENERATE_AUDIO
    elif current_generation_count < max_iterations:
        return SUGGEST_IMPROVEMENTS
    else:
        return END
