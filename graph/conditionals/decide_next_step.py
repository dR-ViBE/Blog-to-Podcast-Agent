from langgraph.graph import END

from graph.consts import GENERATE_AUDIO, SUGGEST_IMPROVEMENTS
from graph.state import GraphState


def decide_next_step(state: GraphState):
    status = state.get("is_acceptable", False)
    max_iterations = state.get("max_generations", 1)
    current_generation_count = state.get("generation_count", 0)
    if status:
        return GENERATE_AUDIO
    elif current_generation_count < max_iterations:
        return SUGGEST_IMPROVEMENTS
    else:
        return END
