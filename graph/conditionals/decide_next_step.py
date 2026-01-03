from langgraph.graph import END
from graph.state import GraphState
from graph.consts import GENERATE_SCRIPT,GENERATE_AUDIO

def decide_next_step(state:GraphState):
    status=state.get("is_acceptable",False)
    max_iterations=state.get("max_generations",1)
    current_generation_count=state.get("generation_count",0)
    if status:
        return GENERATE_AUDIO
    elif current_generation_count<max_iterations:
        return GENERATE_SCRIPT
    else:
        return END
