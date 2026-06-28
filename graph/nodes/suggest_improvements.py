from typing import Dict
from graph.state import GraphState
from graph.chains.improvements_chain import suggest_improvements_chain


def suggest_imporvements(state: GraphState) -> Dict:
    script = state.get("script", "")
    script_evaluation = state.get("script_evaluation", "")

    if not script or not script_evaluation:
        return {"improvement_suggestions": ""}

    suggestions = suggest_improvements_chain.invoke(
        {"script": script, "script_evaluation": script_evaluation})

    return {"improvement_suggestions": suggestions}
