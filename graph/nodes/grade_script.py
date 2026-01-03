from typing import Dict
from graph.state import GraphState
from graph.chains.script_grader_chain import script_grader


def grade_script(state: GraphState) -> Dict:
    print("---Grade Script---")
    script = state.get("script", "")
    # script_evaluation=state.get("script_evaluation","")
    result = script_grader.invoke({"script": script})
    return ({"is_acceptable": result.is_acceptable, "script_evaluation": result.reason})
