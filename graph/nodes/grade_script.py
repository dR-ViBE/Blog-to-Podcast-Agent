from typing import Dict,Any
from graph.state import GraphState
from graph.chains.script_grader_chain import script_grader

def grade_script(state:GraphState)->Dict:
    print("---Grade Script---")
    script=state.get("script","")
    script_evaluation=state.get("script_evaluation","")
    grade,reason=script_grader.invoke({"script":script})
    if grade:
        script_evaluation="acceptable"
    else:



