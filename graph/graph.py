from langgraph.graph import END, StateGraph
from graph.state import GraphState
from graph.consts import GENERATE_SCRIPT, GRADE_SCRIPT, GENERATE_AUDIO
# ACCEPT, REGENERATE,RETRIEVE,
from graph.nodes import generate_audio, grade_script, generate_podcast_script
# from graph.chains import podcast_script_chain, script_grader_chain
from graph.conditionals import decide_next_step

#Create state Graph
workflow = StateGraph(GraphState)

#Creating the nodes
workflow.add_node(GENERATE_AUDIO, generate_audio)
workflow.add_node(GENERATE_SCRIPT, generate_podcast_script)
workflow.add_node(GRADE_SCRIPT, grade_script)
workflow.add_edge(GENERATE_SCRIPT, GRADE_SCRIPT)

#Starting point of the graph
workflow.set_entry_point(GENERATE_SCRIPT)

#Conditional edge
workflow.add_conditional_edges(GRADE_SCRIPT, decide_next_step, {
                               GENERATE_AUDIO: GENERATE_AUDIO, GENERATE_SCRIPT: GENERATE_SCRIPT, END: END})

#Endpoint
workflow.add_edge(GENERATE_AUDIO, END)

#Ready to execute
app=workflow.compile()