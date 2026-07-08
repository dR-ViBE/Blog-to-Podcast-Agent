# graph/graph.py
#
# PURPOSE:
#   Defines the LangGraph StateGraph — the node-and-edge structure that
#   controls the multi-agent pipeline execution order.
#
# MULTI-AGENT ARCHITECTURE (6 nodes):
#
#   retrieve ──▶ plan_episode ──▶ generate_script ──▶ grade_script
#                                       ▲                  │
#                                       │            ┌─────┴──────┐
#                                       │            ▼            ▼
#                               suggest_improvements  generate_audio ──▶ END
#                                 (if rejected          (if accepted)
#                                  & retries left)
#                                       │
#                                    (if rejected & max retries) ──▶ END
#
# KEY ARCHITECTURAL DECISION:
#   The Planner runs ONCE (retrieve → plan_episode → generate_script).
#   On improvement loops, the edge goes directly from suggest_improvements
#   back to generate_script — the planner is NOT re-run because:
#     1. The outline is already in state from the first pass.
#     2. The outline reflects the SOURCE MATERIAL which hasn't changed.
#     3. Only the SCRIPT needs improvement, not the plan itself.
#     4. Re-running the planner would waste an LLM call and might produce
#        a different outline, confusing the writer mid-iteration.

from langgraph.graph import END, StateGraph

from graph.state import GraphState
from graph.consts import (
    RETRIEVE,
    PLAN_EPISODE,          # NEW: Planner Agent
    GENERATE_SCRIPT,
    GRADE_SCRIPT,
    GENERATE_AUDIO,
    SUGGEST_IMPROVEMENTS,
)
from graph.nodes import (
    generate_audio,
    grade_script,
    generate_podcast_script,
    plan_episode,          # NEW: Planner Agent node function
    retrieve_blog_chunks,
    suggest_imporvements,
)
from graph.conditionals import decide_next_step


# ─── Create the StateGraph ──────────────────────────────────────────────────
# StateGraph manages a shared state dict (GraphState) that flows through
# all nodes. Each node reads from state and returns a partial dict that
# is merged back into state.
workflow = StateGraph(GraphState)


# ─── Register Nodes ─────────────────────────────────────────────────────────
# Each node is a Python function registered under a string name.
# The string name is used in edge definitions below.

workflow.add_node(RETRIEVE, retrieve_blog_chunks)            # Retriever Agent
workflow.add_node(PLAN_EPISODE, plan_episode)                # Planner Agent (NEW)
workflow.add_node(GENERATE_SCRIPT, generate_podcast_script)  # Writer Agent
workflow.add_node(GRADE_SCRIPT, grade_script)                # Editor Agent (grading)
workflow.add_node(SUGGEST_IMPROVEMENTS, suggest_imporvements)  # Editor Agent (feedback)
workflow.add_node(GENERATE_AUDIO, generate_audio)            # Audio Agent


# ─── Define Edges (execution order) ─────────────────────────────────────────

# STEP 1: Start at the Planner Agent (generates outline from query)
workflow.set_entry_point(PLAN_EPISODE)

# STEP 2: Retriever Agent reasons about outline to fetch targeted documents
workflow.add_edge(PLAN_EPISODE, RETRIEVE)

# STEP 3: Writer Agent generates a script following the outline and documents
workflow.add_edge(RETRIEVE, GENERATE_SCRIPT)

# STEP 4: After script generation, the Editor Agent grades quality
workflow.add_edge(GENERATE_SCRIPT, GRADE_SCRIPT)

# STEP 5: Conditional routing after grading
#   - If accepted → generate audio
#   - If rejected AND retries left → suggest improvements (then loop back to writer)
#   - If rejected AND max retries reached → END (give up)
workflow.add_conditional_edges(
    GRADE_SCRIPT,
    decide_next_step,
    {
        GENERATE_AUDIO: GENERATE_AUDIO,
        SUGGEST_IMPROVEMENTS: SUGGEST_IMPROVEMENTS,
        END: END,
    },
)

# STEP 6 (loop): After improvement suggestions, go DIRECTLY back to the Writer
# NOTE: We skip the Planner here — the outline is already in state.
workflow.add_edge(SUGGEST_IMPROVEMENTS, GENERATE_SCRIPT)

# STEP 7: After audio generation, the pipeline is complete
workflow.add_edge(GENERATE_AUDIO, END)


# ─── Compile the Graph ──────────────────────────────────────────────────────
# .compile() validates all edges, checks for unreachable nodes, and returns
# a runnable LangGraph object that can be invoked with app.invoke(state).
app = workflow.compile()
