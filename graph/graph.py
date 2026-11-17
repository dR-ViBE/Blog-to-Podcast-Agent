# graph/graph.py
#
# PURPOSE:
#   Defines the LangGraph StateGraph — the node-and-edge structure that
#   controls the multi-agent pipeline execution order.
#
# MULTI-AGENT ARCHITECTURE (6 nodes):
#
#   RETRIEVE ──▶ route_after_retrieval ──▶ PLAN_EPISODE ──▶ GENERATE_SCRIPT ──▶ GRADE_SCRIPT
#                       │                                          ▲                    │
#                       │ (supplemental)                           │              ┌─────┴──────┐
#                       └──────────────────────────▶ GENERATE_SCRIPT    SUGGEST_IMPROVEMENTS  GENERATE_AUDIO ──▶ END
#                       │                                                   (if rejected         (if accepted)
#                       │ (no_content / fail-fast)                           & retries left)
#                       └──────────────────────────▶ END (with retrieval_failure_reason set)
#
# KEY ARCHITECTURAL CHANGES (Fix #2 + #3):
#
#   BEFORE:
#     PLAN_EPISODE (entry, query-only) → RETRIEVE → check_retrieval_status
#       → web_search_required on ANY empty result → END (or proceed)
#
#   AFTER:
#     RETRIEVE (entry) → route_after_retrieval gate
#       Initial pass, docs found       → PLAN_EPISODE (plans from actual docs)
#       Initial pass, no docs          → END (fail-fast, retrieval_failure_reason set)
#       Supplemental pass (Editor gap) → GENERATE_SCRIPT (direct, skip re-planning)
#       Supplemental pass, web needed  → END (HITL approval — only legitimate web search)
#
#   The Planner now ALWAYS runs AFTER retrieval and receives the retrieved docs
#   as context, so the outline reflects what the writer actually has access to.
#   Web search is ONLY offered as supplemental context (via Editor's request_more_context),
#   NEVER as a replacement for user-uploaded documents.

from langgraph.graph import END, StateGraph

from graph.checkpointer import get_checkpointer
from graph.conditionals import decide_next_step

from graph.consts import (
    GENERATE_AUDIO,
    GENERATE_SCRIPT,
    GRADE_SCRIPT,
    PLAN_EPISODE,
    RETRIEVE,
    SUGGEST_IMPROVEMENTS,
)
from graph.nodes import (
    generate_audio,
    generate_podcast_script,
    grade_script,
    plan_episode,
    retrieve_blog_chunks,
    suggest_imporvements,
)
from graph.state import GraphState

# ─── Create the StateGraph ──────────────────────────────────────────────────
# StateGraph manages a shared state dict (GraphState) that flows through
# all nodes. Each node reads from state and returns a partial dict that
# is merged back into state.
workflow = StateGraph(GraphState)


# ─── Register Nodes ─────────────────────────────────────────────────────────
# Each node is a Python function registered under a string name.
# The string name is used in edge definitions below.

workflow.add_node(RETRIEVE, retrieve_blog_chunks)         # Retriever Agent (entry point)
workflow.add_node(PLAN_EPISODE, plan_episode)             # Planner Agent (runs after retrieve)
workflow.add_node(GENERATE_SCRIPT, generate_podcast_script)  # Writer Agent
workflow.add_node(GRADE_SCRIPT, grade_script)            # Editor Agent (grading)
workflow.add_node(SUGGEST_IMPROVEMENTS, suggest_imporvements)  # Editor Agent (feedback)
workflow.add_node(GENERATE_AUDIO, generate_audio)        # Audio Agent


# ─── Define Edges (execution order) ─────────────────────────────────────────

# STEP 1: Entry point is now RETRIEVE (was PLAN_EPISODE).
# The retriever searches ChromaDB first so the planner knows what content exists.
workflow.set_entry_point(RETRIEVE)


def route_after_retrieval(state: GraphState) -> str:
    """
    Routing gate executed after every retrieval attempt.

    Replaces the old check_retrieval_status function, which only had two paths
    (proceed or pause-for-web-approval). This gate handles three distinct cases:

    INITIAL retrieval (context_gaps is not set — first pass through the pipeline):
      - Documents found   → PLAN_EPISODE (planner now has real source material)
      - No documents found → "no_content" → END with retrieval_failure_reason set
        The service layer reads retrieval_failure_reason and returns a precise
        user-facing error (never shows the web search dialog on initial failure).

    SUPPLEMENTAL retrieval (context_gaps set — Editor sent retriever back for more):
      - Documents found      → GENERATE_SCRIPT (skip re-planning, go straight to revision)
      - No docs + web needed → END (HITL web search approval — the ONLY legitimate path)
      - No docs + web approved → GENERATE_SCRIPT (proceed with existing docs)
    """
    is_supplemental = bool(state.get("context_gaps"))
    final_docs = state.get("documents", [])

    if is_supplemental:
        # ── Supplemental path (Editor requested more context) ─────────────────
        # Web search approval is legitimate here — it supplements existing content.
        if state.get("web_search_required", False) and not state.get("web_search_approved", False):
            return END  # Pause for HITL web search approval
        # Supplemental docs acquired (or web approved) — go straight to script revision
        return GENERATE_SCRIPT

    else:
        # ── Initial retrieval path ────────────────────────────────────────────
        # If nothing was found, fail fast — no web search offered on initial pass.
        # retrieval_failure_reason was set by the retriever to explain WHY.
        if state.get("retrieval_failure_reason") or not final_docs:
            return "no_content"  # Mapped to END below; service layer reads the reason
        # Documents found — proceed to plan from them
        return PLAN_EPISODE


# STEP 2: Route after retrieval — central gate replacing check_retrieval_status
workflow.add_conditional_edges(
    RETRIEVE,
    route_after_retrieval,
    {
        PLAN_EPISODE: PLAN_EPISODE,          # Initial pass, docs found → plan from them
        GENERATE_SCRIPT: GENERATE_SCRIPT,    # Supplemental pass → skip re-planning
        "no_content": END,                   # Initial pass, no docs → fail fast
        END: END,                            # Supplemental pass, web approval needed
    },
)

# STEP 3: After planning, go to script generation
# (Planning is skipped on supplemental loops — routed directly above)
workflow.add_edge(PLAN_EPISODE, GENERATE_SCRIPT)

# STEP 4: After script generation, the Editor Agent grades quality
workflow.add_edge(GENERATE_SCRIPT, GRADE_SCRIPT)

# STEP 5: Conditional routing after grading
#   - If accepted → generate audio
#   - If rejected AND retries left → suggest improvements (then loop back to writer)
#   - If request_more_context → back to RETRIEVE (supplemental pass)
#   - If rejected AND max retries reached → END (give up)
workflow.add_conditional_edges(
    GRADE_SCRIPT,
    decide_next_step,
    {
        GENERATE_AUDIO: GENERATE_AUDIO,
        SUGGEST_IMPROVEMENTS: SUGGEST_IMPROVEMENTS,
        RETRIEVE: RETRIEVE,
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
#
# Phase 2 (#6): We attach the SqliteSaver checkpointer so that every node's
# output is written to disk before moving to the next node. On replay (same
# thread_id in configurable), the graph skips already-completed nodes and
# resumes from the last successful checkpoint.
app = workflow.compile(checkpointer=get_checkpointer())
