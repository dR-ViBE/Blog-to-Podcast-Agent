# graph/consts.py
#
# PURPOSE:
#   String constants for LangGraph node names.
#   Using constants instead of raw strings prevents typos from causing
#   silent graph wiring bugs (e.g., "genrate_script" vs "generate_script").
#
# NAMING CONVENTION:
#   The Python variable name is UPPER_SNAKE_CASE.
#   The string value is lower_snake_case (LangGraph convention).

# ── Node Names ───────────────────────────────────────────────────────────────

RETRIEVE = "retrieve"

PLAN_EPISODE = "plan_episode"  # NEW: Planner Agent node

GENERATE_SCRIPT = "generate_script"

GRADE_SCRIPT = "grade_script"

GENERATE_AUDIO = "generate_audio"

SUGGEST_IMPROVEMENTS = "suggest_improvements"

# ── Routing Constants ────────────────────────────────────────────────────────
# Used by decide_next_step conditional but not as node names in the graph.

ACCEPT = "accept"

REGENERATE = "regenerate"
