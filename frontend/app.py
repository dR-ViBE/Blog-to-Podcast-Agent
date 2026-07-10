# frontend/app.py
#
# This is the main Streamlit application for the Blog-to-Podcast Agent.
#
# HOW THIS WORKS:
#   Streamlit re-runs this entire script from top to bottom every time
#   the user interacts with anything (clicks a button, types in a box).
#   We use `st.session_state` (a dictionary that persists between re-runs)
#   to remember things like the API result between interactions.
#
# This app ONLY talks to the FastAPI backend — it never calls LangGraph directly.

import os

import streamlit as st

# Import our API helper functions from utils.py
from utils import check_api_health, fetch_audio_bytes, make_podcast_request, make_ingest_request, fetch_analytics

# ---------------------------------------------------------------------------
# ENVIRONMENT CONFIGURATION
#
# When running locally:  API_BASE_URL is not set, so we default to localhost.
# When running in Docker: docker-compose.yml sets API_BASE_URL=http://api:8000
#   so the Streamlit server (running inside Docker) can reach the FastAPI
#   service via Docker's internal network using the service name "api".
# ---------------------------------------------------------------------------
_DEFAULT_API_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# This must be the very first Streamlit command in the script.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Blog to Podcast Agent",
    page_icon="🎙️",
    layout="wide",  # Use the full browser width
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CUSTOM CSS STYLING
# Streamlit's default look is clean but we enhance it with a few tweaks
# to make the app look more polished and professional.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ── Global font & background ─────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Header area ──────────────────────────────────────────────── */
        .hero-title {
            font-size: 2.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0rem;
            line-height: 1.2;
        }

        .hero-subtitle {
            font-size: 1.1rem;
            color: #888;
            margin-top: 0.3rem;
            margin-bottom: 1.5rem;
        }

        /* ── Result cards ─────────────────────────────────────────────── */
        .result-card {
            background: #1e1e2e;
            border: 1px solid #313147;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
        }

        .metric-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #888;
            margin-bottom: 0.2rem;
        }

        .metric-value {
            font-size: 1.15rem;
            font-weight: 600;
            color: #e0e0f0;
        }

        /* ── Status badges ────────────────────────────────────────────── */
        .badge-accepted {
            background: #1a3a2a;
            color: #4ade80;
            border: 1px solid #4ade80;
            border-radius: 20px;
            padding: 2px 14px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }

        .badge-rejected {
            background: #3a1a1a;
            color: #f87171;
            border: 1px solid #f87171;
            border-radius: 20px;
            padding: 2px 14px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }

        /* ── Divider ──────────────────────────────────────────────────── */
        .section-divider {
            border: none;
            border-top: 1px solid #2a2a3e;
            margin: 1.5rem 0;
        }

        /* ── Sidebar tweaks ───────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: #13131f;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# SESSION STATE INITIALISATION
#
# `st.session_state` is like a dictionary that survives between re-runs.
# We initialise keys here so we can safely read them anywhere in the script
# without getting a KeyError.
# ---------------------------------------------------------------------------
def init_session_state():
    """
    Sets default values in session_state the very first time the app loads.
    On subsequent re-runs these values are already set, so nothing changes.
    """
    defaults = {
        "result": None,  # Stores the last API response dictionary
        "audio_bytes": None,  # Stores the raw MP3 bytes for the audio player
        "audio_filename": None,  # Stores the MP3 filename for the download button
        "is_loading": False,  # True while the API call is in progress
        "error_message": None,  # Stores an error string to display, if any
        "ingested_source": None, # Stores the source URL or file path after ingestion
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_results():
    """
    Clears all stored results from session_state.
    Called when the user clicks the Reset button.
    """
    st.session_state["result"] = None
    st.session_state["audio_bytes"] = None
    st.session_state["audio_filename"] = None
    st.session_state["error_message"] = None


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[str, int]:
    """
    Renders the sidebar configuration panel.

    Returns:
        api_base_url    (str): The FastAPI server URL entered by the user.
        max_generations (int): Max script generation retries.
    """
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        # --- API URL ---
        st.markdown("**🔗 API Base URL**")
        api_base_url = st.text_input(
            label="API Base URL",  # Accessibility label (hidden visually)
            value=_DEFAULT_API_URL,  # Set by API_BASE_URL env var (Docker)
            label_visibility="collapsed",  # Hide label — we show our own above
            help=(
                "The URL where your FastAPI backend is running. "
                "In Docker this is set automatically via the API_BASE_URL "
                "environment variable in docker-compose.yml."
            ),
            key="api_url_input",
        )

        # Live health check indicator
        if st.button("🔍 Check API Status", use_container_width=True):
            with st.spinner("Checking..."):
                is_healthy = check_api_health(api_base_url)
            if is_healthy:
                st.success("✅ API is online and healthy.")
            else:
                st.error(
                    "❌ API is offline. Start it with:\n`poetry run uvicorn api.main:app --reload`"
                )

        st.markdown("---")

        # --- Max generations slider ---
        st.markdown("**🔄 Max Generations**")
        st.caption(
            "How many times the agent is allowed to regenerate and improve "
            "the podcast script if it fails quality checks."
        )
        max_generations = st.slider(
            label="Max Generations",
            min_value=1,
            max_value=10,
            value=3,
            label_visibility="collapsed",
            key="max_gen_slider",
        )

        # Show a helpful note about what the value means
        if max_generations == 1:
            st.info("ℹ️ The agent will make exactly 1 attempt.")
        elif max_generations >= 7:
            st.warning("⚠️ High values may take several minutes to complete.")

        st.markdown("---")

        # --- About section ---
        st.markdown("**📖 About**")
        st.caption(
            "This app converts blog content stored in a ChromaDB vector "
            "database into a podcast script and MP3 audio file using:\n\n"
            "🧠 **LangGraph** pipeline\n\n"
            "💬 **Groq** (LLaMA 3.1) for text generation\n\n"
            "🔊 **ElevenLabs** for voice synthesis"
        )

    return api_base_url, max_generations


# ---------------------------------------------------------------------------
# RESULTS SECTION
# ---------------------------------------------------------------------------
def render_results(api_base_url: str):
    """
    Renders the results panel after a successful podcast generation.

    Reads from st.session_state["result"] — the dict returned by the API.
    Only called when session_state["result"] is not None.

    Args:
        api_base_url: Needed to fetch the audio file from the backend.
    """
    result = st.session_state["result"]

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("## 📊 Results")

    # ── Row 1: Quick metrics ───────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    # Script accepted badge
    with col1:
        accepted = result.get("accepted", False)
        badge_class = "badge-accepted" if accepted else "badge-rejected"
        badge_text = "✅ Accepted" if accepted else "❌ Not Accepted"
        st.markdown(
            f"""
            <div class="result-card">
                <div class="metric-label">Script Quality</div>
                <span class="{badge_class}">{badge_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Generation count
    with col2:
        gen_count = result.get("generation_count", 0)
        st.markdown(
            f"""
            <div class="result-card">
                <div class="metric-label">Generation Attempts</div>
                <div class="metric-value">🔄 {gen_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Audio status
    with col3:
        audio_path = result.get("audio_path")
        audio_status = "🎵 Generated" if audio_path else "🔇 Not Generated"
        st.markdown(
            f"""
            <div class="result-card">
                <div class="metric-label">Audio Status</div>
                <div class="metric-value">{audio_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Evaluation ────────────────────────────────────────────────────────
    evaluation = result.get("evaluation")
    if evaluation:
        st.markdown("### 🧑‍⚖️ Grader Evaluation")
        st.info(evaluation)

    # ── Improvement suggestions ───────────────────────────────────────────
    suggestions = result.get("improvement_suggestions")
    if suggestions:
        st.markdown("### 💡 Improvement Suggestions")
        with st.expander("View editor feedback", expanded=False):
            st.markdown(suggestions)

    # ── Generated Script ──────────────────────────────────────────────────
    script = result.get("script")
    if script:
        st.markdown("### 📝 Generated Podcast Script")
        with st.expander("📖 Click to read the full script", expanded=True):
            # text_area gives a scrollable, copyable view of the script
            st.text_area(
                label="Script text",
                value=script,
                height=350,
                label_visibility="collapsed",
                disabled=True,  # Read-only — user doesn't need to edit this
                key="script_display",
            )

    # ── Audio Player & Download ───────────────────────────────────────────
    if audio_path:
        st.markdown("### 🎧 Podcast Audio")

        # Fetch the audio bytes the first time (or if not yet cached)
        if st.session_state["audio_bytes"] is None:
            with st.spinner("Loading audio player..."):
                audio_result = fetch_audio_bytes(api_base_url, audio_path)

            if audio_result["success"]:
                st.session_state["audio_bytes"] = audio_result["bytes"]
                st.session_state["audio_filename"] = audio_path
            else:
                st.error(audio_result["error"])

        # Render player and download button if we have the bytes
        if st.session_state["audio_bytes"]:
            # Built-in Streamlit audio widget — plays MP3 in the browser
            st.audio(st.session_state["audio_bytes"], format="audio/mp3")

            # Download button — lets the user save the MP3 file locally
            st.download_button(
                label="⬇️ Download MP3",
                data=st.session_state["audio_bytes"],
                file_name=st.session_state["audio_filename"] or "podcast.mp3",
                mime="audio/mpeg",
                use_container_width=False,
                key="download_btn",
            )
    elif not audio_path:
        # Inform the user clearly if audio wasn't generated
        st.warning(
            "🔇 Audio was not generated. This usually means the script did not "
            "pass quality checks within the maximum number of generation attempts. "
            "Try increasing **Max Generations** in the sidebar and running again."
        )


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def render_dashboard(api_base_url: str):
    """
    Renders a beautiful, interactive observability and analytics dashboard.
    Fetches live metrics from FastAPI's /analytics endpoint and displays
    them using premium visual widgets suited for non-developers.
    """
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("## 📊 Observability & System Analytics")
    st.markdown(
        "A real-time dashboard displaying system performance, costs, quality grades, "
        "and security events parsed from the underlying Prometheus registry."
    )

    # --- Fetch Live Metrics ---
    with st.spinner("Fetching system metrics..."):
        res = fetch_analytics(api_base_url)

    if not res["success"]:
        st.error(
            f"❌ Unable to connect to the Analytics endpoint.\n\n"
            f"Please make sure the FastAPI backend is running: `{res['error']}`"
        )
        return

    metrics_data = res["data"]

    # ── Helper function to extract metric values from registry ──────────────────
    def get_val(metric_name: str, label_filters: dict = None) -> float:
        if metric_name not in metrics_data:
            return 0.0
        samples = metrics_data[metric_name].get("samples", [])
        if not samples:
            return 0.0
        if label_filters:
            for s in samples:
                # Check if all filters match the sample's labels
                if all(s.get("labels", {}).get(k) == v for k, v in label_filters.items()):
                    return float(s.get("value", 0.0))
            return 0.0
        # If no filters, return the sum of all sample values (excluding metadata samples like _created)
        return sum(float(s.get("value", 0.0)) for s in samples if not s.get("name", "").endswith("_created"))

    # ── 1. AGGREGATE KEY METRICS ───────────────────────────────────────────────
    # Requests
    req_success = get_val("podcast_requests", {"status": "success"})
    req_no_audio = get_val("podcast_requests", {"status": "no_audio"})
    req_failed = get_val("podcast_requests", {"status": "failed"})
    total_requests = req_success + req_no_audio + req_failed

    success_rate = (req_success / total_requests * 100) if total_requests > 0 else 100.0

    # Active Runs
    active_runs = int(get_val("podcast_active_pipeline_runs"))

    # Cost
    total_cost_usd = get_val("podcast_llm_cost_usd", {"model": "llama-3.1-8b-instant"})

    # Tokens
    prompt_tokens = int(get_val("podcast_tokens_used", {"model": "llama-3.1-8b-instant", "token_type": "prompt"}))
    comp_tokens = int(get_val("podcast_tokens_used", {"model": "llama-3.1-8b-instant", "token_type": "completion"}))
    total_tokens = prompt_tokens + comp_tokens

    # Quality attempts
    first_attempt_ok = get_val("podcast_scripts_accepted", {"first_attempt": "True"})
    revised_attempt_ok = get_val("podcast_scripts_accepted", {"first_attempt": "False"})
    total_accepted = first_attempt_ok + revised_attempt_ok
    first_try_rate = (first_attempt_ok / total_accepted * 100) if total_accepted > 0 else 100.0

    # Security
    injections_blocked = int(get_val("podcast_injection_attempts", {"reason": "pattern_match"}))
    pii_detections = int(get_val("podcast_pii_detections"))
    pii_leaks_prevented = int(get_val("podcast_pii_output_detections"))

    # Ingestion
    ingested_pdf = get_val("podcast_ingestion_chunks", {"source_type": "pdf"})
    ingested_url = get_val("podcast_ingestion_chunks", {"source_type": "url"})
    ingested_txt = get_val("podcast_ingestion_chunks", {"source_type": "text"})
    total_chunks = ingested_pdf + ingested_url + ingested_txt

    # Tool calls
    tool_vector = get_val("podcast_retrieval_tool_calls", {"tool": "search_vectorstore"})
    tool_web = get_val("podcast_retrieval_tool_calls", {"tool": "search_web"})

    # ── 2. METRIC CARDS ROW ───────────────────────────────────────────────────
    st.markdown("### 📈 System Status & Performance")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Requests Run",
            value=int(total_requests),
            delta=f"{int(req_failed)} failed",
            delta_color="inverse",
        )
    with col2:
        st.metric(
            label="Pipeline Success Rate",
            value=f"{success_rate:.1f}%",
            delta="Target: >90%",
            delta_color="off",
        )
    with col3:
        st.metric(
            label="Active In-Flight runs",
            value=active_runs,
            delta="Idle" if active_runs == 0 else "Processing",
            delta_color="normal",
        )
    with col4:
        st.metric(
            label="Total LLM Cost (USD)",
            value=f"${total_cost_usd:.5f}",
            delta=f"{total_tokens:,} tokens",
            delta_color="off",
        )

    # ── 3. COLUMNS: QUALITY vs SECURITY ───────────────────────────────────────
    st.markdown("---")
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### 🎨 Content Quality & Pacing")
        
        # Quality score evaluation
        st.write("**First-Attempt Acceptance Rate**")
        st.caption("Percentage of scripts that pass the Senior Editor on the first try without revisions.")
        st.progress(first_try_rate / 100.0)
        
        quality_rating = "🟢 GOOD" if first_try_rate >= 70 else ("🟡 FAIR" if first_try_rate >= 40 else "🔴 POOR")
        st.write(f"Current Rating: **{quality_rating}** ({first_try_rate:.1f}%)")
        
        st.markdown("""
        **Pacing Value Analysis:**
        * **First-try rate > 70%:** Planner and Writer are highly aligned. Generates scripts in under 30 seconds.
        * **First-try rate < 50%:** Editor rejects drafts frequently. Leads to multiple retry iterations, increasing cost.
        """)

        st.markdown("**Knowledge Base Composition**")
        st.caption("Distribution of chunks stored in ChromaDB vector database by document source type:")
        
        kb_data = {
            "PDF Files": ingested_pdf,
            "Web URLs crawled": ingested_url,
            "Plain Text / Markdown": ingested_txt
        }
        for kb_type, count in kb_data.items():
            st.write(f"- **{kb_type}**: {int(count)} chunks")
        st.write(f"Total Database Size: **{int(total_chunks)} chunks**")

    with right_col:
        st.markdown("### 🛡️ Guardrails & Safety Auditing")
        
        # Status indicators
        sec_col1, sec_col2 = st.columns(2)
        with sec_col1:
            st.markdown(
                f"""
                <div class="result-card" style="text-align: center;">
                    <div class="metric-label" style="font-size: 0.75rem;">Prompt Injections Blocked</div>
                    <div class="metric-value" style="font-size: 2.2rem; color: #f87171;">🚫 {injections_blocked}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sec_col2:
            st.markdown(
                f"""
                <div class="result-card" style="text-align: center;">
                    <div class="metric-label" style="font-size: 0.75rem;">PII Entities Masked</div>
                    <div class="metric-value" style="font-size: 2.2rem; color: #fbbf24;">🔒 {pii_detections}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(f"**PII Leaks Prevented in Outputs:** `{pii_leaks_prevented}` warnings raised")

        st.markdown("""
        **Security Metric Ranges & Rules:**
        * **Injections Blocked:** Any value above 0 shows active defense in action. The PromptInjectionGuard blocks base64, DAN, role-play escape, and prompt extraction patterns.
        * **PII Entities Masked:** Scans input for PERSON, EMAIL, Phone, Credit Card, SSN. Ensures compliance with privacy standards by masking before LLM submission.
        """)

    # ── 4. RETRIEVAL STRATEGY ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Agent Tool Selection")
    st.caption("Tracks how the Retriever Agent selects tools based on the Planner's outline needs:")
    
    tool_col1, tool_col2 = st.columns(2)
    with tool_col1:
        st.markdown(
            f"""
            <div class="result-card" style="border-left: 5px solid #667eea;">
                <div class="metric-label">Local Vector Store Searches</div>
                <div class="metric-value">📚 {int(tool_vector)} searches</div>
                <div style="font-size: 0.8rem; color:#888; margin-top: 0.5rem;">Primary retrieval tool. Always queried first.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tool_col2:
        st.markdown(
            f"""
            <div class="result-card" style="border-left: 5px solid #764ba2;">
                <div class="metric-label">Live Web fallback searches</div>
                <div class="metric-value">🌐 {int(tool_web)} searches</div>
                <div style="font-size: 0.8rem; color:#888; margin-top: 0.5rem;">Called when vector results are sparse or recent facts are missing.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    """
    The main function that assembles and renders the entire Streamlit app.
    Now supports Navigation Tabs (Generator vs Analytics Dashboard).
    """
    # 1. Initialise session state (safe to call on every re-run)
    init_session_state()

    # 2. Render sidebar and capture user configuration
    api_base_url, max_generations = render_sidebar()

    # ── HERO HEADER ────────────────────────────────────────────────────────
    st.markdown(
        '<h1 class="hero-title">🎙️ Blog to Podcast Agent</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-subtitle">Convert blog posts, PDFs, or live web content into an AI-generated podcast — powered by LangGraph, Groq & ElevenLabs.</p>',
        unsafe_allow_html=True,
    )

    # Create top-level tabs
    tab_generator, tab_dashboard = st.tabs(["🎙️ Podcast Generator", "📊 Analytics & Observability"])

    # ── Tab 1: Podcast Generator ───────────────────────────────────────────
    with tab_generator:
        # ── INGESTION EXPANDER ───────────────────────────────────────────────
        with st.expander("📂 Ingest Source Documents (PDF / Text / URL)", expanded=True):
            st.markdown(
                "Upload a document (like a PDF resume, markdown notes) or crawl a URL to store "
                "it in the database before generating a podcast."
            )

            tab1, tab2 = st.tabs(["📄 Upload File (PDF/Text)", "🔗 Crawl URL"])

            with tab1:
                uploaded_file = st.file_uploader(
                    "Upload a PDF or TXT document",
                    type=["pdf", "txt", "md"],
                    key="doc_uploader",
                    label_visibility="collapsed"
                )

                if st.button("🚀 Ingest Document", key="ingest_doc_btn", use_container_width=True):
                    if uploaded_file is not None:
                        with st.spinner("Ingesting and embedding document..."):
                            file_bytes = uploaded_file.read()
                            res = make_ingest_request(
                                api_base_url,
                                file_data=(uploaded_file.name, file_bytes)
                            )
                        if res["success"]:
                            st.session_state["ingested_source"] = res["data"].get("source")
                            st.success(
                                f"✅ Successfully ingested {res['data'].get('chunks')} chunks "
                                f"from PDF: {uploaded_file.name}!\n\n"
                                f"Source path set: `{res['data'].get('source')}`"
                            )
                        else:
                            st.error(f"❌ Ingestion failed: {res['error']}")
                    else:
                        st.warning("⚠️ Please select a file to upload first.")

            with tab2:
                url_to_ingest = st.text_input(
                    "Enter URL to crawl",
                    placeholder="https://example.com/blog-post",
                    key="url_ingest_input",
                    label_visibility="collapsed"
                )

                if st.button("🚀 Ingest URL", key="ingest_url_btn", use_container_width=True):
                    if url_to_ingest and url_to_ingest.startswith("http"):
                        with st.spinner("Crawling and ingesting web page..."):
                            res = make_ingest_request(api_base_url, url_str=url_to_ingest)
                        if res["success"]:
                            st.session_state["ingested_source"] = res["data"].get("source")
                            st.success(
                                f"✅ Successfully ingested {res['data'].get('chunks')} chunks "
                                f"from URL: {url_to_ingest}!\n\n"
                                f"Source filter set: `{res['data'].get('source')}`"
                            )
                        else:
                            st.error(f"❌ Ingestion failed: {res['error']}")
                    else:
                        st.warning("⚠️ Please enter a valid URL starting with http:// or https://")

        # ── INPUT SECTION ──────────────────────────────────────────────────────
        st.markdown("### 🔎 Enter your Query")
        st.caption(
            "Type a topic or keyword that matches the blog content stored in "
            "your ChromaDB vector database. The agent will retrieve relevant "
            "chunks and generate a podcast episode."
        )

        query = st.text_input(
            label="Query",
            placeholder="e.g. key qualifications, summarize resume, Prompt Engineering...",
            label_visibility="collapsed",
            key="query_input",
            max_chars=300,
        )

        st.markdown("### 🎯 Optional: Scope Search by Source (Metadata Filter)")
        st.caption(
            "Only search chunks from the specific source URL or file path provided. "
            "Leave blank to search all ingested sources."
        )
        
        source_filter_val = st.text_input(
            label="Source Filter",
            value=st.session_state["ingested_source"] if st.session_state["ingested_source"] else "",
            placeholder="e.g. outputs/uploads/my_resume.pdf or https://lilianweng.github.io",
            label_visibility="collapsed",
            key="source_filter_input",
            max_chars=300,
        )

        # ── ACTION BUTTONS ─────────────────────────────────────────────────────
        col_btn1, col_btn2, col_spacer = st.columns([1, 1, 4])

        with col_btn1:
            generate_clicked = st.button(
                "🎙️ Generate Podcast",
                type="primary",  # Makes it the prominent blue button
                use_container_width=True,
                key="generate_btn",
                disabled=st.session_state["is_loading"],  # Disable while loading
            )

        with col_btn2:
            reset_clicked = st.button(
                "🔄 Reset",
                type="secondary",
                use_container_width=True,
                key="reset_btn",
                disabled=st.session_state["is_loading"],
            )

        # ── HANDLE RESET ───────────────────────────────────────────────────────
        if reset_clicked:
            reset_results()
            st.rerun()

        # ── HANDLE GENERATE ────────────────────────────────────────────────────
        if generate_clicked:
            if not query or len(query.strip()) < 3:
                st.warning("⚠️ Please enter a query of at least 3 characters.")
                st.stop()

            reset_results()
            st.session_state["is_loading"] = True

            with st.status("🚀 Podcast pipeline running...", expanded=True) as status_box:
                st.write("📡 Connecting to FastAPI backend...")
                st.write(f"🔍 Retrieving blog chunks for query: **{query}**")
                st.write("✍️ Generating podcast script with Groq (LLaMA 3.1)...")
                st.write("🧑‍⚖️ Grading script quality...")
                st.write("💡 Iterating with improvement suggestions if needed...")
                st.write("🔊 Synthesising audio via ElevenLabs...")
                st.write("⏳ This may take 1–3 minutes. Please wait...")

                api_result = make_podcast_request(
                    base_url=api_base_url,
                    query=query.strip(),
                    max_generations=max_generations,
                    source_filter=source_filter_val.strip() if source_filter_val else None,
                )

                if api_result["success"]:
                    st.session_state["result"] = api_result["data"]
                    st.session_state["is_loading"] = False
                    status_box.update(label="✅ Podcast generated successfully!", state="complete")
                else:
                    st.session_state["error_message"] = api_result["error"]
                    st.session_state["is_loading"] = False
                    status_box.update(label="❌ Pipeline failed. See error below.", state="error")

            st.rerun()

        # ── DISPLAY ERROR (if any) ─────────────────────────────────────────────
        if st.session_state["error_message"]:
            st.markdown("---")
            st.error(st.session_state["error_message"])

            with st.expander("🛠️ Troubleshooting Tips"):
                st.markdown(
                    """
                    **API not reachable?**
                    - Make sure the FastAPI server is running:
                      ```
                      poetry run uvicorn api.main:app --reload
                      ```
                    - Check the API Base URL in the sidebar matches the server address.
                    """
                )

        # ── DISPLAY RESULTS (if any) ───────────────────────────────────────────
        if st.session_state["result"]:
            render_results(api_base_url)

        # ── EMPTY STATE ────────────────────────────────────────────────────────
        if (
            st.session_state["result"] is None
            and st.session_state["error_message"] is None
            and not st.session_state["is_loading"]
        ):
            st.markdown("---")
            st.markdown(
                """
                <div style="text-align:center; padding: 3rem 0; color: #555;">
                    <div style="font-size: 3rem;">🎙️</div>
                    <div style="font-size: 1.1rem; margin-top: 0.5rem;">
                        Enter a query above and click <strong>Generate Podcast</strong> to begin.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Tab 2: Observability Dashboard ──────────────────────────────────────
    with tab_dashboard:
        render_dashboard(api_base_url)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
# Streamlit runs the entire script on every interaction, so we call main()
# directly at the bottom (no `if __name__ == "__main__"` guard needed).
main()

