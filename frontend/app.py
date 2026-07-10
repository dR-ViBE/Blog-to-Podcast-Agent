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
from utils import check_api_health, fetch_audio_bytes, make_podcast_request, make_ingest_request, fetch_prometheus_metrics

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
def main():
    """
    The main function that assembles and renders the entire Streamlit app.

    Streamlit re-runs this function every time the user interacts with the UI.
    The order of calls here determines the visual order on the page.
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

    # ── TABS NAVIGATION ──────────────────────────────────────────────────────
    tab_app, tab_dash = st.tabs(["🎙️ Podcast Generator", "📊 Diagnostics & System Metrics"])

    with tab_app:
        # ── INGESTION EXPANDER ───────────────────────────────────────────────
        with st.expander("📂 Ingest Source Documents (PDF / Text / URL)", expanded=True):
            st.markdown(
                "Upload a document (like a PDF resume, markdown notes) or crawl a URL to store "
                "it in the database before generating a podcast."
            )

            tab_ing1, tab_ing2 = st.tabs(["📄 Upload File (PDF/Text)", "🔗 Crawl URL"])

            with tab_ing1:
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

            with tab_ing2:
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

    with tab_dash:
        st.markdown("## 📊 Diagnostics & System Metrics")
        st.markdown(
            "This dashboard aggregates system metrics and evaluation scores directly from the FastAPI "
            "Prometheus registry and offline RAGAS evaluation results. It provides visual insights for non-developers."
        )
        st.markdown("---")

        metrics = fetch_prometheus_metrics(api_base_url)

        if not metrics.get("success", True):
            st.error(f"❌ Could not load metrics: {metrics.get('error')}")
            st.info("Ensure the FastAPI server is running and `ENABLE_METRICS=true` is set in your `.env`.")
        else:
            # ── ROW 1: KEY PERFORMANCE INDICATORS ─────────────────────────────────
            col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

            # Active Pipeline runs
            active = int(metrics.get("active_runs", 0))
            active_color = "🟢 Idle" if active == 0 else f"🟡 {active} In-Flight"
            with col_kpi1:
                st.markdown(
                    f"""
                    <div class="result-card" style="text-align: center;">
                        <div class="metric-label">Pipeline Status</div>
                        <div class="metric-value" style="font-size: 1.5rem; font-weight: 700; color: #764ba2;">{active_color}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Total requests (Success rate calculation)
            reqs = metrics.get("requests", {})
            total_reqs = sum(reqs.values())
            success_rate = (reqs.get("success", 0) / total_reqs * 100) if total_reqs > 0 else 100.0
            with col_kpi2:
                st.markdown(
                    f"""
                    <div class="result-card" style="text-align: center;">
                        <div class="metric-label">Total Runs (Success %)</div>
                        <div class="metric-value" style="font-size: 1.5rem; font-weight: 700; color: #4ade80;">
                            {int(total_reqs)} ({success_rate:.1f}%)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Total cost
            cost = metrics.get("cost_usd", 0.0)
            with col_kpi3:
                st.markdown(
                    f"""
                    <div class="result-card" style="text-align: center;">
                        <div class="metric-label">Total LLM Cost</div>
                        <div class="metric-value" style="font-size: 1.5rem; font-weight: 700; color: #ffb703;">
                            ${cost:.6f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Total tokens
            tokens = metrics.get("tokens", {})
            total_toks = sum(tokens.values())
            with col_kpi4:
                st.markdown(
                    f"""
                    <div class="result-card" style="text-align: center;">
                        <div class="metric-label">Total LLM Tokens</div>
                        <div class="metric-value" style="font-size: 1.5rem; font-weight: 700; color: #667eea;">
                            {int(total_toks):,}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── ROW 2: VISUAL CHART METRICS ───────────────────────────────────────
            st.markdown("### 📈 Pipeline Execution & Tools")
            col_ch1, col_ch2 = st.columns(2)

            with col_ch1:
                st.markdown("**🔄 Agent Generation Decisions**")
                gen_data = metrics.get("generation_attempts", {})
                st.bar_chart(gen_data)
                st.caption("Distribution of outcomes during the script generation loop.")

            with col_ch2:
                st.markdown("**🔍 Retriever Agent Tool Calls**")
                tool_data = metrics.get("tool_calls", {})
                st.bar_chart(tool_data)
                st.caption("How many times the Retriever Agent queried the vector store vs calling live Web search.")

            # ── ROW 3: PRIVACY & SECURITY EVENTS ──────────────────────────────────
            st.markdown("### 🛡️ Privacy & Security Activity Logs")
            col_sec1, col_sec2 = st.columns(2)

            with col_sec1:
                pii_data = metrics.get("pii_detections", {})
                total_pii = sum(pii_data.values())
                st.markdown(f"**🔒 PII Queries Anonymized: `{int(total_pii)}`**")
                if pii_data:
                    st.bar_chart(pii_data)
                    st.caption("Types of personal data (names, emails, etc.) masked in query inputs.")
                else:
                    st.success("No personal data detected in user inputs yet.")

            with col_sec2:
                injections = int(metrics.get("injection_attempts", 0))
                st.markdown(f"**🚫 Blocked Attacks & Jailbreaks: `{injections}`**")
                if injections > 0:
                    st.warning(f"⚠️ Security Guard blocked {injections} prompt injection attempts.")
                    st.caption("Check API logs for query SHA-256 hashes to analyze target payloads.")
                else:
                    st.success("No prompt injection or jailbreak attempts detected.")

            # ── ROW 4: RAGAS QUALITY EVALUATION RESULTS ───────────────────────────
            st.markdown("### 🧪 Quantitative RAGAS Evaluation Results")
            st.markdown(
                "Below are the results of the RAGAS offline evaluation suite run against the Golden Query Dataset. "
                "These metrics rate both our Retrieval precision and output Faithfulness."
            )

            # Try loading RAGAS results from JSON
            ragas_data = None
            ragas_path = Path("tests/eval/ragas_results.json")
            if ragas_path.exists():
                try:
                    with open(ragas_path) as rf:
                        ragas_data = json.load(rf)
                except Exception:
                    pass

            if ragas_data:
                scores = ragas_data.get("summary_scores", {})
                eval_time = ragas_data.get("evaluation_timestamp", "")
                if eval_time:
                    # Clean timestamp
                    eval_time = eval_time.split("T")[0]

                st.caption(f"Last evaluated on: **{eval_time}** | Golden dataset queries: **{ragas_data.get('num_queries', 8)}**")

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r4, col_r5, col_r6 = st.columns(3)

                # Context Precision
                cp = scores.get("context_precision")
                with col_r1:
                    st.metric(
                        label="Context Precision (Target: > 0.75)",
                        value=f"{cp:.4f}" if cp is not None else "N/A",
                        delta="Good" if cp and cp >= 0.75 else "Needs Ingestion" if cp else None,
                        delta_color="normal" if cp and cp >= 0.75 else "inverse"
                    )

                # Context Recall
                cr = scores.get("context_recall")
                with col_r2:
                    st.metric(
                        label="Context Recall (Target: > 0.70)",
                        value=f"{cr:.4f}" if cr is not None else "N/A",
                        delta="Good" if cr and cr >= 0.70 else "Needs Ingestion" if cr else None,
                        delta_color="normal" if cr and cr >= 0.70 else "inverse"
                    )

                # Context Entity Recall
                cer = scores.get("context_entity_recall")
                with col_r3:
                    st.metric(
                        label="Context Entity Recall (Target: > 0.60)",
                        value=f"{cer:.4f}" if cer is not None else "N/A",
                        delta="Good" if cer and cer >= 0.60 else "Low Coverage" if cer else None,
                        delta_color="normal" if cer and cer >= 0.60 else "inverse"
                    )

                # Answer Relevancy
                ar = scores.get("answer_relevancy")
                with col_r4:
                    st.metric(
                        label="Answer Relevancy (Target: > 0.80)",
                        value=f"{ar:.4f}" if ar is not None else "N/A",
                        delta="Good" if ar and ar >= 0.80 else "Tune Prompts" if ar else None,
                        delta_color="normal" if ar and ar >= 0.80 else "inverse"
                    )

                # Faithfulness
                faith = scores.get("faithfulness")
                with col_r5:
                    st.metric(
                        label="Faithfulness (Target: > 0.75)",
                        value=f"{faith:.4f}" if faith is not None else "N/A",
                        delta="No Hallucinations" if faith and faith >= 0.75 else "Hallucination Risk" if faith else None,
                        delta_color="normal" if faith and faith >= 0.75 else "inverse"
                    )

                # Noise Sensitivity
                ns = scores.get("noise_sensitivity")
                with col_r6:
                    # Lower is better
                    st.metric(
                        label="Noise Sensitivity (Target: < 0.30)",
                        value=f"{ns:.4f}" if ns is not None else "N/A",
                        delta="Robust" if ns and ns <= 0.30 else "Fragile" if ns else None,
                        delta_color="normal" if ns and ns <= 0.30 else "inverse"
                    )
            else:
                st.warning("⚠️ RAGAS evaluation results file `tests/eval/ragas_results.json` was not found.")
                st.info("Run the evaluation suite in your terminal to calculate and display these metrics: `poetry run python -m tests.eval.ragas_eval`")

            # ── ROW 5: CORPUS ANALYSIS ──────────────────────────────────────────
            st.markdown("### 📁 Knowledge Base Ingested Corpus")
            col_corp1, col_corp2 = st.columns([1, 2])

            with col_corp1:
                st.markdown("**Ingested Document Types**")
                ing_data = metrics.get("ingested_chunks", {})
                st.caption("Distribution of parsed vector store chunks by their source type.")
                st.bar_chart(ing_data)

            with col_corp2:
                st.markdown("**Metric Value Interpretation Guide**")
                st.markdown(
                    """
                    | Metric Name | Normal Range | Meaning | Status |
                    |---|---|---|---|
                    | **Pipeline Status** | `Idle` (0 active) | No users generating scripts. | ✅ normal |
                    | **Success %** | `> 90%` | Ratio of successful audio creations vs failures. | ✅ healthy |
                    | **Avg Cost / Run** | `< $0.01` | Average Groq LLaMA 3.1 token cost per execution. | ✅ ultra-low |
                    | **PII Queries** | `N/A` | Total input queries that had personal details masked. | 🔒 secure |
                    | **Blocked Attacks** | `0` | Total blocked SQL/prompt injection payloads. | 🛡️ secure |
                    """
                )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
# Streamlit runs the entire script on every interaction, so we call main()
# directly at the bottom (no `if __name__ == "__main__"` guard needed).
import json
from pathlib import Path
main()

