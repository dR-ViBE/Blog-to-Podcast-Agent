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

import streamlit as st

# Import our API helper functions from utils.py
from utils import make_podcast_request, fetch_audio_bytes, check_api_health


# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# This must be the very first Streamlit command in the script.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Blog to Podcast Agent",
    page_icon="🎙️",
    layout="wide",                # Use the full browser width
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
        "result": None,        # Stores the last API response dictionary
        "audio_bytes": None,   # Stores the raw MP3 bytes for the audio player
        "audio_filename": None, # Stores the MP3 filename for the download button
        "is_loading": False,   # True while the API call is in progress
        "error_message": None, # Stores an error string to display, if any
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
            label="API Base URL",              # Accessibility label (hidden visually)
            value="http://127.0.0.1:8000",
            label_visibility="collapsed",      # Hide label — we show our own above
            help="The URL where your FastAPI backend is running.",
            key="api_url_input",
        )

        # Live health check indicator
        if st.button("🔍 Check API Status", use_container_width=True):
            with st.spinner("Checking..."):
                is_healthy = check_api_health(api_base_url)
            if is_healthy:
                st.success("✅ API is online and healthy.")
            else:
                st.error("❌ API is offline. Start it with:\n`poetry run uvicorn api.main:app --reload`")

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
                disabled=True,   # Read-only — user doesn't need to edit this
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
        '<p class="hero-subtitle">Convert blog content into an AI-generated podcast — powered by LangGraph, Groq & ElevenLabs.</p>',
        unsafe_allow_html=True,
    )

    # ── INPUT SECTION ──────────────────────────────────────────────────────
    st.markdown("### 🔎 Enter your Query")
    st.caption(
        "Type a topic or keyword that matches the blog content stored in "
        "your ChromaDB vector database. The agent will retrieve relevant "
        "chunks and generate a podcast episode."
    )

    query = st.text_input(
        label="Query",
        placeholder="e.g. AI Agents, Prompt Engineering, Machine Learning...",
        label_visibility="collapsed",
        key="query_input",
        max_chars=300,
    )

    # ── ACTION BUTTONS ─────────────────────────────────────────────────────
    col_btn1, col_btn2, col_spacer = st.columns([1, 1, 4])

    with col_btn1:
        generate_clicked = st.button(
            "🎙️ Generate Podcast",
            type="primary",        # Makes it the prominent blue button
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
        st.rerun()  # Immediately re-render the page with cleared state

    # ── HANDLE GENERATE ────────────────────────────────────────────────────
    if generate_clicked:

        # --- Basic client-side validation ---
        if not query or len(query.strip()) < 3:
            st.warning("⚠️ Please enter a query of at least 3 characters.")
            st.stop()   # Stop processing the rest of the script

        # Clear any previous results before starting a new run
        reset_results()
        st.session_state["is_loading"] = True

        # --- API Call ---
        # st.status() creates a collapsible progress container
        with st.status("🚀 Podcast pipeline running...", expanded=True) as status_box:

            st.write("📡 Connecting to FastAPI backend...")
            st.write(f"🔍 Retrieving blog chunks for query: **{query}**")
            st.write("✍️ Generating podcast script with Groq (LLaMA 3.1)...")
            st.write("🧑‍⚖️ Grading script quality...")
            st.write("💡 Iterating with improvement suggestions if needed...")
            st.write("🔊 Synthesising audio via ElevenLabs...")
            st.write("⏳ This may take 1–3 minutes. Please wait...")

            # Make the actual API call
            api_result = make_podcast_request(
                base_url=api_base_url,
                query=query.strip(),
                max_generations=max_generations,
            )

            if api_result["success"]:
                # Store the result in session_state so it survives the re-run
                st.session_state["result"] = api_result["data"]
                st.session_state["is_loading"] = False
                status_box.update(
                    label="✅ Podcast generated successfully!", state="complete"
                )
            else:
                # Store the error message and clear loading state
                st.session_state["error_message"] = api_result["error"]
                st.session_state["is_loading"] = False
                status_box.update(
                    label="❌ Pipeline failed. See error below.", state="error"
                )

        # Force a re-run so the results section appears cleanly
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

                **Script not accepted?**
                - Try increasing **Max Generations** in the sidebar.
                - Make sure blog content has been ingested into ChromaDB
                  by running `python ingestion.py` first.

                **ElevenLabs error?**
                - Verify your `ELEVENLABS_API_KEY` is set correctly in `.env`.
                """
            )

    # ── DISPLAY RESULTS (if any) ───────────────────────────────────────────
    if st.session_state["result"]:
        render_results(api_base_url)

    # ── EMPTY STATE (no results yet, no error) ─────────────────────────────
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
                <div style="font-size: 0.85rem; margin-top: 0.5rem; color:#444;">
                    Make sure your FastAPI backend is running and blog content is ingested.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
# Streamlit runs the entire script on every interaction, so we call main()
# directly at the bottom (no `if __name__ == "__main__"` guard needed).
main()
