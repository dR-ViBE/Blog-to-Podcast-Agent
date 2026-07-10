# frontend/utils.py
#
# WHY THIS FILE EXISTS:
#   All network communication with the FastAPI backend is kept here.
#   This means app.py stays clean and readable — it only handles the UI.
#   If the API changes in the future (e.g. new endpoints, auth headers),
#   you only need to update this one file, not the entire app.

import requests  # The library used to make HTTP requests to the FastAPI backend

# ---------------------------------------------------------------------------
# DATA CONTAINERS
# ---------------------------------------------------------------------------
# We use plain Python dictionaries for API results throughout the app.
# They are simple and work well with Streamlit's session_state system.


def make_podcast_request(base_url: str, query: str, max_generations: int, source_filter: str = None) -> dict:
    """
    Sends a POST request to the FastAPI /podcast endpoint.

    This function calls the backend and returns a dictionary containing
    either the successful API response data or an error description.

    Args:
        base_url:        The base URL of the FastAPI server
                         (e.g. "http://127.0.0.1:8000").
        query:           The topic the user typed in (e.g. "AI Agents").
        max_generations: Max number of times the agent will try to improve
                         the script before giving up.
        source_filter:   Optional source path/URL to filter retrieval.

    Returns:
        dict with keys:
            "success" (bool)  — True if the API call worked.
            "data"    (dict)  — The JSON response body from FastAPI (if success).
            "error"   (str)   — A human-friendly error message (if failure).
    """
    url = f"{base_url.rstrip('/')}/podcast"

    # The JSON body that FastAPI's PodcastRequest model expects
    payload = {
        "query": query,
        "max_generations": max_generations,
    }
    if source_filter:
        payload["source_filter"] = source_filter

    try:
        # timeout=300 seconds (5 minutes) — LangGraph can take a while
        # to crawl, generate, grade, and produce audio
        response = requests.post(url, json=payload, timeout=300)

        # raise_for_status() turns HTTP 4xx / 5xx into Python exceptions
        # so we can catch them cleanly below
        response.raise_for_status()

        # Return the parsed JSON from FastAPI as a Python dictionary
        return {"success": True, "data": response.json()}

    except requests.exceptions.ConnectionError:
        # This happens when the FastAPI server is not running at all
        return {
            "success": False,
            "error": (
                "❌ Cannot connect to the API server.\n\n"
                "Please make sure the FastAPI backend is running:\n"
                "`poetry run uvicorn api.main:app --reload`"
            ),
        }

    except requests.exceptions.Timeout:
        # The server took longer than 5 minutes — very unusual
        return {
            "success": False,
            "error": (
                "⏱️ The request timed out after 5 minutes.\n\n"
                "The LangGraph pipeline might be taking too long. "
                "Try reducing Max Generations in the sidebar."
            ),
        }

    except requests.exceptions.HTTPError as exc:
        # The server replied but with an error status (e.g. 422, 500)
        status_code = exc.response.status_code

        # Try to extract the "detail" field FastAPI puts in error responses
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)

        if status_code == 422:
            # 422 = Validation Error — the request body was malformed
            return {
                "success": False,
                "error": f"⚠️ Invalid request (422):\n{detail}",
            }
        elif status_code == 500:
            return {
                "success": False,
                "error": f"🔥 Internal server error (500):\n{detail}",
            }
        else:
            return {
                "success": False,
                "error": f"🚫 HTTP Error {status_code}:\n{detail}",
            }

    except Exception as exc:
        # Catch-all for anything unexpected (bad JSON, etc.)
        return {
            "success": False,
            "error": f"⚠️ Unexpected error: {exc}",
        }


def make_ingest_request(base_url: str, file_data: tuple = None, url_str: str = None) -> dict:
    """
    Sends a POST request to the FastAPI /ingest endpoint.

    Args:
        base_url:  The base URL of the FastAPI server.
        file_data: A tuple of (filename, file_bytes) to upload.
        url_str:   A URL string to crawl and ingest.

    Returns:
        dict with success (bool), data (dict), and error (str).
    """
    api_url = f"{base_url.rstrip('/')}/ingest"

    try:
        if file_data:
            filename, file_bytes = file_data
            files = {"file": (filename, file_bytes, "application/pdf" if filename.endswith(".pdf") else "text/plain")}
            response = requests.post(api_url, files=files, timeout=120)
        elif url_str:
            data = {"url": url_str}
            response = requests.post(api_url, data=data, timeout=120)
        else:
            return {"success": False, "error": "Provide either file_data or url_str."}

        response.raise_for_status()
        return {"success": True, "data": response.json()}

    except Exception as exc:
        return {"success": False, "error": f"Ingestion request failed: {exc}"}



def fetch_audio_bytes(base_url: str, filename: str) -> dict:
    """
    Downloads an MP3 audio file from GET /audio/{filename}.

    Streamlit's audio player and download button both need the raw
    bytes of the file — this function fetches those bytes from the API.

    Args:
        base_url: The base URL of the FastAPI server.
        filename: The MP3 filename returned in the /podcast response
                  (e.g. "podcast_abc123def456.mp3").

    Returns:
        dict with keys:
            "success" (bool)  — True if the download worked.
            "bytes"   (bytes) — Raw MP3 file bytes (if success).
            "error"   (str)   — Human-friendly error message (if failure).
    """
    url = f"{base_url.rstrip('/')}/audio/{filename}"

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        return {"success": True, "bytes": response.content}

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "❌ Cannot connect to the API to download audio.",
        }

    except requests.exceptions.HTTPError as exc:
        if exc.response.status_code == 404:
            return {
                "success": False,
                "error": (
                    f"🔍 Audio file '{filename}' was not found on the server.\n\n"
                    "This can happen if the script was not accepted within the "
                    "max generation limit."
                ),
            }
        return {
            "success": False,
            "error": f"🚫 HTTP Error {exc.response.status_code} while downloading audio.",
        }

    except Exception as exc:
        return {"success": False, "error": f"⚠️ Unexpected error downloading audio: {exc}"}


def check_api_health(base_url: str) -> bool:
    """
    Pings GET /health to check if the FastAPI backend is reachable.

    Returns:
        True  — if the server responds with {"status": "healthy"}.
        False — if the server is offline or returns an error.
    """
    try:
        url = f"{base_url.rstrip('/')}/health"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False
