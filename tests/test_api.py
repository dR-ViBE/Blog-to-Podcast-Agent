# tests/test_api.py
#
# PURPOSE:
#   Tests the FastAPI application layer — routes, request validation,
#   response models, and error handling — without making any real
#   external API calls (no Groq, no ElevenLabs, no ChromaDB).
#
# HOW TESTS WORK WITHOUT REAL SERVICES:
#   We use FastAPI's built-in TestClient which runs the app in-process.
#   Tests that would trigger the LangGraph pipeline (POST /podcast) are
#   tested only at the HTTP validation layer (422 errors) — we verify
#   that FastAPI correctly rejects bad requests before ever reaching the
#   service layer. No actual graph execution happens.
#
# FIXTURE SOURCE:
#   The `fastapi_client` fixture is defined in conftest.py and is shared
#   across this entire test session (created once, reused everywhere).

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK ENDPOINT — GET /health
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for GET /health — the liveness probe endpoint."""

    def test_health_returns_200(self, fastapi_client):
        """Health endpoint must respond with HTTP 200."""
        response = fastapi_client.get("/health")
        assert response.status_code == 200

    def test_health_returns_correct_body(self, fastapi_client):
        """Health endpoint must return {"status": "healthy"}."""
        response = fastapi_client.get("/health")
        data = response.json()
        assert "status" in data, "Response must contain 'status' key"
        assert data["status"] == "healthy"

    def test_health_content_type_is_json(self, fastapi_client):
        """Health response must be JSON."""
        response = fastapi_client.get("/health")
        assert "application/json" in response.headers["content-type"]


# ─────────────────────────────────────────────────────────────────────────────
# ROOT ENDPOINT — GET /
# ─────────────────────────────────────────────────────────────────────────────


class TestRootEndpoint:
    """Tests for GET / — the root redirect."""

    def test_root_redirects(self, fastapi_client):
        """Root path must redirect to /docs (Swagger UI)."""
        # follow_redirects=False so we can inspect the redirect response itself
        response = fastapi_client.get("/", follow_redirects=False)
        # FastAPI's RedirectResponse defaults to 307 Temporary Redirect
        assert response.status_code in (301, 302, 307, 308), (
            f"Expected a redirect status code, got {response.status_code}"
        )

    def test_root_redirects_to_docs(self, fastapi_client):
        """Root redirect must point to /docs."""
        response = fastapi_client.get("/", follow_redirects=False)
        location = response.headers.get("location", "")
        assert "/docs" in location, f"Expected redirect to /docs, got Location: {location}"


# ─────────────────────────────────────────────────────────────────────────────
# OPENAPI SCHEMA — GET /openapi.json and /docs
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenAPISchema:
    """Tests for API documentation endpoints."""

    def test_openapi_schema_available(self, fastapi_client):
        """OpenAPI JSON schema must be accessible."""
        response = fastapi_client.get("/openapi.json")
        assert response.status_code == 200

    def test_openapi_has_correct_title(self, fastapi_client):
        """OpenAPI schema must use the correct API title."""
        response = fastapi_client.get("/openapi.json")
        schema = response.json()
        assert schema["info"]["title"] == "Blog-to-Podcast Agent API"

    def test_openapi_has_correct_version(self, fastapi_client):
        """OpenAPI schema must declare version 1.0.0."""
        response = fastapi_client.get("/openapi.json")
        schema = response.json()
        assert schema["info"]["version"] == "1.0.0"

    def test_openapi_has_podcast_endpoint(self, fastapi_client):
        """OpenAPI schema must document the /podcast endpoint."""
        response = fastapi_client.get("/openapi.json")
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/podcast" in paths, "OpenAPI schema must document /podcast"
        assert "post" in paths["/podcast"], "/podcast must support POST method"

    def test_openapi_has_health_endpoint(self, fastapi_client):
        """OpenAPI schema must document the /health endpoint."""
        response = fastapi_client.get("/openapi.json")
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/health" in paths, "OpenAPI schema must document /health"

    def test_swagger_ui_available(self, fastapi_client):
        """Swagger UI (/docs) must be accessible."""
        response = fastapi_client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


# ─────────────────────────────────────────────────────────────────────────────
# PODCAST ENDPOINT — POST /podcast (validation layer only)
# ─────────────────────────────────────────────────────────────────────────────


class TestPodcastEndpointValidation:
    """
    Tests for POST /podcast request validation.

    IMPORTANT: We only test HTTP 422 validation errors here.
    We do NOT test the happy path (HTTP 200 with a real script and audio)
    because that would require live Groq, ChromaDB, Ollama, and ElevenLabs.

    These tests verify that FastAPI and Pydantic correctly reject invalid
    requests BEFORE the service layer is ever invoked.
    """

    def test_podcast_rejects_empty_body(self, fastapi_client):
        """POST /podcast without a body must return 422 Unprocessable Entity."""
        response = fastapi_client.post("/podcast")
        assert response.status_code == 422

    def test_podcast_rejects_empty_query(self, fastapi_client):
        """POST /podcast with an empty query string must return 422."""
        response = fastapi_client.post("/podcast", json={"query": ""})
        assert response.status_code == 422

    def test_podcast_rejects_short_query(self, fastapi_client):
        """POST /podcast with query shorter than 3 characters must return 422.

        PodcastRequest.query has min_length=3 in the Pydantic model.
        """
        response = fastapi_client.post("/podcast", json={"query": "AB"})
        assert response.status_code == 422

    def test_podcast_rejects_zero_max_generations(self, fastapi_client):
        """max_generations must be >= 1. Zero must be rejected with 422."""
        response = fastapi_client.post(
            "/podcast", json={"query": "AI Agents", "max_generations": 0}
        )
        assert response.status_code == 422

    def test_podcast_rejects_excessive_max_generations(self, fastapi_client):
        """max_generations must be <= 10. 11 must be rejected with 422."""
        response = fastapi_client.post(
            "/podcast", json={"query": "AI Agents", "max_generations": 11}
        )
        assert response.status_code == 422

    def test_podcast_rejects_wrong_content_type(self, fastapi_client):
        """POST /podcast with non-JSON body must return 422."""
        response = fastapi_client.post(
            "/podcast",
            content="query=AI+Agents",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 422

    def test_podcast_validation_error_has_detail_field(self, fastapi_client):
        """422 responses from FastAPI must include a 'detail' field."""
        response = fastapi_client.post("/podcast", json={"query": "AB"})
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body, "422 response must contain 'detail' field"


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO ENDPOINT — GET /audio/{filename}
# ─────────────────────────────────────────────────────────────────────────────


class TestAudioEndpoint:
    """Tests for GET /audio/{filename} — the audio file serving endpoint."""

    def test_audio_nonexistent_file_returns_404(self, fastapi_client):
        """Requesting a non-existent audio file must return 404 Not Found."""
        response = fastapi_client.get("/audio/nonexistent_podcast_file.mp3")
        assert response.status_code == 404

    def test_audio_path_traversal_is_blocked(self, fastapi_client):
        """Path traversal attempts must be rejected with 400 Bad Request.

        An attacker might try GET /audio/../../../etc/passwd to read
        sensitive files. Our route sanitises filenames using Path.name
        and returns 400 if the cleaned name differs from the input.
        """
        response = fastapi_client.get("/audio/../../../etc/passwd")
        # FastAPI/Starlette may resolve the path before it reaches our handler.
        # Either 400 (our validation) or 404 (file not found after sanitising)
        # is acceptable — both mean the attack was blocked.
        assert response.status_code in (400, 404, 422)

    def test_audio_404_response_has_detail(self, fastapi_client):
        """404 responses for missing audio must include a helpful detail message."""
        response = fastapi_client.get("/audio/missing_file.mp3")
        assert response.status_code == 404
        body = response.json()
        assert "detail" in body
