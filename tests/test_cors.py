import pytest
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware
from app.main import app
from app.config import get_settings


def test_cors_middleware_is_registered():
    """Test 1: Verify CORSMiddleware is registered on the FastAPI application."""
    middleware_types = [m.cls for m in app.user_middleware]
    assert CORSMiddleware in middleware_types


def test_cors_allowed_origin_header_accepted():
    """Test 2: Configured allowed origin receives Access-Control-Allow-Origin response header."""
    client = TestClient(app)
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
    }
    response = client.options("/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_disallowed_origin_not_granted():
    """Test 3: Unapproved origin is not granted CORS permission."""
    client = TestClient(app)
    headers = {
        "Origin": "http://unapproved-malicious-site.com",
        "Access-Control-Request-Method": "GET",
    }
    response = client.options("/health", headers=headers)
    assert response.headers.get("access-control-allow-origin") is None


def test_health_endpoint_unchanged():
    """Test 4: Existing /health endpoint behavior remains unchanged."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "NexusAI"}


def test_cors_configurable_via_env(monkeypatch):
    """Test 5: ALLOWED_ORIGINS setting parses comma-separated environment variable values cleanly."""
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "http://frontend.company.com, https://app.nexusai.internal",
    )
    get_settings.cache_clear()
    settings = get_settings()

    origins = settings.get_allowed_origins()
    assert origins == [
        "http://frontend.company.com",
        "https://app.nexusai.internal",
    ]

    # Reset cache
    get_settings.cache_clear()
