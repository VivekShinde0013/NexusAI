import os
from app.config import get_settings, Settings


def test_settings_default(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("EVAL_QUALITY_API_KEY", "")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.APP_NAME == "NexusAI"
    assert settings.APP_VERSION == "0.1.0"
    assert settings.ENVIRONMENT == "development"
    assert settings.DEBUG is True
    assert settings.LLM_API_KEY == ""
    assert settings.LLM_MODEL in ["", "gpt-4o-mini", "gemini-2.5-flash"]


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "OverrideName")
    monkeypatch.setenv("APP_VERSION", "2.0.0")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("LLM_API_KEY", "secretkey")
    monkeypatch.setenv("LLM_MODEL", "gpt-4")

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.APP_NAME == "OverrideName"
    assert settings.APP_VERSION == "2.0.0"
    assert settings.ENVIRONMENT == "production"
    assert settings.DEBUG is False
    assert settings.LLM_API_KEY == "secretkey"
    assert settings.LLM_MODEL == "gpt-4"

    get_settings.cache_clear()
