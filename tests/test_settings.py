"""Phase 1 configuration tests."""

import pytest

from packages.common.settings import (
    ConfigurationError,
    CoreSettings,
    get_core_settings,
    get_orchestrator_settings,
)


def test_core_settings_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_ORCHESTRATOR_URL", "http://localhost:8001")
    get_core_settings.cache_clear()

    settings = get_core_settings()

    assert settings.port == 8000
    assert str(settings.orchestrator_url) == "http://localhost:8001/"


def test_orchestrator_settings_load_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ORCHESTRATOR_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/db"
    )
    monkeypatch.setenv("ORCHESTRATOR_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv(
        "ORCHESTRATOR_OPENROUTER_BASE_URL", "https://api.example.test/v1"
    )
    monkeypatch.setenv("ORCHESTRATOR_OPENROUTER_API_KEY", "test-provider-key")
    monkeypatch.setenv("ORCHESTRATOR_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("ORCHESTRATOR_OLLAMA_ENABLED", "true")
    get_orchestrator_settings.cache_clear()

    settings = get_orchestrator_settings()

    assert settings.port == 8001
    assert str(settings.ollama_base_url) == "http://localhost:11434/"
    assert settings.ollama_enabled is True


def test_missing_required_core_value_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(CoreSettings.model_config, "env_file", None)
    monkeypatch.delenv("CORE_ORCHESTRATOR_URL", raising=False)
    get_core_settings.cache_clear()

    with pytest.raises(ConfigurationError, match="Core configuration is invalid"):
        get_core_settings()
