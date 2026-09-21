"""Typed environment configuration shared by the two service applications."""

from functools import lru_cache

from pydantic import (
    AnyHttpUrl,
    Field,
    PositiveFloat,
    SecretStr,
    ValidationError,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when a service cannot load a complete environment configuration."""


class _Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class CoreSettings(_Settings):
    """Configuration owned by the public Core API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CORE_",
        extra="ignore",
    )

    port: int = Field(default=8000, ge=1, le=65535)
    orchestrator_url: AnyHttpUrl
    request_timeout_seconds: PositiveFloat = 30


class OrchestratorSettings(_Settings):
    """Configuration owned by the private Orchestrator API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ORCHESTRATOR_",
        extra="ignore",
    )

    port: int = Field(default=8001, ge=1, le=65535)
    database_url: str
    redis_url: str
    openrouter_base_url: AnyHttpUrl
    openrouter_api_key: SecretStr
    ollama_base_url: AnyHttpUrl = Field(
        default="http://localhost:11434", validate_default=True
    )
    ollama_enabled: bool = False
    provider_timeout_seconds: PositiveFloat = 60
    fallback_model: str | None = None

    @field_validator("fallback_model", mode="before")
    @classmethod
    def blank_fallback_model_is_not_configured(cls, value: object) -> object:
        """Treat an empty environment variable as an absent fallback."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


def _load(settings_type: type[CoreSettings] | type[OrchestratorSettings]):
    try:
        return settings_type()
    except ValidationError as error:
        missing = [
            str(item["loc"][0]) for item in error.errors() if item["type"] == "missing"
        ]
        detail = ", ".join(missing) or "invalid values"
        service = settings_type.__name__.removesuffix("Settings")
        raise ConfigurationError(
            f"{service} configuration is invalid: {detail}. "
            "Set the values in .env (see .env.example)."
        ) from error


@lru_cache
def get_core_settings() -> CoreSettings:
    """Load and cache Core configuration for dependency injection."""
    return _load(CoreSettings)


@lru_cache
def get_orchestrator_settings() -> OrchestratorSettings:
    """Load and cache Orchestrator configuration for dependency injection."""
    return _load(OrchestratorSettings)
