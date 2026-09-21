"""Application exceptions raised by all AI provider adapters."""


class ProviderError(Exception):
    """Base exception for failures at the provider boundary."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"{provider}: {message}")


class ProviderUnavailableError(ProviderError):
    """Raised when a provider cannot currently accept requests."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider exceeds its configured timeout."""


class ProviderRateLimitError(ProviderError):
    """Raised when a provider rejects a request due to rate limits."""

    def __init__(self, provider: str, retry_after_seconds: float | None) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(provider, "rate limit exceeded")


class InvalidModelError(ProviderError):
    """Raised when a requested model is unknown to a provider."""

    def __init__(self, provider: str, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(provider, f"unknown model: {model_id}")


class ProviderAuthenticationError(ProviderError):
    """Raised when a provider rejects its configured credentials."""


class ProviderUnexpectedResponseError(ProviderError):
    """Raised when a provider returns an unsupported or malformed response."""
