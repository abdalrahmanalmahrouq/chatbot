"""Shared HTTP failure translation for provider adapters."""

import httpx

from packages.providers.exceptions import (
    InvalidModelError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderUnexpectedResponseError,
)


def raise_for_provider_status(
    response: httpx.Response,
    provider: str,
    model_id: str | None = None,
) -> None:
    """Translate an HTTP failure without exposing its raw response body."""
    if response.is_success:
        return

    if response.status_code in {401, 403}:
        raise ProviderAuthenticationError(provider, "authentication failed")

    if response.status_code == 404 and model_id is not None:
        raise InvalidModelError(provider, model_id)

    if response.status_code == 429:
        raise ProviderRateLimitError(provider, _retry_after(response))

    if response.status_code >= 500:
        raise ProviderUnavailableError(provider, "service unavailable")

    raise ProviderError(provider, f"request failed with HTTP {response.status_code}")


def parse_json_object(response: httpx.Response, provider: str) -> dict[str, object]:
    """Decode a JSON object or raise a stable application exception."""
    try:
        payload = response.json()
    except ValueError as error:
        raise ProviderUnexpectedResponseError(
            provider, "response was not valid JSON"
        ) from error

    if not isinstance(payload, dict):
        raise ProviderUnexpectedResponseError(
            provider, "response must be a JSON object"
        )

    return payload


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None
