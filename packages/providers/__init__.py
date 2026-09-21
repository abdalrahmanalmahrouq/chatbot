"""Provider-neutral AI client contracts and concrete adapters."""

from packages.providers.contracts import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatMessage,
    ProviderClient,
    ProviderModel,
    TokenUsage,
)
from packages.providers.exceptions import (
    InvalidModelError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnexpectedResponseError,
)

__all__ = [
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionRequest",
    "ChatMessage",
    "InvalidModelError",
    "ProviderAuthenticationError",
    "ProviderClient",
    "ProviderError",
    "ProviderModel",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "ProviderUnexpectedResponseError",
    "TokenUsage",
]
