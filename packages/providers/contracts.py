"""Provider-neutral types and the common AI provider contract."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class ChatCompletionRequest:
    model: str
    messages: tuple[ChatMessage, ...]
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ProviderModel:
    id: str
    name: str
    provider: str
    context_length: int | None = None


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    id: str
    model: str
    message: ChatMessage
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class ChatCompletionChunk:
    id: str
    model: str
    content: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@runtime_checkable
class ProviderClient(Protocol):
    """The behavior every AI provider adapter must expose."""

    name: str

    async def list_models(self) -> list[ProviderModel]: ...

    async def get_model(self, model_id: str) -> ProviderModel: ...

    async def complete_chat(self, request: ChatCompletionRequest) -> ChatCompletion: ...

    def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def close(self) -> None: ...
