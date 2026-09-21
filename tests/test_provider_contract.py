"""Tests for the provider-neutral client contract."""

import asyncio
from collections.abc import AsyncIterator

from packages.providers import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatMessage,
    ProviderClient,
    ProviderModel,
)


class FakeProviderClient:
    name = "fake"

    async def list_models(self) -> list[ProviderModel]:
        return [ProviderModel(id="fake-model", name="Fake", provider=self.name)]

    async def get_model(self, model_id: str) -> ProviderModel:
        return ProviderModel(id=model_id, name="Fake", provider=self.name)

    async def complete_chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        return ChatCompletion(
            id="fake-completion",
            model=request.model,
            message=ChatMessage(role="assistant", content="hello"),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(
            id="fake-completion", model=request.model, content="hello"
        )

    async def close(self) -> None:
        return None


def test_fake_can_implement_provider_contract() -> None:
    fake = FakeProviderClient()
    request = ChatCompletionRequest(
        model="fake-model",
        messages=(ChatMessage(role="user", content="Hi"),),
    )

    completion = asyncio.run(fake.complete_chat(request))

    assert isinstance(fake, ProviderClient)
    assert completion.message.content == "hello"
