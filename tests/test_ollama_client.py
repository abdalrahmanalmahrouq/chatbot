"""Behavior tests for the Ollama provider adapter."""

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from packages.common.settings import OrchestratorSettings
from packages.providers import ChatCompletionRequest, ChatMessage, InvalidModelError
from packages.providers.factory import close_provider_clients, create_provider_clients
from packages.providers.ollama import OllamaClient


def test_lists_retrieves_and_completes_with_normalized_types() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "llama3.2", "model": "llama3.2"}]},
            )
        if request.url.path == "/api/show":
            assert json.loads(request.content) == {"model": "llama3.2"}
            return httpx.Response(200, json={"details": {"family": "llama"}})

        request_body = json.loads(request.content)
        assert request_body["stream"] is False
        return httpx.Response(
            200,
            json={
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "Hello"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3,
                "eval_count": 1,
            },
        )

    async def scenario() -> None:
        client = _client(handler)

        models = await client.list_models()
        model = await client.get_model("llama3.2")
        completion = await client.complete_chat(_request())
        await client.close()

        assert models == [model]
        assert model.provider == "ollama"
        assert completion.message.content == "Hello"
        assert completion.usage is not None
        assert completion.usage.total_tokens == 4

    asyncio.run(scenario())


def test_streams_newline_delimited_chat_chunks() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        stream = "\n".join(
            [
                '{"model":"llama3.2","message":{"role":"assistant","content":"Hel"},"done":false}',
                '{"model":"llama3.2","message":{"role":"assistant","content":"lo"},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":1}',
            ]
        )
        return httpx.Response(200, text=stream)

    async def scenario() -> None:
        client = _client(handler)
        chunks = [chunk async for chunk in client.stream_chat(_request())]
        await client.close()

        assert [chunk.content for chunk in chunks] == ["Hel", "lo"]
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.total_tokens == 4

    asyncio.run(scenario())


def test_disabled_ollama_is_not_created() -> None:
    settings = OrchestratorSettings(
        database_url="postgresql+asyncpg://user:password@postgres/chatbot",
        redis_url="redis://redis:6379/0",
        openrouter_base_url="https://openrouter.example/v1",
        openrouter_api_key="test-key",
        ollama_enabled=False,
    )

    clients = create_provider_clients(settings)

    assert set(clients) == {"openrouter"}
    asyncio.run(close_provider_clients(clients))


def test_translates_unknown_models() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    async def scenario() -> None:
        client = _client(handler)

        with pytest.raises(InvalidModelError):
            await client.get_model("missing-model")

        await client.close()

    asyncio.run(scenario())


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaClient:
    return OllamaClient(
        base_url="http://ollama.example",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="llama3.2",
        messages=(ChatMessage(role="user", content="Hi"),),
    )
