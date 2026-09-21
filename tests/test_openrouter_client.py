"""Behavior tests for the OpenRouter provider adapter."""

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from packages.providers import (
    ChatCompletionRequest,
    ChatMessage,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnexpectedResponseError,
)
from packages.providers.openrouter import OpenRouterClient


def test_lists_retrieves_and_completes_with_normalized_types() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"

        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "example/model",
                            "name": "Example Model",
                            "context_length": 8192,
                        }
                    ]
                },
            )
        if request.url.path == "/v1/model/example/model":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "example/model",
                        "name": "Example Model",
                        "context_length": 8192,
                    }
                },
            )

        assert request.url.path == "/v1/chat/completions"
        request_body = json.loads(request.content)
        assert request_body["stream"] is False
        return httpx.Response(
            200,
            json={
                "id": "completion-1",
                "model": "example/model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            },
        )

    async def scenario() -> None:
        client = _client(handler)
        request = _request()

        models = await client.list_models()
        model = await client.get_model("example/model")
        completion = await client.complete_chat(request)
        await client.close()

        assert models == [model]
        assert model.provider == "openrouter"
        assert completion.message.content == "Hello"
        assert completion.usage is not None
        assert completion.usage.total_tokens == 4

    asyncio.run(scenario())


def test_streams_normalized_chat_chunks() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        first_chunk = {
            "id": "stream-1",
            "model": "example/model",
            "choices": [{"delta": {"content": "Hel"}, "finish_reason": None}],
        }
        final_chunk = {
            "id": "stream-1",
            "model": "example/model",
            "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}],
        }
        stream = "\n".join(
            [
                f"data: {json.dumps(first_chunk)}",
                f"data: {json.dumps(final_chunk)}",
                "data: [DONE]",
            ]
        )
        return httpx.Response(200, text=stream)

    async def scenario() -> None:
        client = _client(handler)
        chunks = [chunk async for chunk in client.stream_chat(_request())]
        await client.close()

        assert [chunk.content for chunk in chunks] == ["Hel", "lo"]
        assert chunks[-1].finish_reason == "stop"

    asyncio.run(scenario())


def test_translates_rate_limits_without_exposing_response_body() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "2"},
            text="secret upstream detail",
        )

    async def scenario() -> None:
        client = _client(handler)

        with pytest.raises(ProviderRateLimitError) as error:
            await client.complete_chat(_request())

        await client.close()
        assert error.value.retry_after_seconds == 2
        assert "secret upstream detail" not in str(error.value)

    asyncio.run(scenario())


def test_rejects_malformed_provider_response() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    async def scenario() -> None:
        client = _client(handler)

        with pytest.raises(ProviderUnexpectedResponseError):
            await client.complete_chat(_request())

        await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("transport_error", "expected_error"),
    [
        (httpx.ReadTimeout, ProviderTimeoutError),
        (httpx.ConnectError, ProviderUnavailableError),
    ],
)
def test_translates_transport_errors(
    transport_error: type[httpx.RequestError],
    expected_error: type[Exception],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise transport_error("provider failure", request=request)

    async def scenario() -> None:
        client = _client(handler)

        with pytest.raises(expected_error):
            await client.complete_chat(_request())

        await client.close()

    asyncio.run(scenario())


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OpenRouterClient:
    return OpenRouterClient(
        api_key="test-key",
        base_url="https://openrouter.example/v1",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="example/model",
        messages=(ChatMessage(role="user", content="Hi"),),
    )
