"""OpenRouter adapter using its OpenAI-compatible HTTP API."""

import json
from collections.abc import AsyncIterator
from typing import Never
from urllib.parse import quote

import httpx

from packages.providers.contracts import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatMessage,
    ProviderModel,
    TokenUsage,
)
from packages.providers.exceptions import (
    InvalidModelError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnexpectedResponseError,
)
from packages.providers.http import parse_json_object, raise_for_provider_status


class OpenRouterClient:
    """Normalize OpenRouter models and chat completions for the application."""

    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    async def list_models(self) -> list[ProviderModel]:
        response = await self._request("GET", "/models")
        payload = parse_json_object(response, self.name)
        models = payload.get("data")

        if not isinstance(models, list):
            self._unexpected("model list is missing")

        return [self._parse_model(model) for model in models]

    async def get_model(self, model_id: str) -> ProviderModel:
        parts = model_id.split("/", maxsplit=1)
        if len(parts) != 2 or not all(parts):
            raise InvalidModelError(self.name, model_id)

        author = quote(parts[0], safe="")
        slug = quote(parts[1], safe=":")
        response = await self._request(
            "GET", f"/model/{author}/{slug}", model_id=model_id
        )
        payload = parse_json_object(response, self.name)
        return self._parse_model(payload.get("data"))

    async def complete_chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        response = await self._request(
            "POST",
            "/chat/completions",
            model_id=request.model,
            json_body=self._chat_payload(request, stream=False),
        )
        payload = parse_json_object(response, self.name)
        choice = self._first_choice(payload)
        message = choice.get("message")

        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            self._unexpected("chat response message is missing")

        return ChatCompletion(
            id=self._required_string(payload, "id"),
            model=self._required_string(payload, "model"),
            message=ChatMessage(role="assistant", content=message["content"]),
            finish_reason=self._optional_string(choice, "finish_reason"),
            usage=self._parse_usage(payload.get("usage")),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=self._chat_payload(request, stream=True),
            ) as response:
                raise_for_provider_status(response, self.name, request.model)

                async for line in response.aiter_lines():
                    chunk = self._parse_stream_line(line)
                    if chunk is not None:
                        yield chunk
        except httpx.TimeoutException as error:
            raise ProviderTimeoutError(self.name, "request timed out") from error
        except httpx.RequestError as error:
            raise ProviderUnavailableError(
                self.name, "could not reach provider"
            ) from error

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        model_id: str | None = None,
        json_body: dict[str, object] | None = None,
    ) -> httpx.Response:
        try:
            response = await self._client.request(method, path, json=json_body)
        except httpx.TimeoutException as error:
            raise ProviderTimeoutError(self.name, "request timed out") from error
        except httpx.RequestError as error:
            raise ProviderUnavailableError(
                self.name, "could not reach provider"
            ) from error

        raise_for_provider_status(response, self.name, model_id)
        return response

    def _chat_payload(
        self, request: ChatCompletionRequest, stream: bool
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "stream": stream,
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        return payload

    def _parse_stream_line(self, line: str) -> ChatCompletionChunk | None:
        if not line.startswith("data:"):
            return None

        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            return None

        try:
            payload = json.loads(data)
        except json.JSONDecodeError as error:
            raise ProviderUnexpectedResponseError(
                self.name, "stream contained invalid JSON"
            ) from error

        if not isinstance(payload, dict):
            self._unexpected("stream chunk must be a JSON object")

        choice = self._first_choice(payload)
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            self._unexpected("stream chunk delta is missing")

        content = delta.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            self._unexpected("stream chunk content is invalid")

        return ChatCompletionChunk(
            id=self._required_string(payload, "id"),
            model=self._required_string(payload, "model"),
            content=content,
            finish_reason=self._optional_string(choice, "finish_reason"),
            usage=self._parse_usage(payload.get("usage")),
        )

    def _parse_model(self, value: object) -> ProviderModel:
        if not isinstance(value, dict):
            self._unexpected("model entry is invalid")

        model_id = self._required_string(value, "id")
        name = value.get("name", model_id)
        if not isinstance(name, str):
            self._unexpected("model name is invalid")

        context_length = value.get("context_length")
        if context_length is not None and not self._is_integer(context_length):
            self._unexpected("model context length is invalid")

        return ProviderModel(
            id=model_id,
            name=name,
            provider=self.name,
            context_length=context_length,
        )

    def _parse_usage(self, value: object) -> TokenUsage | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            self._unexpected("token usage is invalid")

        input_tokens = value.get("prompt_tokens")
        output_tokens = value.get("completion_tokens")
        total_tokens = value.get("total_tokens")
        if not all(
            self._is_integer(count)
            for count in (input_tokens, output_tokens, total_tokens)
        ):
            self._unexpected("token usage is invalid")

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def _first_choice(self, payload: dict[str, object]) -> dict[str, object]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            self._unexpected("chat response choices are missing")

        choice = choices[0]
        if not isinstance(choice, dict):
            self._unexpected("chat response choice is invalid")

        return choice

    def _required_string(self, value: dict[str, object], key: str) -> str:
        result = value.get(key)
        if not isinstance(result, str):
            self._unexpected(f"response field '{key}' is invalid")
        return result

    def _optional_string(self, value: dict[str, object], key: str) -> str | None:
        result = value.get(key)
        if result is not None and not isinstance(result, str):
            self._unexpected(f"response field '{key}' is invalid")
        return result

    @staticmethod
    def _is_integer(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def _unexpected(self, message: str) -> Never:
        raise ProviderUnexpectedResponseError(self.name, message)
