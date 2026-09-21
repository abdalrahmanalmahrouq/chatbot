"""Ollama adapter using its native HTTP API."""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Never

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
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnexpectedResponseError,
)
from packages.providers.http import parse_json_object, raise_for_provider_status


class OllamaClient:
    """Normalize locally hosted Ollama models and chat completions."""

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    async def list_models(self) -> list[ProviderModel]:
        response = await self._request("GET", "/api/tags")
        payload = parse_json_object(response, self.name)
        models = payload.get("models")

        if not isinstance(models, list):
            self._unexpected("model list is missing")

        return [self._parse_listed_model(model) for model in models]

    async def get_model(self, model_id: str) -> ProviderModel:
        response = await self._request(
            "POST",
            "/api/show",
            model_id=model_id,
            json_body={"model": model_id},
        )
        parse_json_object(response, self.name)
        return ProviderModel(id=model_id, name=model_id, provider=self.name)

    async def complete_chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        response = await self._request(
            "POST",
            "/api/chat",
            model_id=request.model,
            json_body=self._chat_payload(request, stream=False),
        )
        payload = parse_json_object(response, self.name)
        message = payload.get("message")

        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            self._unexpected("chat response message is missing")

        return ChatCompletion(
            id=self._completion_id(),
            model=self._required_string(payload, "model"),
            message=ChatMessage(role="assistant", content=message["content"]),
            finish_reason=self._optional_string(payload, "done_reason"),
            usage=self._parse_usage(payload),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        completion_id = self._completion_id()

        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json=self._chat_payload(request, stream=True),
            ) as response:
                raise_for_provider_status(response, self.name, request.model)

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    yield self._parse_stream_line(line, completion_id)
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
        options: dict[str, object] = {}

        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options

        return payload

    def _parse_stream_line(self, line: str, completion_id: str) -> ChatCompletionChunk:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProviderUnexpectedResponseError(
                self.name, "stream contained invalid JSON"
            ) from error

        if not isinstance(payload, dict):
            self._unexpected("stream chunk must be a JSON object")

        provider_error = payload.get("error")
        if provider_error is not None:
            self._unexpected("provider returned a streaming error")

        message = payload.get("message")
        if not isinstance(message, dict):
            self._unexpected("stream chunk message is missing")

        content = message.get("content", "")
        if not isinstance(content, str):
            self._unexpected("stream chunk content is invalid")

        done = payload.get("done", False)
        if not isinstance(done, bool):
            self._unexpected("stream completion state is invalid")

        finish_reason = self._optional_string(payload, "done_reason") if done else None
        usage = self._parse_usage(payload) if done else None

        return ChatCompletionChunk(
            id=completion_id,
            model=self._required_string(payload, "model"),
            content=content,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _parse_listed_model(self, value: object) -> ProviderModel:
        if not isinstance(value, dict):
            self._unexpected("model entry is invalid")

        model_id = value.get("name", value.get("model"))
        if not isinstance(model_id, str):
            self._unexpected("model identifier is missing")

        return ProviderModel(id=model_id, name=model_id, provider=self.name)

    def _parse_usage(self, payload: dict[str, object]) -> TokenUsage | None:
        input_tokens = payload.get("prompt_eval_count")
        output_tokens = payload.get("eval_count")

        if input_tokens is None and output_tokens is None:
            return None
        if not self._is_integer(input_tokens) or not self._is_integer(output_tokens):
            self._unexpected("token usage is invalid")

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

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
    def _completion_id() -> str:
        return f"ollama-{uuid.uuid4()}"

    @staticmethod
    def _is_integer(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def _unexpected(self, message: str) -> Never:
        raise ProviderUnexpectedResponseError(self.name, message)
