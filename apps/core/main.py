"""Core API application entry point."""

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from httpx import RequestError, TimeoutException

from apps.core.api_models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    OpenAIChatMessage,
)
from apps.core.orchestrator_client import (
    OrchestratorClient,
    OrchestratorResponseError,
)
from packages.common.settings import get_core_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate configuration before accepting Core API requests."""
    settings = get_core_settings()
    client = OrchestratorClient(
        base_url=str(settings.orchestrator_url),
        timeout_seconds=settings.request_timeout_seconds,
    )
    app.state.orchestrator_client = client
    try:
        yield
    finally:
        await client.close()


def get_orchestrator_client(request: Request) -> OrchestratorClient:
    return request.app.state.orchestrator_client


OrchestratorClientDependency = Annotated[
    OrchestratorClient, Depends(get_orchestrator_client)
]


def create_app() -> FastAPI:
    """Create the Core API without embedding gateway logic in route handlers."""
    app = FastAPI(title="Chatbot Core API", lifespan=lifespan)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "core"}

    @app.get("/v1/models", tags=["models"])
    async def list_models(
        client: OrchestratorClientDependency,
    ) -> dict[str, Any]:
        return await _call_orchestrator(client.list_models())

    @app.get("/v1/models/{model_id}", tags=["models"])
    async def get_model(
        model_id: str, client: OrchestratorClientDependency
    ) -> dict[str, Any]:
        return await _call_orchestrator(client.get_model(model_id))

    @app.post(
        "/v1/chat/completions", response_model=ChatCompletionResponse, tags=["chat"]
    )
    async def create_chat_completion(
        body: ChatCompletionRequest,
        request: Request,
        client: OrchestratorClientDependency,
    ) -> ChatCompletionResponse:
        # Open WebUI requests streaming by default. Until the provider-to-client
        # streaming path is added, complete the request normally instead.
        response = await _call_orchestrator(
            client.orchestrate(
                {
                    "model": body.model,
                    "messages": [message.model_dump() for message in body.messages],
                    "temperature": body.temperature,
                    "max_tokens": body.max_tokens,
                    "external_conversation_id": request.headers.get(
                        "x-openwebui-chat-id"
                    ),
                },
                _forwarded_headers(request),
            )
        )
        if not response["accepted"]:
            failure = response.get("failure") or {}
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=failure.get("message", "request was rejected"),
            )
        return ChatCompletionResponse(
            id=f"chatcmpl-{response['run_id']}",
            created=int(time.time()),
            model=response["model"],
            choices=[
                ChatCompletionChoice(
                    message=OpenAIChatMessage(content=response["content"])
                )
            ],
        )

    return app


app = create_app()


def _forwarded_headers(request: Request) -> dict[str, str]:
    """Forward correlation identifiers only; credentials never cross services."""
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    correlation_id = request.headers.get("x-correlation-id", request_id)
    headers = {
        "x-request-id": request_id,
        "x-correlation-id": correlation_id,
    }
    for header_name in (
        "x-openwebui-chat-id",
        "x-openwebui-user-id",
        "x-openwebui-task-id",
    ):
        header_value = request.headers.get(header_name)
        if header_value:
            headers[header_name] = header_value
    return headers


async def _call_orchestrator(awaitable):
    """Translate network failures without exposing client implementation details."""
    try:
        return await awaitable
    except TimeoutException as error:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="orchestrator request timed out",
        ) from error
    except RequestError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="orchestrator is unavailable",
        ) from error
    except OrchestratorResponseError as error:
        raise HTTPException(
            status_code=error.status_code, detail=error.detail
        ) from error


if __name__ == "__main__":
    settings = get_core_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
