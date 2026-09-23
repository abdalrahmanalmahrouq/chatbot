"""Orchestrator API application entry point."""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status

from apps.orchestrator.api_models import OrchestrateRequest, OrchestrateResponse
from apps.orchestrator.dependencies import get_chat_workflow, get_model_catalog
from apps.orchestrator.model_catalog import ModelCatalog, OpenAIModel, OpenAIModelList
from apps.orchestrator.workflow.graph import ChatWorkflow
from apps.orchestrator.workflow.models import ChatWorkflowRequest
from packages.common.logging import (
    configure_logging,
    log_event,
    reset_request_context,
    set_request_context,
)
from packages.common.settings import get_orchestrator_settings
from packages.persistence.database import Database
from packages.persistence.migrations import upgrade_database
from packages.providers.factory import close_provider_clients, create_provider_clients

ModelCatalogDependency = Annotated[ModelCatalog, Depends(get_model_catalog)]
ChatWorkflowDependency = Annotated[ChatWorkflow, Depends(get_chat_workflow)]
logger = logging.getLogger("chatbot.orchestrator")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open and close Orchestrator database and provider resources."""
    settings = get_orchestrator_settings()
    configure_logging("orchestrator", settings.log_level)
    await asyncio.to_thread(upgrade_database)

    database = Database(settings.database_url)
    provider_clients = {}

    try:
        await database.check_connection()
        provider_clients = create_provider_clients(settings)
        app.state.database = database
        app.state.provider_clients = provider_clients
        log_event(logger, logging.INFO, "service_started")
        yield
    finally:
        try:
            await close_provider_clients(provider_clients)
        finally:
            await database.dispose()
            log_event(logger, logging.INFO, "service_stopped")


def create_app() -> FastAPI:
    """Create the internal API that delegates work to Orchestrator services."""
    app = FastAPI(title="Chatbot Orchestrator API", lifespan=lifespan)

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        correlation_id = request.headers.get("x-correlation-id", request_id)
        context = set_request_context(request_id, correlation_id)
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "request_failed",
                method=request.method,
                path=request.url.path,
            )
            raise
        else:
            response.headers["x-request-id"] = request_id
            response.headers["x-correlation-id"] = correlation_id
            log_event(
                logger,
                logging.INFO,
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round((time.perf_counter() - started_at) * 1000),
            )
            return response
        finally:
            reset_request_context(context)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "orchestrator"}

    @app.get("/v1/models", response_model=OpenAIModelList, tags=["models"])
    async def list_models(catalog: ModelCatalogDependency) -> OpenAIModelList:
        return await catalog.list_models()

    @app.get("/v1/models/{model_id}", response_model=OpenAIModel, tags=["models"])
    async def get_model(model_id: str, catalog: ModelCatalogDependency) -> OpenAIModel:
        model = await catalog.find_model(model_id)
        if model is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return OpenAIModel(id=model.model.id)

    @app.post("/v1/orchestrate", response_model=OrchestrateResponse, tags=["chat"])
    async def orchestrate(
        body: OrchestrateRequest,
        request: Request,
        workflow: ChatWorkflowDependency,
    ) -> OrchestrateResponse:
        """Delegate one internal chat request to the guarded workflow."""
        workflow_request = ChatWorkflowRequest(
            request_id=request.headers.get("x-request-id", str(uuid.uuid4())),
            correlation_id=request.headers.get(
                "x-correlation-id",
                request.headers.get("x-request-id", str(uuid.uuid4())),
            ),
            messages=tuple(message.to_chat_message() for message in body.messages),
            model=body.model,
            conversation_id=body.conversation_id,
            external_conversation_id=body.external_conversation_id,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
        response = OrchestrateResponse.from_workflow(
            await workflow.run(workflow_request)
        )
        if response.failure is not None:
            log_event(
                logger,
                logging.WARNING,
                "workflow_controlled_failure",
                failure_code=response.failure.code,
            )
        return response

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_orchestrator_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
