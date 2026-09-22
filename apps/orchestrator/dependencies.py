"""FastAPI dependencies owned by the Orchestrator application."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from apps.orchestrator.conversation_service import ConversationService
from apps.orchestrator.guardrailer import ContentPolicy, GuardrailsClient
from apps.orchestrator.model_catalog import ModelCatalog
from apps.orchestrator.model_selection import ModelSelector
from apps.orchestrator.workflow.graph import ChatWorkflow
from apps.orchestrator.workflow.nodes import WorkflowNodes
from packages.common.settings import OrchestratorSettings, get_orchestrator_settings
from packages.persistence.database import Database, get_session
from packages.providers.contracts import ProviderClient


async def get_database_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Provide an application-managed database session to an API handler."""
    database: Database = request.app.state.database
    async for session in get_session(database):
        yield session


def get_model_catalog(request: Request) -> ModelCatalog:
    """Provide a catalog backed by the enabled application provider clients."""
    provider_clients: dict[str, ProviderClient] = request.app.state.provider_clients
    return ModelCatalog(provider_clients)


def get_orchestrator_settings_dependency() -> OrchestratorSettings:
    """Provide the typed settings used by Orchestrator-owned services."""
    return get_orchestrator_settings()


async def get_chat_workflow(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_database_session)],
) -> ChatWorkflow:
    """Build one workflow with a request-scoped persistence service."""
    settings = get_orchestrator_settings()
    provider_clients: dict[str, ProviderClient] = request.app.state.provider_clients
    guardrails = GuardrailsClient(
        ContentPolicy(
            input_max_characters=settings.input_max_characters,
            output_max_characters=settings.output_max_characters,
            blocked_terms=settings.configured_blocked_terms,
        )
    )
    nodes = WorkflowNodes(
        guardrails=guardrails,
        model_selector=ModelSelector(
            ModelCatalog(provider_clients), settings.fallback_model
        ),
        conversations=ConversationService(session),
    )
    return ChatWorkflow(nodes)
