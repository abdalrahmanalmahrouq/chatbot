"""FastAPI dependencies owned by the Orchestrator application."""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from apps.orchestrator.model_catalog import ModelCatalog
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
