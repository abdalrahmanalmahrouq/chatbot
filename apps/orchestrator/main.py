"""Orchestrator API application entry point."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from packages.common.settings import get_orchestrator_settings
from packages.persistence.database import Database
from packages.persistence.migrations import upgrade_database
from packages.providers.factory import close_provider_clients, create_provider_clients


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open and close Orchestrator database and provider resources."""
    settings = get_orchestrator_settings()
    await asyncio.to_thread(upgrade_database)

    database = Database(settings.database_url)
    provider_clients = {}

    try:
        await database.check_connection()
        provider_clients = create_provider_clients(settings)
        app.state.database = database
        app.state.provider_clients = provider_clients
        yield
    finally:
        try:
            await close_provider_clients(provider_clients)
        finally:
            await database.dispose()


def create_app() -> FastAPI:
    """Create the Orchestrator API without adding workflow logic prematurely."""
    app = FastAPI(title="Chatbot Orchestrator API", lifespan=lifespan)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "orchestrator"}

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_orchestrator_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
