"""Orchestrator API application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI

from packages.common.settings import get_orchestrator_settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Validate configuration before accepting Orchestrator API requests."""
    get_orchestrator_settings()
    yield


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
