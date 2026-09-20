"""Core API application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI

from packages.common.settings import get_core_settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Validate configuration before accepting Core API requests."""
    get_core_settings()
    yield


def create_app() -> FastAPI:
    """Create the Core API without embedding gateway logic in route handlers."""
    app = FastAPI(title="Chatbot Core API", lifespan=lifespan)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "core"}

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_core_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
