"""FastAPI dependencies owned by the Orchestrator application."""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from packages.persistence.database import Database, get_session


async def get_database_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Provide an application-managed database session to an API handler."""
    database: Database = request.app.state.database
    async for session in get_session(database):
        yield session
