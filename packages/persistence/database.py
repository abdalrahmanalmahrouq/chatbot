"""Async SQLAlchemy engine and session management."""

from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


class Database:
    """Own the Orchestrator database engine and its async session factory."""

    def __init__(self, database_url: str) -> None:
        self.engine = create_async_engine(database_url, pool_pre_ping=True)
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def check_connection(self) -> None:
        """Verify that PostgreSQL accepts async connections."""
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def dispose(self) -> None:
        """Close all connections in the engine pool."""
        await self.engine.dispose()


async def get_session(database: Database) -> AsyncIterator[AsyncSession]:
    """Yield a session for a caller that supplies the application database."""
    async with database.session_factory() as session:
        yield session
