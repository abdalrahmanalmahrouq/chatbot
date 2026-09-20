"""Run the checked-in Alembic migrations for the Orchestrator database."""

from pathlib import Path

from alembic import command
from alembic.config import Config


def upgrade_database() -> None:
    """Upgrade the database to the latest explicit migration revision."""
    repository_root = Path(__file__).resolve().parents[2]
    config = Config(str(repository_root / "alembic.ini"))
    command.upgrade(config, "head")
