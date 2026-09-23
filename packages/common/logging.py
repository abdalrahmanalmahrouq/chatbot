"""Small structured logging helpers shared by the two API applications."""

import json
import logging
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class JsonFormatter(logging.Formatter):
    """Render only intentionally supplied fields as one JSON log line."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "service": self._service,
            "event": record.getMessage(),
            "request_id": _request_id.get(),
            "correlation_id": _correlation_id.get(),
        }
        payload.update(getattr(record, "event_fields", {}))
        return json.dumps(payload, default=str)


def configure_logging(service: str, log_level: str) -> None:
    """Configure the service logger once without changing third-party loggers."""
    logger = logging.getLogger(f"chatbot.{service}")
    logger.setLevel(log_level)
    logger.propagate = False
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(service))
    logger.addHandler(handler)


def set_request_context(request_id: str, correlation_id: str) -> tuple[Token, Token]:
    """Set correlation values for all application logs during one request."""
    return _request_id.set(request_id), _correlation_id.set(correlation_id)


def reset_request_context(tokens: tuple[Token, Token]) -> None:
    """Clear request-specific context after a request is complete."""
    request_token, correlation_token = tokens
    _request_id.reset(request_token)
    _correlation_id.reset(correlation_token)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: object,
) -> None:
    """Log known-safe metadata without prompts, headers, or credentials."""
    logger.log(level, event, extra={"event_fields": fields})
