"""Unit tests for transactional conversation writes without a live database."""

import asyncio
import uuid
from typing import cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from apps.orchestrator.conversation_service import (
    ChatRunResult,
    ConversationService,
    ConversationWriteConflictError,
)
from packages.persistence.models import Conversation, Message, MessageRole, RunStatus


class FakeTransaction:
    def __init__(self) -> None:
        self.exception: BaseException | None = None

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> bool:
        self.exception = exception
        return False


class FakeSession:
    def __init__(self, flush_error: Exception | None = None) -> None:
        self.transaction = FakeTransaction()
        self.flush_error = flush_error
        self.added: list[object] = []

    def begin(self) -> FakeTransaction:
        return self.transaction

    def add_all(self, objects: list[object]) -> None:
        self.added.extend(objects)

    async def flush(self) -> None:
        if self.flush_error is not None:
            raise self.flush_error


def test_store_exchange_adds_ordered_messages_and_run_in_one_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation_id = uuid.uuid4()
    conversation = Conversation(id=conversation_id)
    fake_session = FakeSession()
    service = ConversationService(cast(AsyncSession, fake_session))
    monkeypatch.setattr(
        service,
        "_locked_conversation",
        AsyncMock(return_value=conversation),
    )
    monkeypatch.setattr(service, "_ensure_request_is_new", AsyncMock())
    monkeypatch.setattr(service, "_next_sequence_number", AsyncMock(return_value=4))

    run = asyncio.run(
        service.store_exchange(
            conversation_id=conversation_id,
            user_content="Hello",
            assistant_content="Hi there",
            run_result=ChatRunResult(
                request_id="request-1",
                correlation_id="correlation-1",
                status=RunStatus.COMPLETED,
                provider="openrouter",
                model="example/model",
                input_tokens=3,
                output_tokens=2,
            ),
        )
    )

    messages = [item for item in fake_session.added if isinstance(item, Message)]
    assert [(message.role, message.sequence_number) for message in messages] == [
        (MessageRole.USER, 4),
        (MessageRole.ASSISTANT, 5),
    ]
    assert run.request_id == "request-1"
    assert fake_session.transaction.exception is None


def test_store_exchange_reports_database_write_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation_id = uuid.uuid4()
    flush_error = IntegrityError("insert", {}, RuntimeError("duplicate"))
    fake_session = FakeSession(flush_error=flush_error)
    service = ConversationService(cast(AsyncSession, fake_session))
    monkeypatch.setattr(
        service,
        "_locked_conversation",
        AsyncMock(return_value=Conversation(id=conversation_id)),
    )
    monkeypatch.setattr(service, "_ensure_request_is_new", AsyncMock())
    monkeypatch.setattr(service, "_next_sequence_number", AsyncMock(return_value=0))

    with pytest.raises(ConversationWriteConflictError):
        asyncio.run(
            service.store_exchange(
                conversation_id=conversation_id,
                user_content="Hello",
                assistant_content=None,
                run_result=ChatRunResult(
                    request_id="request-1",
                    correlation_id="correlation-1",
                    status=RunStatus.FAILED,
                    error_code="provider_unavailable",
                ),
            )
        )

    assert isinstance(fake_session.transaction.exception, IntegrityError)
