"""Transactional conversation and chat-run persistence for Orchestrator."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from packages.persistence.models import (
    Conversation,
    Message,
    MessageRole,
    OrchestrationRun,
    RunStatus,
)


class ConversationConflictError(RuntimeError):
    """Raised when creating a conversation with an existing external ID."""

    def __init__(self, external_id: str) -> None:
        self.external_id = external_id
        super().__init__(f"conversation already exists for external ID: {external_id}")


class ConversationNotFoundError(LookupError):
    """Raised when a requested conversation identifier does not exist."""


class ConversationIdentifierError(ValueError):
    """Raised when a continuation request has ambiguous identifiers."""


class ChatRunConflictError(RuntimeError):
    """Raised when a chat request identifier has already been recorded."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(f"chat request was already recorded: {request_id}")


class ConversationWriteConflictError(RuntimeError):
    """Raised when concurrent writes violate a conversation database constraint."""


@dataclass(frozen=True, slots=True)
class ChatRunResult:
    """The non-message outcome to store for one chat request."""

    request_id: str
    correlation_id: str
    status: RunStatus
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and value < 0:
                raise ValueError("token counts cannot be negative")


class ConversationService:
    """Manage conversation history and run records through an async session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_conversation(self, external_id: str | None = None) -> Conversation:
        """Create a conversation, rejecting a reused external identifier."""
        try:
            async with self._session.begin():
                if external_id is not None:
                    existing_conversation = await self._session.scalar(
                        select(Conversation.id).where(
                            Conversation.external_id == external_id
                        )
                    )
                    if existing_conversation is not None:
                        raise ConversationConflictError(external_id)

                conversation = Conversation(external_id=external_id)
                self._session.add(conversation)
                await self._session.flush()
                return conversation
        except IntegrityError as error:
            if external_id is not None:
                raise ConversationConflictError(external_id) from error
            raise

    async def continue_conversation(
        self,
        *,
        conversation_id: uuid.UUID | None = None,
        external_id: str | None = None,
    ) -> Conversation:
        """Load one known conversation using exactly one identifier."""
        if (conversation_id is None) == (external_id is None):
            raise ConversationIdentifierError(
                "provide exactly one of conversation_id or external_id"
            )

        async with self._session.begin():
            statement = select(Conversation)
            if conversation_id is not None:
                statement = statement.where(Conversation.id == conversation_id)
                identifier = str(conversation_id)
            else:
                statement = statement.where(Conversation.external_id == external_id)
                identifier = external_id

            conversation = await self._session.scalar(statement)
            if conversation is None:
                raise ConversationNotFoundError(f"unknown conversation: {identifier}")
            return conversation

    async def store_exchange(
        self,
        *,
        conversation_id: uuid.UUID,
        user_content: str,
        assistant_content: str | None,
        run_result: ChatRunResult,
    ) -> OrchestrationRun:
        """Persist one chat request, messages, and its run result atomically."""
        try:
            async with self._session.begin():
                conversation = await self._locked_conversation(conversation_id)
                await self._ensure_request_is_new(run_result.request_id)

                next_sequence = await self._next_sequence_number(conversation.id)
                messages = [
                    Message(
                        conversation_id=conversation.id,
                        sequence_number=next_sequence,
                        role=MessageRole.USER,
                        content=user_content,
                    )
                ]
                if assistant_content is not None:
                    messages.append(
                        Message(
                            conversation_id=conversation.id,
                            sequence_number=next_sequence + 1,
                            role=MessageRole.ASSISTANT,
                            content=assistant_content,
                        )
                    )

                run = OrchestrationRun(
                    conversation_id=conversation.id,
                    request_id=run_result.request_id,
                    correlation_id=run_result.correlation_id,
                    status=run_result.status,
                    provider=run_result.provider,
                    model=run_result.model,
                    input_tokens=run_result.input_tokens,
                    output_tokens=run_result.output_tokens,
                    error_code=run_result.error_code,
                    error_message=run_result.error_message,
                )
                conversation.updated_at = datetime.now(UTC)
                self._session.add_all([*messages, run])
                await self._session.flush()
                return run
        except IntegrityError as error:
            raise ConversationWriteConflictError(
                "could not write a consistent conversation exchange"
            ) from error

    async def _locked_conversation(self, conversation_id: uuid.UUID) -> Conversation:
        statement = (
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .with_for_update()
        )
        conversation = await self._session.scalar(statement)
        if conversation is None:
            raise ConversationNotFoundError(f"unknown conversation: {conversation_id}")
        return conversation

    async def _ensure_request_is_new(self, request_id: str) -> None:
        existing_run = await self._session.scalar(
            select(OrchestrationRun.id).where(OrchestrationRun.request_id == request_id)
        )
        if existing_run is not None:
            raise ChatRunConflictError(request_id)

    async def _next_sequence_number(self, conversation_id: uuid.UUID) -> int:
        last_sequence = await self._session.scalar(
            select(func.max(Message.sequence_number)).where(
                Message.conversation_id == conversation_id
            )
        )
        if last_sequence is None:
            return 0
        return int(last_sequence) + 1
