"""HTTP request and response models owned by the Orchestrator API."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from apps.orchestrator.workflow.models import ChatWorkflowResponse, WorkflowFailure
from packages.providers.contracts import ChatMessage


class OrchestrateMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    def to_chat_message(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.content)


class OrchestrateRequest(BaseModel):
    messages: list[OrchestrateMessage] = Field(min_length=1)
    model: str | None = None
    conversation_id: uuid.UUID | None = None
    external_conversation_id: str | None = Field(default=None, min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_conversation_identifier(self) -> "OrchestrateRequest":
        """A request must choose one stable way to identify a conversation."""
        if (
            self.conversation_id is not None
            and self.external_conversation_id is not None
        ):
            raise ValueError(
                "provide either conversation_id or external_conversation_id, not both"
            )
        return self


class OrchestrationFailure(BaseModel):
    code: str
    message: str

    @classmethod
    def from_workflow(cls, failure: WorkflowFailure) -> "OrchestrationFailure":
        return cls(code=failure.code, message=failure.message)


class OrchestrateResponse(BaseModel):
    accepted: bool
    content: str | None = None
    model: str | None = None
    conversation_id: uuid.UUID | None = None
    run_id: uuid.UUID | None = None
    failure: OrchestrationFailure | None = None

    @classmethod
    def from_workflow(cls, response: ChatWorkflowResponse) -> "OrchestrateResponse":
        failure = (
            OrchestrationFailure.from_workflow(response.failure)
            if response.failure is not None
            else None
        )
        return cls(
            accepted=response.accepted,
            content=response.content,
            model=response.model,
            conversation_id=response.conversation_id,
            run_id=response.run_id,
            failure=failure,
        )
