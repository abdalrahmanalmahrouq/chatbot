"""Public workflow values and its documented LangGraph state."""

import uuid
from dataclasses import dataclass
from typing import TypedDict

from apps.orchestrator.guardrailer.models import GuardrailDecision
from apps.orchestrator.model_selection import SelectedModel
from packages.persistence.models import Conversation, OrchestrationRun, RunStatus
from packages.providers.contracts import ChatCompletion, ChatMessage


@dataclass(frozen=True, slots=True)
class ChatWorkflowRequest:
    request_id: str
    correlation_id: str
    messages: tuple[ChatMessage, ...]
    model: str | None = None
    conversation_id: uuid.UUID | None = None
    external_conversation_id: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class WorkflowFailure:
    code: str
    message: str
    status: RunStatus = RunStatus.FAILED


@dataclass(frozen=True, slots=True)
class ChatWorkflowResponse:
    accepted: bool
    content: str | None
    model: str | None
    conversation_id: uuid.UUID | None
    run_id: uuid.UUID | None
    failure: WorkflowFailure | None = None


class OrchestrationState(TypedDict):
    """Every node reads/writes only the state fields documented here."""

    request: ChatWorkflowRequest
    input_guardrail: GuardrailDecision | None
    conversation: Conversation | None
    selected_model: SelectedModel | None
    provider_response: ChatCompletion | None
    output_guardrail: GuardrailDecision | None
    failure: WorkflowFailure | None
    persisted_run: OrchestrationRun | None
    final_response: ChatWorkflowResponse | None
