"""Focused node functions for the Orchestrator LangGraph workflow."""

from dataclasses import dataclass

from apps.orchestrator.conversation_service import (
    ChatRunResult,
    ConversationConflictError,
    ConversationIdentifierError,
    ConversationNotFoundError,
    ConversationService,
)
from apps.orchestrator.guardrailer.audit import (
    rejected_input_event,
    rejected_output_event,
)
from apps.orchestrator.guardrailer.guardrails_client import GuardrailsClient
from apps.orchestrator.model_selection import (
    FallbackModelNotConfiguredError,
    ModelNotFoundError,
    ModelSelector,
)
from apps.orchestrator.workflow.models import OrchestrationState, WorkflowFailure
from packages.persistence.models import Conversation, RunStatus
from packages.providers.contracts import ChatCompletionRequest
from packages.providers.exceptions import ProviderError


@dataclass(slots=True)
class WorkflowNodes:
    """Dependencies and implementations for the seven workflow nodes."""

    guardrails: GuardrailsClient
    model_selector: ModelSelector
    conversations: ConversationService

    def input_validation(self, state: OrchestrationState) -> dict[str, object]:
        decision = self.guardrails.check_input(state["request"].messages)
        if decision.approved:
            return {"input_guardrail": decision}
        return {"input_guardrail": decision, "failure": self._failure(decision)}

    async def conversation_preparation(
        self, state: OrchestrationState
    ) -> dict[str, object]:
        request = state["request"]
        try:
            if request.conversation_id is not None:
                conversation = await self.conversations.continue_conversation(
                    conversation_id=request.conversation_id
                )
            elif request.external_conversation_id is not None:
                conversation = await self._continue_or_create_external_conversation(
                    request.external_conversation_id
                )
            else:
                conversation = await self.conversations.create_conversation()
        except ConversationConflictError:
            return {
                "failure": WorkflowFailure(
                    "conversation_conflict", "conversation exists"
                )
            }
        except ConversationNotFoundError:
            return {
                "failure": WorkflowFailure(
                    "conversation_not_found", "conversation was not found"
                )
            }
        except ConversationIdentifierError:
            return {
                "failure": WorkflowFailure(
                    "invalid_conversation_identifier", "invalid conversation identifier"
                )
            }
        return {"conversation": conversation}

    async def _continue_or_create_external_conversation(
        self, external_id: str
    ) -> Conversation:
        """Keep every repeated Open WebUI chat ID on one conversation."""
        try:
            return await self.conversations.continue_conversation(
                external_id=external_id
            )
        except ConversationNotFoundError:
            try:
                return await self.conversations.create_conversation(
                    external_id=external_id
                )
            except ConversationConflictError:
                # Another request created the same external conversation first.
                return await self.conversations.continue_conversation(
                    external_id=external_id
                )

    async def model_selection(self, state: OrchestrationState) -> dict[str, object]:
        try:
            selected_model = await self.model_selector.select(state["request"].model)
        except ModelNotFoundError:
            return {
                "failure": WorkflowFailure(
                    "model_not_found", "requested model is not enabled"
                )
            }
        except FallbackModelNotConfiguredError:
            return {
                "failure": WorkflowFailure(
                    "fallback_model_not_configured", "a model must be requested"
                )
            }
        return {"selected_model": selected_model}

    async def provider_execution(self, state: OrchestrationState) -> dict[str, object]:
        selected = state["selected_model"]
        if selected is None:
            return {
                "failure": WorkflowFailure(
                    "model_not_selected", "model was not selected"
                )
            }
        request = state["request"]
        try:
            response = await selected.provider_client.complete_chat(
                ChatCompletionRequest(
                    model=selected.model.id,
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
            )
        except ProviderError:
            return {
                "failure": WorkflowFailure("provider_failed", "provider request failed")
            }
        return {"provider_response": response}

    def output_validation(self, state: OrchestrationState) -> dict[str, object]:
        response = state["provider_response"]
        if response is None:
            return {
                "failure": WorkflowFailure(
                    "missing_provider_response", "provider returned no response"
                )
            }
        decision = self.guardrails.check_output(response.message.content)
        if decision.approved:
            return {"output_guardrail": decision}
        return {"output_guardrail": decision, "failure": self._failure(decision)}

    async def persistence(self, state: OrchestrationState) -> dict[str, object]:
        failure = state["failure"]
        response = state["provider_response"]
        selected = state["selected_model"]
        audit_events = ()
        if state["input_guardrail"] and not state["input_guardrail"].approved:
            audit_events = (rejected_input_event(state["input_guardrail"]),)
        if state["output_guardrail"] and not state["output_guardrail"].approved:
            audit_events = (rejected_output_event(state["output_guardrail"]),)
        result = ChatRunResult(
            request_id=state["request"].request_id,
            correlation_id=state["request"].correlation_id,
            status=failure.status if failure else RunStatus.COMPLETED,
            provider=selected.provider_client.name if selected else None,
            model=selected.model.id if selected else None,
            error_code=failure.code if failure else None,
            error_message=failure.message if failure else None,
        )
        conversation = state["conversation"]
        if conversation is None:
            run = await self.conversations.record_run(result, audit_events)
        else:
            run = await self.conversations.store_exchange(
                conversation_id=conversation.id,
                user_content=state["request"].messages[-1].content,
                assistant_content=(
                    response.message.content if response and not failure else None
                ),
                run_result=result,
                audit_events=audit_events,
            )
        return {"persisted_run": run}

    def final_response(self, state: OrchestrationState) -> dict[str, object]:
        from apps.orchestrator.workflow.models import ChatWorkflowResponse

        failure = state["failure"]
        response = state["provider_response"]
        conversation = state["conversation"]
        run = state["persisted_run"]
        return {
            "final_response": ChatWorkflowResponse(
                accepted=failure is None,
                content=response.message.content
                if response and failure is None
                else None,
                model=response.model if response and failure is None else None,
                conversation_id=conversation.id if conversation else None,
                run_id=run.id if run else None,
                failure=failure,
            )
        }

    @staticmethod
    def _failure(decision: object) -> WorkflowFailure:
        return WorkflowFailure(
            code=getattr(decision, "code", None) or "guardrail_rejected",
            message=getattr(decision, "message", None) or "request was rejected",
        )
