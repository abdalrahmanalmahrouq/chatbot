"""Input guardrail entry point."""

from apps.orchestrator.guardrailer.models import GuardrailDecision
from apps.orchestrator.guardrailer.policy_engine import ContentPolicy
from apps.orchestrator.guardrailer.validators import validate_messages
from packages.providers.contracts import ChatMessage


class InputGuard:
    def __init__(self, policy: ContentPolicy) -> None:
        self._policy = policy

    def check(self, messages: tuple[ChatMessage, ...]) -> GuardrailDecision:
        return validate_messages(messages, self._policy)
