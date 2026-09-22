"""The local guardrail facade used by workflow nodes."""

from apps.orchestrator.guardrailer.input_guard import InputGuard
from apps.orchestrator.guardrailer.models import GuardrailDecision
from apps.orchestrator.guardrailer.policy_engine import ContentPolicy
from packages.providers.contracts import ChatMessage


class GuardrailsClient:
    """Runs input/output policy without exposing policy internals to nodes."""

    def __init__(self, policy: ContentPolicy) -> None:
        self._input_guard = InputGuard(policy)
        self._policy = policy

    def check_input(self, messages: tuple[ChatMessage, ...]) -> GuardrailDecision:
        return self._input_guard.check(messages)

    def check_output(self, content: str, *, final: bool = True) -> GuardrailDecision:
        return self._policy.check_output(content, final=final)

    def new_stream_validator(self) -> "StreamingOutputValidator":
        return StreamingOutputValidator(self)


class StreamingOutputValidator:
    """Avoid emitting a prefix that might complete a split blocked term."""

    def __init__(self, guardrails: GuardrailsClient) -> None:
        self._guardrails = guardrails
        self._received = ""
        self._pending = ""
        self._holdback = guardrails._policy.streaming_holdback_characters

    def add(self, content: str) -> tuple[str, GuardrailDecision]:
        self._received += content
        decision = self._guardrails.check_output(self._received, final=False)
        if not decision.approved:
            return "", decision
        self._pending += content
        safe_length = len(self._pending) - self._holdback
        if safe_length <= 0:
            return "", decision
        safe_content = self._pending[:safe_length]
        self._pending = self._pending[safe_length:]
        return safe_content, decision

    def finish(self) -> tuple[str, GuardrailDecision]:
        decision = self._guardrails.check_output(self._received)
        if not decision.approved:
            return "", decision
        safe_content, self._pending = self._pending, ""
        return safe_content, decision
