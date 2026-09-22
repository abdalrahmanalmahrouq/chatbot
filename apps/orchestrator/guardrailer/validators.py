"""Request-shape validation independent from HTTP route validation."""

from apps.orchestrator.guardrailer.models import GuardrailDecision
from apps.orchestrator.guardrailer.policy_engine import ContentPolicy
from packages.providers.contracts import ChatMessage


def validate_messages(
    messages: tuple[ChatMessage, ...], policy: ContentPolicy
) -> GuardrailDecision:
    """Validate message structure and inbound configured policy."""
    if not messages:
        return GuardrailDecision.reject(
            "empty_messages", "at least one chat message is required"
        )
    if messages[-1].role != "user":
        return GuardrailDecision.reject(
            "last_message_not_user", "the final chat message must be from the user"
        )

    total_characters = 0
    for message in messages:
        if message.role not in {"system", "user", "assistant"}:
            return GuardrailDecision.reject(
                "invalid_message_role", "chat messages contain an unsupported role"
            )
        if not message.content.strip():
            return GuardrailDecision.reject(
                "empty_message_content", "chat messages cannot be empty"
            )
        total_characters += len(message.content)

    length_decision = policy.check_input_length(total_characters)
    if not length_decision.approved:
        return length_decision
    combined_content = "\n".join(item.content for item in messages)
    if policy.find_blocked_term(combined_content) is not None:
        return GuardrailDecision.reject(
            "blocked_input", "chat messages contain blocked content"
        )
    return GuardrailDecision.allow()
