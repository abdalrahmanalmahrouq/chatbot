"""Configured content policy applied before and after provider execution."""

from dataclasses import dataclass

from apps.orchestrator.guardrailer.models import GuardrailDecision


@dataclass(frozen=True, slots=True)
class ContentPolicy:
    """The small, explicit policy set required for this project phase."""

    input_max_characters: int
    output_max_characters: int
    blocked_terms: tuple[str, ...] = ()

    def check_input_length(self, character_count: int) -> GuardrailDecision:
        if character_count > self.input_max_characters:
            return GuardrailDecision.reject(
                "input_too_long", "chat messages exceed the configured input limit"
            )
        return GuardrailDecision.allow()

    def check_output(self, content: str, *, final: bool) -> GuardrailDecision:
        if final and not content.strip():
            return GuardrailDecision.reject(
                "empty_output", "provider returned an empty response"
            )
        if len(content) > self.output_max_characters:
            return GuardrailDecision.reject(
                "output_too_long", "provider output exceeds the configured limit"
            )
        if self.find_blocked_term(content) is not None:
            return GuardrailDecision.reject(
                "blocked_output", "provider output contains blocked content"
            )
        return GuardrailDecision.allow()

    def find_blocked_term(self, content: str) -> str | None:
        normalized = content.casefold()
        return next((term for term in self.blocked_terms if term in normalized), None)

    @property
    def streaming_holdback_characters(self) -> int:
        return max((len(term) - 1 for term in self.blocked_terms), default=0)
