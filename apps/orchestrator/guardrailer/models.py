"""Types exchanged by the guardrail policy components."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    """A safe policy decision; it never contains request or provider content."""

    approved: bool
    code: str | None = None
    message: str | None = None

    @classmethod
    def allow(cls) -> "GuardrailDecision":
        return cls(approved=True)

    @classmethod
    def reject(cls, code: str, message: str) -> "GuardrailDecision":
        return cls(approved=False, code=code, message=message)
