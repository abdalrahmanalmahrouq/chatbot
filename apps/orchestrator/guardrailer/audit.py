"""Safe guardrail audit metadata builders."""

from apps.orchestrator.conversation_service import AuditEventRecord
from apps.orchestrator.guardrailer.models import GuardrailDecision


def rejected_input_event(decision: GuardrailDecision) -> AuditEventRecord:
    return AuditEventRecord(
        event_type="input_guardrail_rejected",
        details={"code": decision.code or "input_rejected"},
    )


def rejected_output_event(decision: GuardrailDecision) -> AuditEventRecord:
    return AuditEventRecord(
        event_type="output_guardrail_rejected",
        details={"code": decision.code or "output_rejected"},
    )
