"""LangGraph assembly; node behavior remains in the nodes module."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from apps.orchestrator.workflow.models import (
    ChatWorkflowRequest,
    ChatWorkflowResponse,
    OrchestrationState,
)
from apps.orchestrator.workflow.nodes import WorkflowNodes


class ChatWorkflow:
    def __init__(self, nodes: WorkflowNodes) -> None:
        self._graph = self._build_graph(nodes)

    async def run(self, request: ChatWorkflowRequest) -> ChatWorkflowResponse:
        state = await self._graph.ainvoke(self._initial_state(request))
        response = state["final_response"]
        if response is None:
            raise RuntimeError("workflow ended without a final response")
        return response

    @staticmethod
    def _build_graph(nodes: WorkflowNodes):
        graph = StateGraph(OrchestrationState)
        graph.add_node("input_validation", nodes.input_validation)
        graph.add_node("conversation_preparation", nodes.conversation_preparation)
        graph.add_node("model_selection", nodes.model_selection)
        graph.add_node("provider_execution", nodes.provider_execution)
        graph.add_node("output_validation", nodes.output_validation)
        graph.add_node("persistence", nodes.persistence)
        graph.add_node("final_response", nodes.final_response)
        graph.add_edge(START, "input_validation")
        graph.add_conditional_edges(
            "input_validation",
            ChatWorkflow._next_node,
            {"continue": "conversation_preparation", "persist": "persistence"},
        )
        graph.add_conditional_edges(
            "conversation_preparation",
            ChatWorkflow._next_node,
            {"continue": "model_selection", "persist": "persistence"},
        )
        graph.add_conditional_edges(
            "model_selection",
            ChatWorkflow._next_node,
            {"continue": "provider_execution", "persist": "persistence"},
        )
        graph.add_conditional_edges(
            "provider_execution",
            ChatWorkflow._next_node,
            {"continue": "output_validation", "persist": "persistence"},
        )
        graph.add_edge("output_validation", "persistence")
        graph.add_edge("persistence", "final_response")
        graph.add_edge("final_response", END)
        return graph.compile()

    @staticmethod
    def _initial_state(request: ChatWorkflowRequest) -> OrchestrationState:
        return {
            "request": request,
            "input_guardrail": None,
            "conversation": None,
            "selected_model": None,
            "provider_response": None,
            "output_guardrail": None,
            "failure": None,
            "persisted_run": None,
            "final_response": None,
        }

    @staticmethod
    def _next_node(state: OrchestrationState) -> Literal["continue", "persist"]:
        return "persist" if state["failure"] is not None else "continue"
