"""Tests for Core's fixed HTTP boundary to Orchestrator."""

import asyncio

import httpx
import pytest

from apps.core.orchestrator_client import (
    OrchestratorClient,
    OrchestratorResponseError,
)


def test_core_client_calls_only_the_orchestrator_model_path() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"object": "list", "data": []})

    client = OrchestratorClient(
        "http://orchestrator:8001",
        1,
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(client.list_models())
    asyncio.run(client.close())

    assert result == {"object": "list", "data": []}


def test_core_client_keeps_an_upstream_http_status_for_route_translation() -> None:
    client = OrchestratorClient(
        "http://orchestrator:8001",
        1,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(404, json={"detail": "model not found"})
        ),
    )

    with pytest.raises(OrchestratorResponseError) as error:
        asyncio.run(client.get_model("missing"))
    asyncio.run(client.close())

    assert error.value.status_code == 404
