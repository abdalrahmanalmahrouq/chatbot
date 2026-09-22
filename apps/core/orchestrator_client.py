"""Small HTTP client for Core's fixed Orchestrator integration."""

from typing import Any

import httpx


class OrchestratorResponseError(RuntimeError):
    """An HTTP response from Orchestrator that Core can translate safely."""

    def __init__(self, status_code: int, detail: object | None = None) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"orchestrator responded with HTTP {status_code}")


class OrchestratorClient:
    """Forward only known Core operations to the configured Orchestrator URL."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds, transport=transport
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> dict[str, Any]:
        return await self._get_json("/v1/models")

    async def get_model(self, model_id: str) -> dict[str, Any]:
        return await self._get_json(f"/v1/models/{model_id}")

    async def orchestrate(
        self, body: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        response = await self._client.post(
            "/v1/orchestrate", json=body, headers=headers
        )
        return self._response_json(response)

    async def _get_json(self, path: str) -> dict[str, Any]:
        response = await self._client.get(path)
        return self._response_json(response)

    @staticmethod
    def _response_json(response: httpx.Response) -> dict[str, Any]:
        if response.is_error:
            detail: object | None = None
            try:
                detail = response.json()
            except ValueError:
                pass
            raise OrchestratorResponseError(response.status_code, detail)
        return response.json()
