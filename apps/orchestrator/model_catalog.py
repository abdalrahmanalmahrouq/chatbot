"""Collect enabled provider models into a public OpenAI-compatible catalog."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from packages.providers.contracts import ProviderClient, ProviderModel


class ModelCatalogConflictError(RuntimeError):
    """Raised when enabled providers expose the same public model identifier."""


class OpenAIModel(BaseModel):
    """The public model shape expected by OpenAI-compatible clients."""

    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "chatbot"


class OpenAIModelList(BaseModel):
    """The public wrapper shape for a list of models."""

    object: Literal["list"] = "list"
    data: list[OpenAIModel]


@dataclass(frozen=True, slots=True)
class AvailableModel:
    """Keep the provider association internal to Orchestrator services."""

    model: ProviderModel
    client: ProviderClient


class ModelCatalog:
    """Collect models from enabled providers without exposing provider details."""

    def __init__(self, provider_clients: dict[str, ProviderClient]) -> None:
        self._provider_clients = provider_clients

    async def list_models(self) -> OpenAIModelList:
        """Return every enabled model in the OpenAI-compatible public shape."""
        available_models = await self._collect_models()
        models = [
            OpenAIModel(id=available.model.id)
            for available in available_models.values()
        ]
        models.sort(key=lambda model: model.id)
        return OpenAIModelList(data=models)

    async def find_model(self, model_id: str) -> AvailableModel | None:
        """Find a model and its provider association for internal use."""
        available_models = await self._collect_models()
        return available_models.get(model_id)

    async def _collect_models(self) -> dict[str, AvailableModel]:
        models_by_id: dict[str, AvailableModel] = {}

        for provider_client in self._provider_clients.values():
            provider_models = await provider_client.list_models()

            for provider_model in provider_models:
                if provider_model.id in models_by_id:
                    raise ModelCatalogConflictError(
                        f"multiple providers expose model '{provider_model.id}'"
                    )
                models_by_id[provider_model.id] = AvailableModel(
                    model=provider_model,
                    client=provider_client,
                )

        return models_by_id
