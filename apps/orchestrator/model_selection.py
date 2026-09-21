"""Resolve requested model identifiers to one enabled provider client."""

from dataclasses import dataclass

from apps.orchestrator.model_catalog import AvailableModel, ModelCatalog
from packages.providers.contracts import ProviderClient, ProviderModel


class ModelNotFoundError(LookupError):
    """Raised when a requested or fallback model is not enabled."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"unknown model: {model_id}")


class FallbackModelNotConfiguredError(RuntimeError):
    """Raised when a caller omits a model without configuring a fallback."""

    def __init__(self) -> None:
        super().__init__("no model was requested and no fallback model is configured")


@dataclass(frozen=True, slots=True)
class SelectedModel:
    """An internal model-to-provider resolution result."""

    model: ProviderModel
    provider_client: ProviderClient


class ModelSelector:
    """Select one known model, using a fallback only when no model was requested."""

    def __init__(self, catalog: ModelCatalog, fallback_model: str | None) -> None:
        self._catalog = catalog
        self._fallback_model = fallback_model

    async def select(self, requested_model: str | None) -> SelectedModel:
        """Resolve a requested model or the explicitly configured fallback."""
        model_id = requested_model or self._fallback_model
        if model_id is None:
            raise FallbackModelNotConfiguredError()

        available_model = await self._catalog.find_model(model_id)
        if available_model is None:
            raise ModelNotFoundError(model_id)

        return self._selected_model(available_model)

    @staticmethod
    def _selected_model(available_model: AvailableModel) -> SelectedModel:
        return SelectedModel(
            model=available_model.model,
            provider_client=available_model.client,
        )
