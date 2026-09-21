"""Tests for resolving a requested model to an enabled provider."""

import asyncio

import pytest

from apps.orchestrator.model_catalog import ModelCatalog
from apps.orchestrator.model_selection import (
    FallbackModelNotConfiguredError,
    ModelNotFoundError,
    ModelSelector,
)
from packages.providers.contracts import ProviderModel


class FakeProviderClient:
    name = "fake"

    def __init__(self, models: list[ProviderModel]) -> None:
        self._models = models

    async def list_models(self) -> list[ProviderModel]:
        return self._models


def test_selects_requested_model_and_its_provider() -> None:
    provider = FakeProviderClient(
        [ProviderModel(id="chosen", name="Chosen", provider="fake")]
    )
    selector = ModelSelector(ModelCatalog({"fake": provider}), fallback_model=None)

    selected = asyncio.run(selector.select("chosen"))

    assert selected.model.id == "chosen"
    assert selected.provider_client is provider


def test_uses_configured_fallback_only_when_model_is_omitted() -> None:
    provider = FakeProviderClient(
        [ProviderModel(id="fallback", name="Fallback", provider="fake")]
    )
    selector = ModelSelector(
        ModelCatalog({"fake": provider}), fallback_model="fallback"
    )

    selected = asyncio.run(selector.select(None))

    assert selected.model.id == "fallback"


def test_rejects_unknown_requested_models_instead_of_falling_back() -> None:
    provider = FakeProviderClient(
        [ProviderModel(id="fallback", name="Fallback", provider="fake")]
    )
    selector = ModelSelector(
        ModelCatalog({"fake": provider}), fallback_model="fallback"
    )

    with pytest.raises(ModelNotFoundError, match="unknown model: missing"):
        asyncio.run(selector.select("missing"))


def test_requires_fallback_when_model_is_omitted() -> None:
    selector = ModelSelector(ModelCatalog({}), fallback_model=None)

    with pytest.raises(FallbackModelNotConfiguredError):
        asyncio.run(selector.select(None))
