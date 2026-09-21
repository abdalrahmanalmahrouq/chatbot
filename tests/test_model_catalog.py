"""Tests for collecting provider models into one public catalog."""

import asyncio

import pytest

from apps.orchestrator.model_catalog import ModelCatalog, ModelCatalogConflictError
from packages.providers.contracts import ProviderModel


class FakeProviderClient:
    def __init__(self, name: str, models: list[ProviderModel]) -> None:
        self.name = name
        self._models = models

    async def list_models(self) -> list[ProviderModel]:
        return self._models


def test_combines_enabled_models_without_provider_details() -> None:
    catalog = ModelCatalog(
        {
            "openrouter": FakeProviderClient(
                "openrouter",
                [ProviderModel(id="zeta", name="Zeta", provider="openrouter")],
            ),
            "ollama": FakeProviderClient(
                "ollama",
                [ProviderModel(id="alpha", name="Alpha", provider="ollama")],
            ),
        }
    )

    model_list = asyncio.run(catalog.list_models())

    assert model_list.model_dump() == {
        "object": "list",
        "data": [
            {"id": "alpha", "object": "model", "created": 0, "owned_by": "chatbot"},
            {"id": "zeta", "object": "model", "created": 0, "owned_by": "chatbot"},
        ],
    }


def test_rejects_duplicate_model_identifiers() -> None:
    catalog = ModelCatalog(
        {
            "first": FakeProviderClient(
                "first",
                [ProviderModel(id="shared", name="Shared", provider="first")],
            ),
            "second": FakeProviderClient(
                "second",
                [ProviderModel(id="shared", name="Shared", provider="second")],
            ),
        }
    )

    with pytest.raises(ModelCatalogConflictError, match="multiple providers"):
        asyncio.run(catalog.list_models())
