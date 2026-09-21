"""Create configured provider clients for the Orchestrator lifecycle."""

from collections.abc import Mapping

from packages.common.settings import OrchestratorSettings
from packages.providers.contracts import ProviderClient
from packages.providers.ollama import OllamaClient
from packages.providers.openrouter import OpenRouterClient


def create_provider_clients(
    settings: OrchestratorSettings,
) -> dict[str, ProviderClient]:
    """Create enabled providers without making network requests."""
    clients: dict[str, ProviderClient] = {
        OpenRouterClient.name: OpenRouterClient(
            api_key=settings.openrouter_api_key.get_secret_value(),
            base_url=str(settings.openrouter_base_url),
            timeout_seconds=settings.provider_timeout_seconds,
        )
    }

    if settings.ollama_enabled:
        clients[OllamaClient.name] = OllamaClient(
            base_url=str(settings.ollama_base_url),
            timeout_seconds=settings.provider_timeout_seconds,
        )

    return clients


async def close_provider_clients(clients: Mapping[str, ProviderClient]) -> None:
    """Close every configured provider's HTTP connections."""
    for client in clients.values():
        await client.close()
