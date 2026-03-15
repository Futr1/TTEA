from __future__ import annotations

from ..config import EnvironmentConfig
from ..exceptions import EnvironmentIntegrationError
from .base import WebEnvironmentAdapter
from .miniwob import MiniWoBEnvironmentAdapter
from .webarena import WebArenaEnvironmentAdapter


def build_environment_adapter(config: EnvironmentConfig) -> WebEnvironmentAdapter | None:
    if not config.enabled:
        return None
    if config.provider == "webarena":
        return WebArenaEnvironmentAdapter(config)
    if config.provider == "miniwob":
        return MiniWoBEnvironmentAdapter(config)
    raise EnvironmentIntegrationError(f"Unsupported environment provider: {config.provider}")
