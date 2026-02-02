from __future__ import annotations

import os

import pytest

from vibe.core.config import MissingAPIKeyError, VibeConfig


def test_default_active_model_is_glm_47() -> None:
    config = VibeConfig()
    assert config.active_model == "glm-4.7"
    active_model = config.get_active_model()
    assert active_model.alias == "glm-4.7"
    provider = config.get_provider_for_model(active_model)
    assert provider.api_key_env_var == "ZAI_API_KEY"


def test_missing_zai_key_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("VIBE_ACTIVE_MODEL", raising=False)
    with pytest.raises(MissingAPIKeyError) as exc:
        VibeConfig()
    assert exc.value.env_key == "ZAI_API_KEY"
