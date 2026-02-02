from __future__ import annotations

import json

import httpx
import pytest
import respx

from vibe.core.config import (
    ModelConfig,
    ProviderConfig,
    ResponseFormatConfig,
    ZAIThinkingConfig,
    ZAIWebSearchConfig,
)
from vibe.core.llm.backend.generic import GenericBackend
from vibe.core.types import LLMMessage, Role


@pytest.mark.asyncio
async def test_zai_payload_includes_model_controls_and_web_search() -> None:
    base_url = "https://api.z.ai"
    provider = ProviderConfig(
        name="zai-coding",
        api_base=f"{base_url}/api/coding/paas/v4",
        api_key_env_var="ZAI_API_KEY",
        api_style="zai",
        thinking=ZAIThinkingConfig(type="enabled", clear_thinking=True),
        web_search=ZAIWebSearchConfig(
            enable=True,
            search_engine="search-prime",
            count=3,
            search_result=True,
            content_size="high",
        ),
    )
    model = ModelConfig(
        name="glm-4.7",
        provider="zai-coding",
        alias="glm-4.7",
        max_tokens=64,
        do_sample=True,
        top_p=0.9,
        stop=["<stop>"],
        response_format=ResponseFormatConfig(type="json_object"),
        user_id="user-123",
        request_id="req-456",
        tool_stream=True,
    )
    messages = [
        LLMMessage(role=Role.user, content="Hello"),
        LLMMessage(
            role=Role.assistant, content="Hi", reasoning_content="hidden reasoning"
        ),
    ]

    with respx.mock(base_url=base_url) as mock_api:
        route = mock_api.post("/api/coding/paas/v4/chat/completions").mock(
            return_value=httpx.Response(
                status_code=200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        )
        backend = GenericBackend(provider=provider)
        result = await backend.complete(
            model=model,
            messages=messages,
            temperature=0.2,
            tools=None,
            max_tokens=model.max_tokens,
            tool_choice="any",
            extra_headers=None,
        )

    assert route.called
    body = json.loads(route.calls[0].request.content)

    assert body["model"] == "glm-4.7"
    assert body["max_tokens"] == 64
    assert body["do_sample"] is True
    assert body["top_p"] == 0.9
    assert body["stop"] == ["<stop>"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["user_id"] == "user-123"
    assert body["request_id"] == "req-456"
    assert body["tool_stream"] is True
    assert body["tool_choice"] == "auto"
    assert body["thinking"] == {"type": "enabled", "clear_thinking": True}

    tools = body.get("tools") or []
    web_search = next(
        (tool["web_search"] for tool in tools if tool.get("type") == "web_search"),
        None,
    )
    assert web_search is not None
    assert web_search["enable"] == "True"
    assert web_search["search_engine"] == "search-prime"
    assert web_search["count"] == "3"
    assert web_search["search_result"] == "True"
    assert web_search["content_size"] == "high"

    for msg in body["messages"]:
        assert "reasoning_content" not in msg

    assert result.message.content == "ok"
