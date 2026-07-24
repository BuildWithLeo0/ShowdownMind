from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from showdown_mind.models import (
    ModelConfigurationError,
    ModelRequest,
    OpenAICompatibleModelClient,
    live_model_client_from_env,
)


def fake_sdk_client() -> SimpleNamespace:
    completion = SimpleNamespace(
        id="chatcmpl-test",
        model="gpt-5.6-luna",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"action_id":"move:thunderbolt"}')
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        ),
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=completion),
            )
        ),
        close=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_openai_compatible_client_maps_request_and_usage() -> None:
    sdk_client = fake_sdk_client()
    client = OpenAICompatibleModelClient(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="gpt-5.6-luna",
        client=sdk_client,
    )

    response = await client.complete(
        ModelRequest(
            system_prompt="Return JSON.",
            user_prompt='{"legal_actions":[]}',
        )
    )

    request = sdk_client.chat.completions.create.await_args.kwargs
    assert request["model"] == "gpt-5.6-luna"
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][0]["role"] == "system"
    assert response.response_id == "chatcmpl-test"
    assert response.usage is not None
    assert response.usage.total_tokens == 120


@pytest.mark.asyncio
async def test_openai_compatible_client_closes_sdk_client() -> None:
    sdk_client = fake_sdk_client()
    client = OpenAICompatibleModelClient(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="gpt-5.6-luna",
        client=sdk_client,
    )

    await client.aclose()

    sdk_client.close.assert_awaited_once()


def test_live_client_requires_project_specific_api_key() -> None:
    with pytest.raises(ModelConfigurationError, match="SHOWDOWN_MIND_API_KEY"):
        live_model_client_from_env({})


def test_remote_base_url_requires_https() -> None:
    with pytest.raises(ModelConfigurationError, match="localhost"):
        OpenAICompatibleModelClient(
            api_key="test-key",
            base_url="http://provider.example/v1",
            model="gpt-5.6-luna",
        )
