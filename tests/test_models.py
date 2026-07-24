from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from showdown_mind.models import (
    ModelConfigurationError,
    ModelRequest,
    ModelCallError,
    ModelTool,
    OpenAICompatibleModelClient,
    live_model_client_from_env,
)


def battle_tool() -> ModelTool:
    return ModelTool(
        name="choose_battle_action",
        description="Choose a legal action.",
        parameters={
            "type": "object",
            "properties": {
                "action_id": {
                    "type": "string",
                    "enum": ["move:thunderbolt"],
                }
            },
            "required": ["action_id"],
            "additionalProperties": False,
        },
    )


def fake_sdk_client(
    *,
    tool_calls: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    if tool_calls is None:
        tool_calls = [
            SimpleNamespace(
                id="call-test",
                type="function",
                function=SimpleNamespace(
                    name="choose_battle_action",
                    arguments='{"action_id":"move:thunderbolt"}',
                ),
            )
        ]
    completion = SimpleNamespace(
        id="chatcmpl-test",
        model="gpt-5.6-luna",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=tool_calls)
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
            system_prompt="Call the tool.",
            user_prompt='{"legal_actions":[]}',
            tool=battle_tool(),
        )
    )

    request = sdk_client.chat.completions.create.await_args.kwargs
    assert request["model"] == "gpt-5.6-luna"
    assert "response_format" not in request
    assert request["messages"][0]["role"] == "system"
    assert request["parallel_tool_calls"] is False
    assert "extra_body" not in request
    assert request["tool_choice"]["function"]["name"] == "choose_battle_action"
    assert request["tools"][0]["function"]["strict"] is True
    assert response.content == '{"action_id":"move:thunderbolt"}'
    assert response.response_id == "chatcmpl-test"
    assert response.tool_call_id == "call-test"
    assert response.usage is not None
    assert response.usage.total_tokens == 120


@pytest.mark.asyncio
async def test_openai_compatible_client_can_disable_provider_thinking() -> None:
    sdk_client = fake_sdk_client()
    client = OpenAICompatibleModelClient(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        thinking_mode="disabled",
        client=sdk_client,
    )

    await client.complete(
        ModelRequest(
            system_prompt="Call the tool.",
            user_prompt="{}",
            tool=battle_tool(),
        )
    )

    request = sdk_client.chat.completions.create.await_args.kwargs
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_calls", "message"),
    [
        ([], "exactly one"),
        (
            [
                SimpleNamespace(
                    id="call-test",
                    type="function",
                    function=SimpleNamespace(
                        name="another_tool",
                        arguments="{}",
                    ),
                )
            ],
            "unexpected tool",
        ),
        (
            [
                SimpleNamespace(
                    id=f"call-{index}",
                    type="function",
                    function=SimpleNamespace(
                        name="choose_battle_action",
                        arguments='{"action_id":"move:thunderbolt"}',
                    ),
                )
                for index in range(2)
            ],
            "exactly one",
        ),
    ],
)
async def test_openai_compatible_client_rejects_invalid_tool_calls(
    tool_calls: list[SimpleNamespace],
    message: str,
) -> None:
    client = OpenAICompatibleModelClient(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="gpt-5.6-luna",
        client=fake_sdk_client(tool_calls=tool_calls),
    )

    with pytest.raises(ModelCallError, match=message):
        await client.complete(
            ModelRequest(
                system_prompt="Call the tool.",
                user_prompt="{}",
                tool=battle_tool(),
            )
        )


@pytest.mark.asyncio
async def test_openai_compatible_client_requires_tool_call_id() -> None:
    tool_call = SimpleNamespace(
        id=None,
        type="function",
        function=SimpleNamespace(
            name="choose_battle_action",
            arguments='{"action_id":"move:thunderbolt"}',
        ),
    )
    client = OpenAICompatibleModelClient(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="gpt-5.6-luna",
        client=fake_sdk_client(tool_calls=[tool_call]),
    )

    with pytest.raises(ModelCallError, match="call ID"):
        await client.complete(
            ModelRequest(
                system_prompt="Call the tool.",
                user_prompt="{}",
                tool=battle_tool(),
            )
        )


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


def test_rejects_unknown_thinking_mode() -> None:
    with pytest.raises(ModelConfigurationError, match="thinking_mode"):
        OpenAICompatibleModelClient(
            api_key="test-key",
            base_url="https://provider.example/v1",
            model="test-model",
            thinking_mode="sometimes",
        )
