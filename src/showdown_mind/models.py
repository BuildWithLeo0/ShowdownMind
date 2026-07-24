from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from os import environ
from typing import Any, Protocol
from urllib.parse import urlparse

from openai import AsyncOpenAI

from showdown_mind.domain import TokenUsage

DEFAULT_BASE_URL = "https://www.codexapis.com/v1"
DEFAULT_MODEL = "gpt-5.6-luna"
API_KEY_ENV = "SHOWDOWN_MIND_API_KEY"
BASE_URL_ENV = "SHOWDOWN_MIND_BASE_URL"
MODEL_ENV = "SHOWDOWN_MIND_MODEL"
THINKING_ENV = "SHOWDOWN_MIND_THINKING"
ACTION_TOOL_NAME = "choose_battle_action"
TACTICAL_TOOL_NAME = "analyze_battle_options"


@dataclass(frozen=True)
class ModelTool:
    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = True


@dataclass(frozen=True)
class ModelRequest:
    system_prompt: str
    user_prompt: str
    tool: ModelTool
    tool_history: tuple[ToolExchange, ...] = ()


@dataclass(frozen=True)
class ToolExchange:
    tool_call_id: str
    tool_name: str
    arguments: str
    result: str


@dataclass(frozen=True)
class ModelResponse:
    content: str
    model_id: str
    response_id: str | None = None
    tool_call_id: str | None = None
    usage: TokenUsage | None = None


class ModelClient(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one model response for one policy attempt."""


class ModelCallError(RuntimeError):
    """Raised when a model provider does not return a usable response."""


class ScriptedModelClient:
    """A deterministic model double for policy tests."""

    def __init__(self, responses: list[str], model_id: str = "scripted"):
        self._responses = list(responses)
        self.model_id = model_id
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise ModelCallError("ScriptedModelClient has no responses left")
        return ModelResponse(
            self._responses.pop(0),
            self.model_id,
            tool_call_id=f"scripted-tool-call-{len(self.requests)}",
        )


class DeterministicModelClient:
    """Exercise the model boundary without pretending to be an LLM."""

    model_id = "deterministic-smoke-model"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        payload = json.loads(request.user_prompt)
        if request.tool.name == TACTICAL_TOOL_NAME:
            return ModelResponse(
                "{}",
                self.model_id,
                tool_call_id="deterministic-tactical-tool-call",
            )
        actions = payload["legal_actions"]
        moves = [action for action in actions if action["kind"] == "move"]
        selected = max(
            moves or actions,
            key=lambda action: action.get(
                "base_power",
                action.get("details", {}).get("base_power", 0),
            ),
        )
        content = json.dumps(
            {
                "action_id": selected["action_id"],
                "confidence": 1.0,
                "reason_codes": ["DAMAGE"],
                "short_rationale": "Choose the legal move with the highest base power.",
            }
        )
        return ModelResponse(
            content,
            self.model_id,
            tool_call_id="deterministic-tool-call",
        )


class ModelConfigurationError(ValueError):
    """Raised when a live model client is not configured safely."""


class OpenAICompatibleModelClient:
    """A thin async adapter for OpenAI-compatible Chat Completions APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        thinking_mode: str | None = None,
        client: Any | None = None,
    ):
        if not api_key:
            raise ModelConfigurationError("API key must not be empty")
        if not model:
            raise ModelConfigurationError("model must not be empty")
        if thinking_mode not in {None, "enabled", "disabled"}:
            raise ModelConfigurationError(
                "thinking_mode must be enabled, disabled, or unset"
            )
        _validate_base_url(base_url)

        self.model_id = model
        self.base_url = base_url.rstrip("/")
        self.thinking_mode = thinking_mode
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=0,
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        extra_body = (
            {"thinking": {"type": self.thinking_mode}}
            if self.thinking_mode is not None
            else None
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        for exchange in request.tool_history:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": exchange.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": exchange.tool_name,
                                    "arguments": exchange.arguments,
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": exchange.tool_call_id,
                        "content": exchange.result,
                    },
                ]
            )
        try:
            completion = await self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": request.tool.name,
                            "description": request.tool.description,
                            "parameters": request.tool.parameters,
                            "strict": request.tool.strict,
                        },
                    }
                ],
                tool_choice={
                    "type": "function",
                    "function": {"name": request.tool.name},
                },
                parallel_tool_calls=False,
                **({"extra_body": extra_body} if extra_body is not None else {}),
            )
        except Exception as exc:
            raise ModelCallError(f"{type(exc).__name__}: {exc}") from exc
        if not completion.choices:
            raise ModelCallError("Model response contained no choices")

        tool_calls = completion.choices[0].message.tool_calls or []
        if len(tool_calls) != 1:
            raise ModelCallError(
                "Model response must contain exactly one native tool call"
            )
        tool_call = tool_calls[0]
        if tool_call.type != "function":
            raise ModelCallError("Model response tool call must have type function")
        if tool_call.function.name != request.tool.name:
            raise ModelCallError(
                "Model called unexpected tool "
                f"{tool_call.function.name!r}; expected {request.tool.name!r}"
            )
        arguments = tool_call.function.arguments
        if not isinstance(arguments, str) or not arguments.strip():
            raise ModelCallError("Model tool call contained no arguments")
        if not isinstance(tool_call.id, str) or not tool_call.id.strip():
            raise ModelCallError("Model tool call contained no call ID")

        usage = None
        if completion.usage is not None:
            usage = TokenUsage(
                input_tokens=int(completion.usage.prompt_tokens),
                output_tokens=int(completion.usage.completion_tokens),
                total_tokens=int(completion.usage.total_tokens),
            )
        return ModelResponse(
            content=arguments,
            model_id=str(completion.model or self.model_id),
            response_id=str(completion.id) if completion.id else None,
            tool_call_id=tool_call.id,
            usage=usage,
        )

    async def aclose(self) -> None:
        await self._client.close()


def live_model_client_from_env(
    values: Mapping[str, str] | None = None,
) -> OpenAICompatibleModelClient:
    source = environ if values is None else values
    api_key = source.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise ModelConfigurationError(
            f"{API_KEY_ENV} is not set; load it from .env or export it"
        )
    return OpenAICompatibleModelClient(
        api_key=api_key,
        base_url=source.get(BASE_URL_ENV, DEFAULT_BASE_URL).strip(),
        model=source.get(MODEL_ENV, DEFAULT_MODEL).strip(),
        thinking_mode=source.get(THINKING_ENV, "").strip() or None,
    )


def _validate_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ModelConfigurationError("base_url must be an absolute HTTP or HTTPS URL")
    if parsed.scheme == "http" and parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ModelConfigurationError(
            "unencrypted HTTP base_url is allowed only for localhost"
        )
