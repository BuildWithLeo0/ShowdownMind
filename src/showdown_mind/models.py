from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelRequest:
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class ModelResponse:
    content: str
    model_id: str


class ModelClient(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one model response for one policy attempt."""


class ScriptedModelClient:
    """A deterministic model double for policy tests."""

    def __init__(self, responses: list[str], model_id: str = "scripted"):
        self._responses = list(responses)
        self.model_id = model_id
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("ScriptedModelClient has no responses left")
        return ModelResponse(self._responses.pop(0), self.model_id)


class DeterministicModelClient:
    """Exercise the model boundary without pretending to be an LLM."""

    model_id = "deterministic-smoke-model"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        payload = json.loads(request.user_prompt)
        actions = payload["legal_actions"]
        moves = [action for action in actions if action["kind"] == "move"]
        selected = max(
            moves or actions,
            key=lambda action: action.get("details", {}).get("base_power", 0),
        )
        content = json.dumps(
            {
                "action_id": selected["action_id"],
                "confidence": 1.0,
                "reason_codes": ["SMOKE_TEST"],
                "short_rationale": "Deterministic model boundary validation.",
            }
        )
        return ModelResponse(content, self.model_id)
