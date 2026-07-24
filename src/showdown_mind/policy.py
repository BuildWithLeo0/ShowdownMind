from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from showdown_mind.actions import ActionCatalog
from showdown_mind.domain import (
    BattleSnapshot,
    PolicyDecision,
    PolicyResult,
    TokenUsage,
)
from showdown_mind.models import ModelCallError, ModelClient, ModelRequest
from showdown_mind.policy_input import (
    POLICY_INPUT_FORMATS,
    CompiledPolicyInput,
    compile_policy_input,
)

SYSTEM_PROMPT = """You choose one legal action in a Pokémon Showdown battle.
Use only the player-visible state in the request.
The input schema is full-v1, pruned-v1, or compact-v1.
In pruned-v1 and compact-v1, omitted optional values are false, empty, or unknown.
Return one JSON object and no other text.
The action_id must exactly match a legal action_id."""


class PolicyFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_responses: tuple[str, ...],
        model_ids: tuple[str, ...],
        response_ids: tuple[str, ...],
        usages: tuple[TokenUsage, ...],
        errors: tuple[str, ...],
        attempts: int,
        elapsed_seconds: float,
        policy_input: CompiledPolicyInput,
    ):
        super().__init__(message)
        self.raw_responses = raw_responses
        self.model_ids = model_ids
        self.response_ids = response_ids
        self.usages = usages
        self.errors = errors
        self.attempts = attempts
        self.elapsed_seconds = elapsed_seconds
        self.policy_input = policy_input


class SingleCallPolicy:
    """One normal model call, with at most one format-repair call."""

    def __init__(
        self,
        model_client: ModelClient,
        *,
        timeout_seconds: float = 45.0,
        max_repairs: int = 1,
        input_format: str = "pruned",
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_repairs not in (0, 1):
            raise ValueError("max_repairs must be 0 or 1")
        if input_format not in POLICY_INPUT_FORMATS:
            choices = ", ".join(POLICY_INPUT_FORMATS)
            raise ValueError(
                f"Unknown policy input format {input_format!r}; "
                f"choose one of: {choices}"
            )
        self._model_client = model_client
        self._timeout_seconds = timeout_seconds
        self._max_repairs = max_repairs
        self._input_format = input_format

    async def decide(
        self,
        snapshot: BattleSnapshot,
        catalog: ActionCatalog,
    ) -> PolicyResult:
        policy_input = compile_policy_input(snapshot, self._input_format)
        raw_responses: list[str] = []
        model_ids: list[str] = []
        response_ids: list[str] = []
        usages: list[TokenUsage] = []
        errors: list[str] = []
        started = time.monotonic()
        request = ModelRequest(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=policy_input.canonical_json(),
        )

        for attempt in range(self._max_repairs + 1):
            try:
                response = await asyncio.wait_for(
                    self._model_client.complete(request),
                    timeout=self._timeout_seconds,
                )
            except (TimeoutError, ModelCallError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                errors.append(error)
                if attempt >= self._max_repairs:
                    self._raise_failure(
                        cause=exc,
                        raw_responses=raw_responses,
                        model_ids=model_ids,
                        response_ids=response_ids,
                        usages=usages,
                        errors=errors,
                        attempts=attempt + 1,
                        started=started,
                        policy_input=policy_input,
                    )
                # No model output exists to repair. Retry the same request once.
                continue

            raw_responses.append(response.content)
            model_ids.append(response.model_id)
            if response.response_id is not None:
                response_ids.append(response.response_id)
            if response.usage is not None:
                usages.append(response.usage)
            try:
                decision = self._parse_decision(response.content, catalog)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                errors.append(error)
                if attempt >= self._max_repairs:
                    self._raise_failure(
                        cause=exc,
                        raw_responses=raw_responses,
                        model_ids=model_ids,
                        response_ids=response_ids,
                        usages=usages,
                        errors=errors,
                        attempts=attempt + 1,
                        started=started,
                        policy_input=policy_input,
                    )
                request = self._repair_request(
                    policy_input=policy_input,
                    invalid_response=response.content,
                    error=error,
                )
                continue

            return PolicyResult(
                decision=decision,
                attempts=attempt + 1,
                raw_responses=tuple(raw_responses),
                model_ids=tuple(model_ids),
                response_ids=tuple(response_ids),
                usages=tuple(usages),
                errors=tuple(errors),
                elapsed_seconds=round(time.monotonic() - started, 6),
                policy_input_format=policy_input.format_name,
                policy_input_hash=policy_input.fingerprint(),
                policy_input_characters=policy_input.characters,
                policy_input=policy_input.payload,
            )

        raise AssertionError("unreachable")

    @staticmethod
    def _raise_failure(
        *,
        cause: Exception,
        raw_responses: list[str],
        model_ids: list[str],
        response_ids: list[str],
        usages: list[TokenUsage],
        errors: list[str],
        attempts: int,
        started: float,
        policy_input: CompiledPolicyInput,
    ) -> None:
        raise PolicyFailure(
            "Policy did not produce a valid decision",
            raw_responses=tuple(raw_responses),
            model_ids=tuple(model_ids),
            response_ids=tuple(response_ids),
            usages=tuple(usages),
            errors=tuple(errors),
            attempts=attempts,
            elapsed_seconds=round(time.monotonic() - started, 6),
            policy_input=policy_input,
        ) from cause

    @staticmethod
    def _parse_decision(content: str, catalog: ActionCatalog) -> PolicyDecision:
        value = json.loads(content)
        if not isinstance(value, dict):
            raise TypeError("response must be a JSON object")

        action_id = value.get("action_id")
        if not isinstance(action_id, str):
            raise TypeError("action_id must be a string")
        if not catalog.contains(action_id):
            raise ValueError(f"action_id {action_id!r} is not currently legal")

        confidence = value.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise TypeError("confidence must be a number")
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")

        reason_codes = value.get("reason_codes", [])
        if not isinstance(reason_codes, list) or not all(
            isinstance(code, str) for code in reason_codes
        ):
            raise TypeError("reason_codes must be a list of strings")

        rationale = value.get("short_rationale", "")
        if not isinstance(rationale, str):
            raise TypeError("short_rationale must be a string")

        return PolicyDecision(
            action_id=action_id,
            confidence=confidence,
            reason_codes=tuple(reason_codes),
            short_rationale=rationale[:500],
        )

    @staticmethod
    def _repair_request(
        *,
        policy_input: CompiledPolicyInput,
        invalid_response: str,
        error: str,
    ) -> ModelRequest:
        actions = policy_input.payload["legal_actions"]
        valid_ids = [str(action["action_id"]) for action in actions]
        repair_payload: dict[str, Any] = {
            "error": error,
            "invalid_response": invalid_response,
            "battle": policy_input.payload,
            "valid_action_ids": valid_ids,
            "instruction": "Return corrected JSON only.",
        }
        return ModelRequest(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=json.dumps(
                repair_payload,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
