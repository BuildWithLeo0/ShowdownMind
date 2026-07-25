from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from showdown_mind.actions import ActionCatalog
from showdown_mind.domain import (
    BattleSnapshot,
    PolicyDecision,
    PolicyResult,
    TokenUsage,
)
from showdown_mind.models import (
    ACTION_TOOL_NAME,
    TACTICAL_TOOL_NAME,
    ModelCallError,
    ModelClient,
    ModelRequest,
    ModelTool,
    ModelResponse,
    ToolExchange,
)
from showdown_mind.policy_input import (
    POLICY_INPUT_FORMATS,
    CompiledPolicyInput,
    compile_policy_input,
)
from showdown_mind.tactics import (
    TacticalAdvisor,
    compact_tactical_analysis_for_model,
)

MAX_RATIONALE_CHARACTERS = 240
POLICY_MODES = ("direct", "tactical-tool")
REASON_CODES = (
    "DAMAGE",
    "ACCURACY",
    "STAB",
    "SURVIVAL",
    "TYPE_MATCHUP",
    "STATUS",
    "SETUP",
    "SPEED_CONTROL",
    "RESOURCE_PRESERVATION",
    "FORCED_SWITCH",
    "INFORMATION",
    "WEATHER",
    "OTHER",
)
REASON_CODE_ALIASES = {
    "KO": "DAMAGE",
    "KO_PROBABILITY": "DAMAGE",
}

SYSTEM_PROMPT = """You choose one legal action in a Pokémon Showdown battle.
Use only the player-visible state in the request.
The input schema is full-v1, pruned-v1, or compact-v1.
In pruned-v1 and compact-v1, omitted optional values are false, empty, or unknown.
Call choose_battle_action exactly once.
The short_rationale must be one concise public sentence explaining the choice,
not private chain-of-thought."""

TACTICAL_SYSTEM_PROMPT = """You choose one legal action in a Pokémon Showdown battle.
Use only the player-visible state and tool result in the request.
First call analyze_battle_options exactly once. After receiving its result,
call choose_battle_action exactly once.
Treat tactical values as estimates, not hidden facts.
The short_rationale must be one concise public sentence explaining the choice,
not private chain-of-thought."""


class PolicyFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_responses: tuple[str, ...],
        model_ids: tuple[str, ...],
        response_ids: tuple[str, ...],
        tool_call_ids: tuple[str, ...],
        usages: tuple[TokenUsage, ...],
        errors: tuple[str, ...],
        attempts: int,
        elapsed_seconds: float,
        policy_input: CompiledPolicyInput,
        model_calls: int | None = None,
        expected_model_calls: int = 1,
        tool_names: tuple[str, ...] = (),
        tool_executions: tuple[dict[str, Any], ...] = (),
        tactical_analysis: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.raw_responses = raw_responses
        self.model_ids = model_ids
        self.response_ids = response_ids
        self.tool_call_ids = tool_call_ids
        self.usages = usages
        self.errors = errors
        self.attempts = attempts
        self.elapsed_seconds = elapsed_seconds
        self.policy_input = policy_input
        self.model_calls = attempts if model_calls is None else model_calls
        self.expected_model_calls = expected_model_calls
        self.tool_names = tool_names
        self.tool_executions = tool_executions
        self.tactical_analysis = tactical_analysis or {}


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
        *,
        battle: Any | None = None,
    ) -> PolicyResult:
        policy_input = compile_policy_input(snapshot, self._input_format)
        raw_responses: list[str] = []
        model_ids: list[str] = []
        response_ids: list[str] = []
        tool_call_ids: list[str] = []
        usages: list[TokenUsage] = []
        errors: list[str] = []
        started = time.monotonic()
        tool = self._action_tool(catalog)
        request = self._model_request(
            user_prompt=policy_input.canonical_json(),
            tool=tool,
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
                        tool_call_ids=tool_call_ids,
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
            if response.tool_call_id is not None:
                tool_call_ids.append(response.tool_call_id)
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
                        tool_call_ids=tool_call_ids,
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
                    tool=tool,
                )
                continue

            return PolicyResult(
                decision=decision,
                attempts=attempt + 1,
                raw_responses=tuple(raw_responses),
                model_ids=tuple(model_ids),
                response_ids=tuple(response_ids),
                tool_call_ids=tuple(tool_call_ids),
                usages=tuple(usages),
                errors=tuple(errors),
                elapsed_seconds=round(time.monotonic() - started, 6),
                policy_input_format=policy_input.format_name,
                policy_input_hash=policy_input.fingerprint(),
                policy_input_characters=policy_input.characters,
                policy_input=policy_input.payload,
                model_calls=attempt + 1,
                expected_model_calls=1,
                tool_names=tuple(ACTION_TOOL_NAME for _ in tool_call_ids),
            )

        raise AssertionError("unreachable")

    @staticmethod
    def _raise_failure(
        *,
        cause: Exception,
        raw_responses: list[str],
        model_ids: list[str],
        response_ids: list[str],
        tool_call_ids: list[str],
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
            tool_call_ids=tuple(tool_call_ids),
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

        required = {
            "action_id",
            "confidence",
            "reason_codes",
            "short_rationale",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise TypeError(f"missing required fields: {', '.join(missing)}")
        extra = sorted(set(value).difference(required))
        if extra:
            raise TypeError(f"unexpected fields: {', '.join(extra)}")

        action_id = value["action_id"]
        if not isinstance(action_id, str):
            raise TypeError("action_id must be a string")
        if not catalog.contains(action_id):
            raise ValueError(f"action_id {action_id!r} is not currently legal")

        confidence = value["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("confidence must be a number")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        reason_codes = value["reason_codes"]
        if not isinstance(reason_codes, list) or not all(
            isinstance(code, str) for code in reason_codes
        ):
            raise TypeError("reason_codes must be a list of strings")
        reason_codes = list(
            dict.fromkeys(
                REASON_CODE_ALIASES.get(code, code)
                for code in reason_codes
            )
        )
        if not 1 <= len(reason_codes) <= 3:
            raise ValueError("reason_codes must contain between 1 and 3 values")
        invalid_codes = sorted(set(reason_codes).difference(REASON_CODES))
        if invalid_codes:
            raise ValueError(f"unknown reason_codes: {', '.join(invalid_codes)}")

        rationale = value["short_rationale"]
        if not isinstance(rationale, str):
            raise TypeError("short_rationale must be a string")
        rationale = rationale.strip()
        if not rationale:
            raise ValueError("short_rationale must not be empty")
        if len(rationale) > MAX_RATIONALE_CHARACTERS:
            rationale = rationale[:MAX_RATIONALE_CHARACTERS].rstrip()

        return PolicyDecision(
            action_id=action_id,
            confidence=confidence,
            reason_codes=tuple(reason_codes),
            short_rationale=rationale,
        )

    @staticmethod
    def _repair_request(
        *,
        policy_input: CompiledPolicyInput,
        invalid_response: str,
        error: str,
        tool: ModelTool,
        tool_history: tuple[ToolExchange, ...] = (),
        system_prompt: str = SYSTEM_PROMPT,
    ) -> ModelRequest:
        actions = policy_input.payload["legal_actions"]
        valid_ids = [str(action["action_id"]) for action in actions]
        repair_payload: dict[str, Any] = {
            "error": error,
            "invalid_response": invalid_response,
            "battle": policy_input.payload,
            "valid_action_ids": valid_ids,
            "instruction": f"Call {ACTION_TOOL_NAME} once with corrected arguments.",
        }
        return SingleCallPolicy._model_request(
            user_prompt=json.dumps(
                repair_payload,
                ensure_ascii=False,
                sort_keys=True,
            ),
            tool=tool,
            system_prompt=system_prompt,
            tool_history=tool_history,
        )

    @staticmethod
    def _model_request(
        *,
        user_prompt: str,
        tool: ModelTool,
        system_prompt: str = SYSTEM_PROMPT,
        tool_history: tuple[ToolExchange, ...] = (),
    ) -> ModelRequest:
        return ModelRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tool=tool,
            tool_history=tool_history,
        )

    @staticmethod
    def _action_tool(catalog: ActionCatalog) -> ModelTool:
        valid_ids = [action.action_id for action in catalog.actions]
        return ModelTool(
            name=ACTION_TOOL_NAME,
            description=(
                "Select exactly one currently legal Pokémon Showdown action and "
                "give a brief public explanation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "string",
                        "enum": valid_ids,
                        "description": "One action ID from the current legal list.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": (
                            "Self-reported confidence from 0 to 1; this is not "
                            "guaranteed to be calibrated."
                        ),
                    },
                    "reason_codes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(REASON_CODES),
                        },
                        "minItems": 1,
                        "maxItems": 3,
                        "description": "One to three concise factors behind the choice.",
                    },
                    "short_rationale": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_RATIONALE_CHARACTERS,
                        "description": (
                            "One concise public sentence explaining the choice; "
                            "do not provide private chain-of-thought."
                        ),
                    },
                },
                "required": [
                    "action_id",
                    "confidence",
                    "reason_codes",
                    "short_rationale",
                ],
                "additionalProperties": False,
            },
        )


@dataclass
class _PolicyTrace:
    raw_responses: list[str] = field(default_factory=list)
    model_ids: list[str] = field(default_factory=list)
    response_ids: list[str] = field(default_factory=list)
    tool_call_ids: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    usages: list[TokenUsage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    tool_executions: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0

    def record(self, response: ModelResponse, tool_name: str) -> None:
        self.raw_responses.append(response.content)
        self.model_ids.append(response.model_id)
        if response.response_id is not None:
            self.response_ids.append(response.response_id)
        if response.tool_call_id is not None:
            self.tool_call_ids.append(response.tool_call_id)
            self.tool_names.append(tool_name)
        if response.usage is not None:
            self.usages.append(response.usage)


class TacticalToolPolicy(SingleCallPolicy):
    """A bounded native tool workflow followed by one validated action call."""

    expected_model_calls = 2

    def __init__(
        self,
        model_client: ModelClient,
        *,
        tactical_advisor: TacticalAdvisor | None = None,
        timeout_seconds: float = 45.0,
        max_repairs: int = 1,
        input_format: str = "pruned",
    ):
        super().__init__(
            model_client,
            timeout_seconds=timeout_seconds,
            max_repairs=max_repairs,
            input_format=input_format,
        )
        self._tactical_advisor = tactical_advisor or TacticalAdvisor()

    async def decide(
        self,
        snapshot: BattleSnapshot,
        catalog: ActionCatalog,
        *,
        battle: Any | None = None,
    ) -> PolicyResult:
        if battle is None:
            raise ValueError("tactical-tool policy requires the current battle")

        policy_input = compile_policy_input(snapshot, self._input_format)
        trace = _PolicyTrace()
        started = time.monotonic()
        retries = 0
        tactical_analysis: dict[str, Any] = {}

        analysis_tool = self._analysis_tool()
        analysis_request = self._model_request(
            user_prompt=policy_input.canonical_json(),
            tool=analysis_tool,
            system_prompt=TACTICAL_SYSTEM_PROMPT,
        )
        analysis_response: ModelResponse | None = None
        for attempt in range(self._max_repairs + 1):
            try:
                response = await self._complete_traced(
                    analysis_request,
                    trace,
                    TACTICAL_TOOL_NAME,
                )
                self._parse_analysis_arguments(response.content)
            except (
                TimeoutError,
                ModelCallError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                trace.errors.append(f"{type(exc).__name__}: {exc}")
                if attempt >= self._max_repairs:
                    self._raise_tactical_failure(
                        cause=exc,
                        trace=trace,
                        retries=retries,
                        started=started,
                        policy_input=policy_input,
                        tactical_analysis=tactical_analysis,
                    )
                retries += 1
                continue
            analysis_response = response
            break

        if analysis_response is None or analysis_response.tool_call_id is None:
            raise AssertionError("validated tactical tool call must have an ID")

        tactical_analysis = self._tactical_advisor.analyze(battle, catalog)
        model_analysis = compact_tactical_analysis_for_model(
            tactical_analysis
        )
        result_json = json.dumps(
            model_analysis,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        exchange = ToolExchange(
            tool_call_id=analysis_response.tool_call_id,
            tool_name=TACTICAL_TOOL_NAME,
            arguments=analysis_response.content,
            result=result_json,
        )
        trace.tool_executions.append(
            {
                "tool_call_id": exchange.tool_call_id,
                "tool_name": exchange.tool_name,
                "arguments": {},
                "result": tactical_analysis,
                "audit_result_characters": len(
                    json.dumps(
                        tactical_analysis,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ),
                "model_result_characters": len(result_json),
            }
        )

        action_tool = self._action_tool(catalog)
        action_request = self._model_request(
            user_prompt=policy_input.canonical_json(),
            tool=action_tool,
            system_prompt=TACTICAL_SYSTEM_PROMPT,
            tool_history=(exchange,),
        )
        for attempt in range(self._max_repairs + 1):
            try:
                response = await self._complete_traced(
                    action_request,
                    trace,
                    ACTION_TOOL_NAME,
                )
                decision = self._parse_decision(response.content, catalog)
            except (TimeoutError, ModelCallError) as exc:
                trace.errors.append(f"{type(exc).__name__}: {exc}")
                if attempt >= self._max_repairs:
                    self._raise_tactical_failure(
                        cause=exc,
                        trace=trace,
                        retries=retries,
                        started=started,
                        policy_input=policy_input,
                        tactical_analysis=tactical_analysis,
                    )
                retries += 1
                continue
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                trace.errors.append(error)
                if attempt >= self._max_repairs:
                    self._raise_tactical_failure(
                        cause=exc,
                        trace=trace,
                        retries=retries,
                        started=started,
                        policy_input=policy_input,
                        tactical_analysis=tactical_analysis,
                    )
                retries += 1
                action_request = self._repair_request(
                    policy_input=policy_input,
                    invalid_response=response.content,
                    error=error,
                    tool=action_tool,
                    tool_history=(exchange,),
                    system_prompt=TACTICAL_SYSTEM_PROMPT,
                )
                continue

            return PolicyResult(
                decision=decision,
                attempts=1 + retries,
                raw_responses=tuple(trace.raw_responses),
                model_ids=tuple(trace.model_ids),
                response_ids=tuple(trace.response_ids),
                tool_call_ids=tuple(trace.tool_call_ids),
                usages=tuple(trace.usages),
                errors=tuple(trace.errors),
                elapsed_seconds=round(time.monotonic() - started, 6),
                policy_input_format=policy_input.format_name,
                policy_input_hash=policy_input.fingerprint(),
                policy_input_characters=policy_input.characters,
                policy_input=policy_input.payload,
                model_calls=trace.model_calls,
                expected_model_calls=self.expected_model_calls,
                tool_names=tuple(trace.tool_names),
                tool_executions=tuple(trace.tool_executions),
                tactical_analysis=tactical_analysis,
            )

        raise AssertionError("unreachable")

    async def _complete_traced(
        self,
        request: ModelRequest,
        trace: _PolicyTrace,
        tool_name: str,
    ) -> ModelResponse:
        trace.model_calls += 1
        response = await asyncio.wait_for(
            self._model_client.complete(request),
            timeout=self._timeout_seconds,
        )
        trace.record(response, tool_name)
        return response

    @staticmethod
    def _parse_analysis_arguments(content: str) -> None:
        value = json.loads(content)
        if not isinstance(value, dict):
            raise TypeError("tactical tool arguments must be a JSON object")
        if value:
            extra = ", ".join(sorted(str(key) for key in value))
            raise ValueError(f"tactical tool accepts no arguments: {extra}")

    @staticmethod
    def _analysis_tool() -> ModelTool:
        return ModelTool(
            name=TACTICAL_TOOL_NAME,
            description=(
                "Calculate deterministic tactical facts for every currently "
                "legal move and switch using only player-visible battle state."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        )

    @staticmethod
    def _raise_tactical_failure(
        *,
        cause: Exception,
        trace: _PolicyTrace,
        retries: int,
        started: float,
        policy_input: CompiledPolicyInput,
        tactical_analysis: dict[str, Any],
    ) -> None:
        raise PolicyFailure(
            "Tactical policy did not produce a valid decision",
            raw_responses=tuple(trace.raw_responses),
            model_ids=tuple(trace.model_ids),
            response_ids=tuple(trace.response_ids),
            tool_call_ids=tuple(trace.tool_call_ids),
            usages=tuple(trace.usages),
            errors=tuple(trace.errors),
            attempts=1 + retries,
            elapsed_seconds=round(time.monotonic() - started, 6),
            policy_input=policy_input,
            model_calls=trace.model_calls,
            expected_model_calls=TacticalToolPolicy.expected_model_calls,
            tool_names=tuple(trace.tool_names),
            tool_executions=tuple(trace.tool_executions),
            tactical_analysis=tactical_analysis,
        ) from cause


def make_policy(
    model_client: ModelClient,
    *,
    policy_mode: str = "direct",
    input_format: str = "pruned",
) -> SingleCallPolicy:
    if policy_mode == "direct":
        return SingleCallPolicy(model_client, input_format=input_format)
    if policy_mode == "tactical-tool":
        return TacticalToolPolicy(model_client, input_format=input_format)
    choices = ", ".join(POLICY_MODES)
    raise ValueError(f"unknown policy mode {policy_mode!r}; choose one of: {choices}")
