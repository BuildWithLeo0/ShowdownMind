from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from poke_env.data import to_id_str

from showdown_mind.domain import BattleSnapshot, TokenUsage
from showdown_mind.models import (
    PLAN_TOOL_NAME,
    ModelCallError,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelTool,
)


PLAN_SCHEMA = "battle-plan-v1"
RISK_POSTURES = ("conservative", "balanced", "aggressive")
REPLAN_TRIGGERS = (
    "preserve_fainted",
    "target_fainted",
    "opponent_tera",
    "own_tera",
    "belief_changed",
)
MAX_PLAN_TEXT_CHARACTERS = 180

PLANNER_SYSTEM_PROMPT = """You maintain a concise plan for one Pokémon Showdown battle.
Use only the player-visible state, memory, beliefs, and estimates in the request.
The plan should guide several future turns without prescribing a fixed move sequence.
Call update_battle_plan exactly once. Return public summaries, not chain-of-thought."""


@dataclass(frozen=True)
class BattlePlan:
    version: int
    created_turn: int
    win_condition: str
    preserve: tuple[str, ...]
    priority_targets: tuple[str, ...]
    tera_policy: str
    risk_posture: str
    replan_triggers: tuple[str, ...]
    schema: str = PLAN_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerResult:
    plan: BattlePlan
    raw_responses: tuple[str, ...]
    model_ids: tuple[str, ...]
    response_ids: tuple[str, ...]
    tool_call_ids: tuple[str, ...]
    usages: tuple[TokenUsage, ...]
    errors: tuple[str, ...]
    model_calls: int
    elapsed_seconds: float


class PlannerFailure(RuntimeError):
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
        model_calls: int,
        elapsed_seconds: float,
    ):
        super().__init__(message)
        self.raw_responses = raw_responses
        self.model_ids = model_ids
        self.response_ids = response_ids
        self.tool_call_ids = tool_call_ids
        self.usages = usages
        self.errors = errors
        self.model_calls = model_calls
        self.elapsed_seconds = elapsed_seconds


class BattlePlanner:
    def __init__(
        self,
        model_client: ModelClient,
        *,
        timeout_seconds: float = 45.0,
        max_repairs: int = 1,
    ):
        self._model_client = model_client
        self._timeout_seconds = timeout_seconds
        self._max_repairs = max_repairs

    async def update(
        self,
        *,
        snapshot: BattleSnapshot,
        context: dict[str, Any],
        previous: BattlePlan | None,
    ) -> PlannerResult:
        raw_responses: list[str] = []
        model_ids: list[str] = []
        response_ids: list[str] = []
        tool_call_ids: list[str] = []
        usages: list[TokenUsage] = []
        errors: list[str] = []
        started = time.monotonic()
        tool = _plan_tool(snapshot)
        request = ModelRequest(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=json.dumps(
                {
                    "schema": "planner-context-v1",
                    **context,
                    "previous_plan": previous.to_dict() if previous else None,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            tool=tool,
        )
        for attempt in range(self._max_repairs + 1):
            try:
                response = await asyncio.wait_for(
                    self._model_client.complete(request),
                    timeout=self._timeout_seconds,
                )
                _record_response(
                    response,
                    raw_responses,
                    model_ids,
                    response_ids,
                    tool_call_ids,
                    usages,
                )
                plan = _parse_plan(
                    response.content,
                    snapshot=snapshot,
                    version=(previous.version + 1 if previous else 1),
                )
            except (
                TimeoutError,
                ModelCallError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                error = f"{type(exc).__name__}: {exc}"
                errors.append(error)
                if attempt >= self._max_repairs:
                    raise PlannerFailure(
                        "Planner did not produce a valid battle plan",
                        raw_responses=tuple(raw_responses),
                        model_ids=tuple(model_ids),
                        response_ids=tuple(response_ids),
                        tool_call_ids=tuple(tool_call_ids),
                        usages=tuple(usages),
                        errors=tuple(errors),
                        model_calls=attempt + 1,
                        elapsed_seconds=round(time.monotonic() - started, 6),
                    ) from exc
                request = ModelRequest(
                    system_prompt=PLANNER_SYSTEM_PROMPT,
                    user_prompt=json.dumps(
                        {
                            "error": error,
                            "planner_context": context,
                            "previous_plan": (
                                previous.to_dict() if previous else None
                            ),
                            "instruction": (
                                f"Call {PLAN_TOOL_NAME} once with corrected arguments."
                            ),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    tool=tool,
                )
                continue
            return PlannerResult(
                plan=plan,
                raw_responses=tuple(raw_responses),
                model_ids=tuple(model_ids),
                response_ids=tuple(response_ids),
                tool_call_ids=tuple(tool_call_ids),
                usages=tuple(usages),
                errors=tuple(errors),
                model_calls=attempt + 1,
                elapsed_seconds=round(time.monotonic() - started, 6),
            )
        raise AssertionError("unreachable")


def neutral_plan(turn: int) -> BattlePlan:
    return BattlePlan(
        version=0,
        created_turn=turn,
        win_condition="Preserve options and improve the current matchup.",
        preserve=(),
        priority_targets=(),
        tera_policy="Hold Tera unless it prevents an immediate loss.",
        risk_posture="balanced",
        replan_triggers=(
            "preserve_fainted",
            "target_fainted",
            "opponent_tera",
            "own_tera",
            "belief_changed",
        ),
    )


def _parse_plan(
    content: str,
    *,
    snapshot: BattleSnapshot,
    version: int,
) -> BattlePlan:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise TypeError("plan response must be a JSON object")
    required = {
        "win_condition",
        "preserve",
        "priority_targets",
        "tera_policy",
        "risk_posture",
        "replan_triggers",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise TypeError(f"missing plan fields: {', '.join(missing)}")
    extra = sorted(set(value).difference(required))
    if extra:
        raise TypeError(f"unexpected plan fields: {', '.join(extra)}")

    own_species = {
        to_id_str(str(pokemon.get("species") or ""))
        for pokemon in snapshot.own_side.get("team", [])
        if isinstance(pokemon, dict) and not pokemon.get("fainted")
    }
    opponent_species = {
        to_id_str(str(pokemon.get("species") or ""))
        for pokemon in snapshot.opponent_side.get("revealed_team", [])
        if isinstance(pokemon, dict) and not pokemon.get("fainted")
    }
    preserve = _species_list(value["preserve"], "preserve", own_species)
    targets = _species_list(
        value["priority_targets"],
        "priority_targets",
        opponent_species,
    )
    risk = value["risk_posture"]
    if risk not in RISK_POSTURES:
        raise ValueError(f"unknown risk posture: {risk}")
    triggers = value["replan_triggers"]
    if not isinstance(triggers, list) or not all(
        isinstance(item, str) for item in triggers
    ):
        raise TypeError("replan_triggers must be a list of strings")
    unknown_triggers = sorted(set(triggers).difference(REPLAN_TRIGGERS))
    if unknown_triggers:
        raise ValueError(
            f"unknown replan triggers: {', '.join(unknown_triggers)}"
        )
    required_triggers = []
    if preserve:
        required_triggers.append("preserve_fainted")
    if targets:
        required_triggers.append("target_fainted")
    return BattlePlan(
        version=version,
        created_turn=snapshot.turn,
        win_condition=_plan_text(value["win_condition"], "win_condition"),
        preserve=preserve,
        priority_targets=targets,
        tera_policy=_plan_text(value["tera_policy"], "tera_policy"),
        risk_posture=risk,
        replan_triggers=tuple(
            dict.fromkeys([*triggers, *required_triggers])
        ),
    )


def _plan_tool(snapshot: BattleSnapshot) -> ModelTool:
    own_species = sorted(
        {
            to_id_str(str(pokemon.get("species") or ""))
            for pokemon in snapshot.own_side.get("team", [])
            if isinstance(pokemon, dict)
            and pokemon.get("species")
            and not pokemon.get("fainted")
        }
    )
    opponent_species = sorted(
        {
            to_id_str(str(pokemon.get("species") or ""))
            for pokemon in snapshot.opponent_side.get("revealed_team", [])
            if isinstance(pokemon, dict)
            and pokemon.get("species")
            and not pokemon.get("fainted")
        }
    )
    return ModelTool(
        name=PLAN_TOOL_NAME,
        description=(
            "Update the concise, multi-turn plan for the current battle without "
            "choosing the current move."
        ),
        parameters={
            "type": "object",
            "properties": {
                "win_condition": _text_schema(),
                "preserve": {
                    "type": "array",
                    "items": {"type": "string", "enum": own_species},
                    "maxItems": 3,
                },
                "priority_targets": {
                    "type": "array",
                    "items": {"type": "string", "enum": opponent_species},
                    "maxItems": 3,
                },
                "tera_policy": _text_schema(),
                "risk_posture": {
                    "type": "string",
                    "enum": list(RISK_POSTURES),
                },
                "replan_triggers": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(REPLAN_TRIGGERS),
                    },
                    "minItems": 1,
                    "maxItems": len(REPLAN_TRIGGERS),
                },
            },
            "required": [
                "win_condition",
                "preserve",
                "priority_targets",
                "tera_policy",
                "risk_posture",
                "replan_triggers",
            ],
            "additionalProperties": False,
        },
    )


def _record_response(
    response: ModelResponse,
    raw_responses: list[str],
    model_ids: list[str],
    response_ids: list[str],
    tool_call_ids: list[str],
    usages: list[TokenUsage],
) -> None:
    raw_responses.append(response.content)
    model_ids.append(response.model_id)
    if response.response_id is not None:
        response_ids.append(response.response_id)
    if response.tool_call_id is not None:
        tool_call_ids.append(response.tool_call_id)
    if response.usage is not None:
        usages.append(response.usage)


def _species_list(
    value: Any,
    field_name: str,
    visible_species: set[str],
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise TypeError(f"{field_name} must be a list of strings")
    normalized = tuple(
        dict.fromkeys(to_id_str(item) for item in value if to_id_str(item))
    )
    invalid = sorted(set(normalized).difference(visible_species))
    if invalid:
        raise ValueError(
            f"{field_name} contains non-visible species: {', '.join(invalid)}"
        )
    if len(normalized) > 3:
        raise ValueError(f"{field_name} must contain at most 3 species")
    return normalized


def _plan_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text[:MAX_PLAN_TEXT_CHARACTERS].rstrip()


def _text_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": MAX_PLAN_TEXT_CHARACTERS,
    }
