from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from poke_env.data import to_id_str

from showdown_mind.actions import ActionCatalog
from showdown_mind.battle_memory import (
    BattleMemory,
    OpponentPrediction,
    PREDICTION_KINDS,
    PokeEnvEventAdapter,
)
from showdown_mind.beliefs import (
    BeliefState,
    RuleBeliefTracker,
    belief_changes,
)
from showdown_mind.domain import (
    BattleSnapshot,
    PolicyDecision,
    PolicyResult,
    TokenUsage,
)
from showdown_mind.models import (
    ACTION_TOOL_NAME,
    PLAN_TOOL_NAME,
    TACTICAL_TOOL_NAME,
    ModelCallError,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelTool,
)
from showdown_mind.planning import (
    BattlePlan,
    BattlePlanner,
    PlannerFailure,
    neutral_plan,
)
from showdown_mind.policy import (
    MAX_RATIONALE_CHARACTERS,
    REASON_CODE_ALIASES,
    REASON_CODES,
    PolicyFailure,
    SingleCallPolicy,
)
from showdown_mind.policy_input import CompiledPolicyInput, compile_policy_input
from showdown_mind.tactics import (
    TacticalAdvisor,
    compact_tactical_analysis_for_model,
)


CONTROLLED_INPUT_SCHEMA = "controlled-agent-v1"
MAX_CONTEXT_CHARACTERS = 24_000
CONTROLLED_SYSTEM_PROMPT = """You choose one legal action in a Pokémon Showdown battle.
Use only the player-visible state, memory, hypotheses, plan, and estimates provided.
Hypotheses are uncertain and must not be treated as hidden facts.
Follow the battle plan when useful, but prioritize a legal action now.
Call choose_battle_action exactly once. Give one concise public rationale, not chain-of-thought."""


@dataclass
class ControlledBattleState:
    memory: BattleMemory
    belief: BeliefState | None = None
    plan: BattlePlan | None = None
    pending_replan: bool = False


class ControlledAgentPolicy(SingleCallPolicy):
    """Automatic internal tools, optional planning, and one final action call."""

    def __init__(
        self,
        model_client: ModelClient,
        *,
        tactical_advisor: TacticalAdvisor | None = None,
        belief_tracker: RuleBeliefTracker | None = None,
        event_adapter: PokeEnvEventAdapter | None = None,
        planner: BattlePlanner | None = None,
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
        self._belief_tracker = belief_tracker or RuleBeliefTracker()
        self._event_adapter = event_adapter or PokeEnvEventAdapter()
        self._planner = planner or BattlePlanner(
            model_client,
            timeout_seconds=timeout_seconds,
            max_repairs=max_repairs,
        )
        self._states: dict[str, ControlledBattleState] = {}

    async def decide(
        self,
        snapshot: BattleSnapshot,
        catalog: ActionCatalog,
        *,
        battle: Any | None = None,
    ) -> PolicyResult:
        if battle is None:
            raise ValueError("controlled-agent policy requires the current battle")
        state = self._states.setdefault(
            snapshot.battle_id,
            ControlledBattleState(memory=BattleMemory(snapshot.battle_id)),
        )
        enrichment_errors: list[str] = []
        try:
            new_events = self._event_adapter.consume(battle, state.memory)
            state.memory.consume(new_events)
        except Exception as exc:
            enrichment_errors.append(_component_error("event_adapter", exc))
            new_events = ()
        previous_belief = state.belief
        try:
            current_belief = self._belief_tracker.update(snapshot, state.memory)
            changes = belief_changes(previous_belief, current_belief)
        except Exception as exc:
            enrichment_errors.append(_component_error("belief_tracker", exc))
            current_belief = previous_belief or BeliefState(
                battle_id=snapshot.battle_id,
                updated_turn=snapshot.turn,
                unavailable_priors=("belief_tracker_failed",),
            )
            changes = ()
        state.belief = current_belief

        active_subject = (
            f"opponent:{to_id_str(str(snapshot.opponent_side.get('active') or ''))}"
        )
        candidate_moves = current_belief.possible_move_ids(active_subject)
        belief_view = current_belief.model_view(
            active_subject=active_subject,
        )
        try:
            tactical_analysis = self._tactical_advisor.analyze(
                battle,
                catalog,
                opponent_candidate_move_ids=candidate_moves,
            )
        except Exception as exc:
            enrichment_errors.append(_component_error("tactical_advisor", exc))
            tactical_analysis = _unavailable_tactical_analysis(
                catalog,
                error_type=type(exc).__name__,
            )
        tactical_view = compact_tactical_analysis_for_model(tactical_analysis)
        battle_input = compile_policy_input(snapshot, self._input_format)
        event_values = tuple(event.to_dict() for event in new_events)
        trigger = _plan_trigger(state, new_events=event_values, changes=changes)

        trace = _ControlledTrace()
        planner_errors: list[str] = []
        planner_usages: list[TokenUsage] = []
        planner_model_calls = 0
        planner_elapsed = 0.0
        plan_update: dict[str, Any] = {}
        if trigger:
            try:
                planner_result = await self._planner.update(
                    snapshot=snapshot,
                    context={
                        "battle": battle_input.payload,
                        "memory": state.memory.model_view(),
                        "beliefs": belief_view,
                        "tactical": tactical_view,
                        "trigger": trigger,
                    },
                    previous=state.plan,
                )
                state.plan = planner_result.plan
                plan_update = planner_result.plan.to_dict()
                trace.add_planner_result(planner_result)
                planner_errors.extend(planner_result.errors)
                planner_usages.extend(planner_result.usages)
                planner_model_calls = planner_result.model_calls
                planner_elapsed = planner_result.elapsed_seconds
            except PlannerFailure as exc:
                trace.add_planner_failure(exc)
                planner_errors.extend(exc.errors)
                planner_usages.extend(exc.usages)
                planner_model_calls = exc.model_calls
                planner_elapsed = exc.elapsed_seconds
                if state.plan is None:
                    state.plan = neutral_plan(snapshot.turn)
                    plan_update = state.plan.to_dict()
            state.pending_replan = False
        if state.plan is None:
            state.plan = neutral_plan(snapshot.turn)

        try:
            policy_input = _compile_controlled_input(
                battle=battle_input.payload,
                battle_input_format=battle_input.format_name,
                memory=state.memory.model_view(),
                beliefs=belief_view,
                plan=state.plan.to_dict(),
                tactical=tactical_view,
            )
        except ValueError as exc:
            if self._input_format == "compact":
                raise
            enrichment_errors.append(_component_error("context_compaction", exc))
            battle_input = compile_policy_input(snapshot, "compact")
            policy_input = _compile_controlled_input(
                battle=battle_input.payload,
                battle_input_format=battle_input.format_name,
                memory=state.memory.model_view(),
                beliefs=belief_view,
                plan=state.plan.to_dict(),
                tactical=tactical_view,
            )
        tool_executions = (
            {
                "tool_call_id": "",
                "tool_name": TACTICAL_TOOL_NAME,
                "execution_kind": "internal",
                "arguments": {
                    "opponent_candidate_move_ids": list(candidate_moves)
                },
                "result": tactical_analysis,
            },
        )
        action_tool = _controlled_action_tool(catalog)
        request = ModelRequest(
            system_prompt=CONTROLLED_SYSTEM_PROMPT,
            user_prompt=policy_input.canonical_json(),
            tool=action_tool,
        )
        started = time.monotonic()
        action_errors: list[str] = []
        action_calls = 0
        for attempt in range(self._max_repairs + 1):
            action_calls += 1
            try:
                response = await asyncio.wait_for(
                    self._model_client.complete(request),
                    timeout=self._timeout_seconds,
                )
                trace.add_response(response, ACTION_TOOL_NAME)
                decision = _parse_controlled_decision(response.content, catalog)
            except (TimeoutError, ModelCallError) as exc:
                action_errors.append(f"{type(exc).__name__}: {exc}")
                if attempt >= self._max_repairs:
                    self._raise_failure(
                        cause=exc,
                        snapshot=snapshot,
                        state=state,
                        event_values=event_values,
                        changes=changes,
                        trigger=trigger,
                        plan_update=plan_update,
                        tactical_analysis=tactical_analysis,
                        policy_input=policy_input,
                        trace=trace,
                        action_errors=action_errors,
                        action_attempts=attempt + 1,
                        action_calls=action_calls,
                        planner_model_calls=planner_model_calls,
                        planner_usages=planner_usages,
                        planner_errors=planner_errors,
                        planner_elapsed=planner_elapsed,
                        enrichment_errors=enrichment_errors,
                        tool_executions=tool_executions,
                        started=started,
                    )
                continue
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                action_errors.append(error)
                if attempt >= self._max_repairs:
                    self._raise_failure(
                        cause=exc,
                        snapshot=snapshot,
                        state=state,
                        event_values=event_values,
                        changes=changes,
                        trigger=trigger,
                        plan_update=plan_update,
                        tactical_analysis=tactical_analysis,
                        policy_input=policy_input,
                        trace=trace,
                        action_errors=action_errors,
                        action_attempts=attempt + 1,
                        action_calls=action_calls,
                        planner_model_calls=planner_model_calls,
                        planner_usages=planner_usages,
                        planner_errors=planner_errors,
                        planner_elapsed=planner_elapsed,
                        enrichment_errors=enrichment_errors,
                        tool_executions=tool_executions,
                        started=started,
                    )
                request = _repair_action_request(
                    policy_input,
                    invalid_response=response.content,
                    error=error,
                    tool=action_tool,
                )
                continue

            prediction = OpponentPrediction(
                kind=str(decision.opponent_prediction["kind"]),
                detail=str(decision.opponent_prediction["detail"]),
                confidence=float(decision.opponent_prediction["confidence"]),
                decision_turn=snapshot.turn,
            )
            state.memory.set_prediction(prediction)
            state.pending_replan = decision.request_replan
            return PolicyResult(
                decision=decision,
                attempts=attempt + 1,
                raw_responses=tuple(trace.raw_responses),
                model_ids=tuple(trace.model_ids),
                response_ids=tuple(trace.response_ids),
                tool_call_ids=tuple(trace.tool_call_ids),
                usages=tuple(trace.usages),
                errors=tuple(action_errors),
                elapsed_seconds=round(time.monotonic() - started, 6),
                policy_input_format=policy_input.format_name,
                policy_input_hash=policy_input.fingerprint(),
                policy_input_characters=policy_input.characters,
                policy_input=policy_input.payload,
                model_calls=planner_model_calls + action_calls,
                expected_model_calls=2 if trigger else 1,
                tool_names=tuple(trace.tool_names),
                tool_executions=tool_executions,
                tactical_analysis=tactical_analysis,
                new_events=event_values,
                memory=state.memory.model_view(),
                belief_state=current_belief.to_dict(),
                belief_changes=changes,
                battle_plan=state.plan.to_dict(),
                plan_update=plan_update,
                plan_trigger=trigger,
                planner_model_calls=planner_model_calls,
                planner_usages=tuple(planner_usages),
                planner_errors=tuple(planner_errors),
                planner_elapsed_seconds=planner_elapsed,
                enrichment_errors=tuple(enrichment_errors),
            )
        raise AssertionError("unreachable")

    def forget_battle(self, battle_id: str) -> None:
        self._states.pop(battle_id, None)

    def state_for(self, battle_id: str) -> ControlledBattleState | None:
        return self._states.get(battle_id)

    @staticmethod
    def _raise_failure(
        *,
        cause: Exception,
        snapshot: BattleSnapshot,
        state: ControlledBattleState,
        event_values: tuple[dict[str, Any], ...],
        changes: tuple[dict[str, Any], ...],
        trigger: str,
        plan_update: dict[str, Any],
        tactical_analysis: dict[str, Any],
        policy_input: CompiledPolicyInput,
        trace: _ControlledTrace,
        action_errors: list[str],
        action_attempts: int,
        action_calls: int,
        planner_model_calls: int,
        planner_usages: list[TokenUsage],
        planner_errors: list[str],
        planner_elapsed: float,
        enrichment_errors: list[str],
        tool_executions: tuple[dict[str, Any], ...],
        started: float,
    ) -> None:
        raise PolicyFailure(
            "Controlled Agent did not produce a valid decision",
            raw_responses=tuple(trace.raw_responses),
            model_ids=tuple(trace.model_ids),
            response_ids=tuple(trace.response_ids),
            tool_call_ids=tuple(trace.tool_call_ids),
            usages=tuple(trace.usages),
            errors=tuple(action_errors),
            attempts=action_attempts,
            elapsed_seconds=round(time.monotonic() - started, 6),
            policy_input=policy_input,
            model_calls=planner_model_calls + action_calls,
            expected_model_calls=2 if trigger else 1,
            tool_names=tuple(trace.tool_names),
            tool_executions=tool_executions,
            tactical_analysis=tactical_analysis,
            new_events=event_values,
            memory=state.memory.model_view(),
            belief_state=(
                state.belief.to_dict() if state.belief is not None else {}
            ),
            belief_changes=changes,
            battle_plan=state.plan.to_dict() if state.plan is not None else {},
            plan_update=plan_update,
            plan_trigger=trigger,
            planner_model_calls=planner_model_calls,
            planner_usages=tuple(planner_usages),
            planner_errors=tuple(planner_errors),
            planner_elapsed_seconds=planner_elapsed,
            enrichment_errors=tuple(enrichment_errors),
        ) from cause


@dataclass
class _ControlledTrace:
    raw_responses: list[str] = field(default_factory=list)
    model_ids: list[str] = field(default_factory=list)
    response_ids: list[str] = field(default_factory=list)
    tool_call_ids: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    usages: list[TokenUsage] = field(default_factory=list)

    def add_response(self, response: ModelResponse, tool_name: str) -> None:
        self.raw_responses.append(response.content)
        self.model_ids.append(response.model_id)
        if response.response_id is not None:
            self.response_ids.append(response.response_id)
        if response.tool_call_id is not None:
            self.tool_call_ids.append(response.tool_call_id)
            self.tool_names.append(tool_name)
        if response.usage is not None:
            self.usages.append(response.usage)

    def add_planner_result(self, result: Any) -> None:
        self.raw_responses.extend(result.raw_responses)
        self.model_ids.extend(result.model_ids)
        self.response_ids.extend(result.response_ids)
        self.tool_call_ids.extend(result.tool_call_ids)
        self.tool_names.extend(PLAN_TOOL_NAME for _ in result.tool_call_ids)
        self.usages.extend(result.usages)

    def add_planner_failure(self, failure: PlannerFailure) -> None:
        self.raw_responses.extend(failure.raw_responses)
        self.model_ids.extend(failure.model_ids)
        self.response_ids.extend(failure.response_ids)
        self.tool_call_ids.extend(failure.tool_call_ids)
        self.tool_names.extend(PLAN_TOOL_NAME for _ in failure.tool_call_ids)
        self.usages.extend(failure.usages)


def _plan_trigger(
    state: ControlledBattleState,
    *,
    new_events: tuple[dict[str, Any], ...],
    changes: tuple[dict[str, Any], ...],
) -> str:
    if state.plan is None:
        return "initial"
    if state.pending_replan:
        return "policy_requested"
    plan = state.plan
    configured = set(plan.replan_triggers)
    for event in new_events:
        if event.get("kind") != "faint":
            continue
        actor = str(event.get("actor") or "")
        side, _, species = actor.partition(":")
        if (
            side == "own"
            and species in plan.preserve
            and "preserve_fainted" in configured
        ):
            return "preserve_fainted"
        if (
            side == "opponent"
            and species in plan.priority_targets
            and "target_fainted" in configured
        ):
            return "target_fainted"
    for event in new_events:
        if event.get("kind") != "tera":
            continue
        actor = str(event.get("actor") or "")
        if actor.startswith("own:") and "own_tera" in configured:
            return "own_tera"
        if (
            actor.startswith("opponent:")
            and "opponent_tera" in configured
        ):
            return "opponent_tera"
    event_kinds = {str(event.get("kind") or "") for event in new_events}
    if (
        "belief_changed" in configured
        and event_kinds.intersection({"item_revealed", "ability_revealed"})
        and any(
            change.get("after") == "likely" for change in changes
        )
    ):
        return "belief_changed"
    return ""


def _compile_controlled_input(
    *,
    battle: dict[str, Any],
    battle_input_format: str,
    memory: dict[str, Any],
    beliefs: dict[str, Any],
    plan: dict[str, Any],
    tactical: dict[str, Any],
) -> CompiledPolicyInput:
    payload = {
        "schema": CONTROLLED_INPUT_SCHEMA,
        "battle_input_format": battle_input_format,
        "battle": battle,
        "memory": memory,
        "beliefs": beliefs,
        "plan": plan,
        "tactical": tactical,
    }
    compiled = CompiledPolicyInput(CONTROLLED_INPUT_SCHEMA, payload)
    recent = memory.get("recent_events", [])
    hypotheses = beliefs.get("hypotheses", [])
    while compiled.characters > MAX_CONTEXT_CHARACTERS and recent:
        recent.pop(0)
        compiled = CompiledPolicyInput(CONTROLLED_INPUT_SCHEMA, payload)
    while compiled.characters > MAX_CONTEXT_CHARACTERS and hypotheses:
        for index in range(len(hypotheses) - 1, -1, -1):
            if hypotheses[index].get("confidence") == "possible":
                hypotheses.pop(index)
                break
        else:
            hypotheses.pop()
        compiled = CompiledPolicyInput(CONTROLLED_INPUT_SCHEMA, payload)
    if compiled.characters > MAX_CONTEXT_CHARACTERS:
        raise ValueError(
            f"controlled Agent context exceeds {MAX_CONTEXT_CHARACTERS} characters"
        )
    return compiled


def _component_error(component: str, error: Exception) -> str:
    return f"{component}:{type(error).__name__}: {error}"


def _unavailable_tactical_analysis(
    catalog: ActionCatalog,
    *,
    error_type: str,
) -> dict[str, Any]:
    return {
        "schema": "tactical-analysis-unavailable-v1",
        "actions": [
            {
                "action_id": action.action_id,
                "kind": action.kind,
                "note": "Tactical estimate unavailable for this decision.",
            }
            for action in catalog.actions
        ],
        "limitations": (
            "Automatic tactical analysis failed; the legal action catalog "
            "remains authoritative."
        ),
        "error_type": error_type,
    }


def _controlled_action_tool(catalog: ActionCatalog) -> ModelTool:
    return ModelTool(
        name=ACTION_TOOL_NAME,
        description=(
            "Select one legal battle action, predict the opponent's next public "
            "action class, and give a brief public explanation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action_id": {
                    "type": "string",
                    "enum": [action.action_id for action in catalog.actions],
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "opponent_prediction": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": list(PREDICTION_KINDS),
                        },
                        "detail": {"type": "string", "maxLength": 80},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["kind", "detail", "confidence"],
                    "additionalProperties": False,
                },
                "request_replan": {"type": "boolean"},
                "reason_codes": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(REASON_CODES)},
                    "minItems": 1,
                    "maxItems": 3,
                },
                "short_rationale": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_RATIONALE_CHARACTERS,
                },
            },
            "required": [
                "action_id",
                "confidence",
                "opponent_prediction",
                "request_replan",
                "reason_codes",
                "short_rationale",
            ],
            "additionalProperties": False,
        },
    )


def _parse_controlled_decision(
    content: str,
    catalog: ActionCatalog,
) -> PolicyDecision:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise TypeError("response must be a JSON object")
    required = {
        "action_id",
        "confidence",
        "opponent_prediction",
        "request_replan",
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
    if not isinstance(action_id, str) or not catalog.contains(action_id):
        raise ValueError(f"action_id {action_id!r} is not currently legal")
    confidence = _confidence(value["confidence"], "confidence")
    prediction = value["opponent_prediction"]
    if not isinstance(prediction, dict):
        raise TypeError("opponent_prediction must be an object")
    if set(prediction) != {"kind", "detail", "confidence"}:
        raise TypeError("opponent_prediction fields are invalid")
    kind = prediction["kind"]
    if kind not in PREDICTION_KINDS:
        raise ValueError(f"unknown opponent prediction kind: {kind}")
    detail = prediction["detail"]
    if not isinstance(detail, str):
        raise TypeError("prediction detail must be a string")
    prediction_confidence = _confidence(
        prediction["confidence"],
        "prediction confidence",
    )
    request_replan = value["request_replan"]
    if not isinstance(request_replan, bool):
        raise TypeError("request_replan must be a boolean")
    reason_codes = value["reason_codes"]
    if not isinstance(reason_codes, list) or not all(
        isinstance(code, str) for code in reason_codes
    ):
        raise TypeError("reason_codes must be a list of strings")
    normalized = tuple(
        dict.fromkeys(REASON_CODE_ALIASES.get(code, code) for code in reason_codes)
    )
    if not 1 <= len(normalized) <= 3:
        raise ValueError("reason_codes must contain between 1 and 3 values")
    invalid = sorted(set(normalized).difference(REASON_CODES))
    if invalid:
        raise ValueError(f"unknown reason_codes: {', '.join(invalid)}")
    rationale = value["short_rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("short_rationale must be a non-empty string")
    return PolicyDecision(
        action_id=action_id,
        confidence=confidence,
        reason_codes=normalized,
        short_rationale=rationale.strip()[:MAX_RATIONALE_CHARACTERS].rstrip(),
        opponent_prediction={
            "kind": kind,
            "detail": detail.strip()[:80],
            "confidence": prediction_confidence,
        },
        request_replan=request_replan,
    )


def _repair_action_request(
    policy_input: CompiledPolicyInput,
    *,
    invalid_response: str,
    error: str,
    tool: ModelTool,
) -> ModelRequest:
    battle = policy_input.payload["battle"]
    return ModelRequest(
        system_prompt=CONTROLLED_SYSTEM_PROMPT,
        user_prompt=json.dumps(
            {
                "error": error,
                "invalid_response": invalid_response,
                "decision_context": policy_input.payload,
                "valid_action_ids": [
                    action["action_id"]
                    for action in battle.get("legal_actions", [])
                ],
                "instruction": (
                    f"Call {ACTION_TOOL_NAME} once with corrected arguments."
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        tool=tool,
    )


def _confidence(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    number = float(value)
    if not 0 <= number <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return number
