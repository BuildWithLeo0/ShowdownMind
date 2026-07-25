import json

import pytest
from poke_env.battle import Move
from poke_env.player.battle_order import SingleBattleOrder

from showdown_mind.actions import ActionCatalog, CatalogEntry
from showdown_mind.domain import BattleSnapshot, LegalAction
from showdown_mind.models import ACTION_TOOL_NAME, TACTICAL_TOOL_NAME, ScriptedModelClient
from showdown_mind.policy import (
    PolicyFailure,
    SingleCallPolicy,
    TacticalToolPolicy,
)
from showdown_mind.tactics import (
    TACTICAL_ANALYSIS_SCHEMA,
    TACTICAL_MODEL_VIEW,
)


def catalog() -> ActionCatalog:
    action = LegalAction(
        "move:tackle",
        "move",
        "Use tackle",
        {"base_power": 40},
    )
    order = SingleBattleOrder(Move("tackle", gen=9))
    return ActionCatalog((CatalogEntry(action, order),))


def snapshot() -> BattleSnapshot:
    return BattleSnapshot(
        schema_version="1.0",
        battle_id="battle-test",
        request_id=1,
        turn=1,
        battle_format="gen9randombattle",
        own_side={},
        opponent_side={},
        field={},
        resources={},
        legal_actions=catalog().actions,
    )


def decision_json(
    *,
    action_id: str = "move:tackle",
    confidence: float | bool = 0.8,
    reason_codes: list[str] | None = None,
    short_rationale: str = "Best available damage.",
) -> str:
    return json.dumps(
        {
            "action_id": action_id,
            "confidence": confidence,
            "reason_codes": reason_codes or ["DAMAGE"],
            "short_rationale": short_rationale,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("input_format", ["full", "pruned", "compact"])
async def test_policy_accepts_one_valid_json_decision(input_format: str) -> None:
    client = ScriptedModelClient(
        [decision_json()],
        model_id="test-model",
    )

    result = await SingleCallPolicy(
        client,
        input_format=input_format,
    ).decide(snapshot(), catalog())

    assert result.decision.action_id == "move:tackle"
    assert result.attempts == 1
    assert result.model_ids == ("test-model",)
    assert result.errors == ()
    assert result.policy_input_format == f"{input_format}-v1"
    request = client.requests[0]
    assert request.tool.name == "choose_battle_action"
    assert request.tool.strict
    assert request.tool.parameters["properties"]["action_id"]["enum"] == ["move:tackle"]
    assert set(request.tool.parameters["required"]) == {
        "action_id",
        "confidence",
        "reason_codes",
        "short_rationale",
    }


@pytest.mark.asyncio
async def test_policy_repairs_once_after_invalid_output() -> None:
    client = ScriptedModelClient(
        [
            "not json",
            decision_json(),
        ]
    )

    result = await SingleCallPolicy(client).decide(snapshot(), catalog())

    assert result.attempts == 2
    assert len(result.errors) == 1
    repair_payload = json.loads(client.requests[1].user_prompt)
    assert repair_payload["valid_action_ids"] == ["move:tackle"]
    assert repair_payload["battle"]["schema"] == "pruned-v1"


@pytest.mark.asyncio
async def test_policy_fails_after_one_repair() -> None:
    client = ScriptedModelClient(["{}", '{"action_id": "illegal"}'])

    with pytest.raises(PolicyFailure) as error:
        await SingleCallPolicy(client).decide(snapshot(), catalog())

    assert len(client.requests) == 2
    assert len(error.value.errors) == 2


@pytest.mark.asyncio
async def test_policy_counts_attempts_when_model_never_returns() -> None:
    client = ScriptedModelClient([])

    with pytest.raises(PolicyFailure) as error:
        await SingleCallPolicy(client).decide(snapshot(), catalog())

    assert error.value.attempts == 2
    assert error.value.raw_responses == ()
    assert error.value.model_ids == ()
    assert client.requests[0] == client.requests[1]


@pytest.mark.asyncio
async def test_policy_rejects_boolean_confidence() -> None:
    client = ScriptedModelClient(
        [
            decision_json(confidence=True),
            decision_json(),
        ]
    )

    result = await SingleCallPolicy(client).decide(snapshot(), catalog())

    assert result.attempts == 2
    assert "confidence must be a number" in result.errors[0]


@pytest.mark.asyncio
async def test_policy_requires_a_short_public_rationale() -> None:
    client = ScriptedModelClient(
        [
            decision_json(short_rationale=" "),
            decision_json(short_rationale="Preserve momentum with reliable damage."),
        ]
    )

    result = await SingleCallPolicy(client).decide(snapshot(), catalog())

    assert result.attempts == 2
    assert "short_rationale must not be empty" in result.errors[0]
    assert result.decision.short_rationale == "Preserve momentum with reliable damage."


@pytest.mark.asyncio
async def test_policy_enforces_rationale_length_limit() -> None:
    client = ScriptedModelClient([decision_json(short_rationale="x" * 241)])

    result = await SingleCallPolicy(client).decide(snapshot(), catalog())

    assert result.attempts == 1
    assert result.errors == ()
    assert result.decision.short_rationale == "x" * 240


@pytest.mark.asyncio
async def test_policy_rejects_unknown_reason_codes() -> None:
    client = ScriptedModelClient(
        [
            decision_json(reason_codes=["MADE_UP"]),
            decision_json(reason_codes=["DAMAGE"]),
        ]
    )

    result = await SingleCallPolicy(client).decide(snapshot(), catalog())

    assert result.attempts == 2
    assert "unknown reason_codes" in result.errors[0]


@pytest.mark.asyncio
async def test_policy_accepts_weather_and_accuracy_reason_codes() -> None:
    client = ScriptedModelClient(
        [decision_json(reason_codes=["WEATHER", "ACCURACY", "STAB"])]
    )

    result = await SingleCallPolicy(client).decide(snapshot(), catalog())

    assert result.attempts == 1
    assert result.decision.reason_codes == ("WEATHER", "ACCURACY", "STAB")


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["KO", "KO_PROBABILITY"])
async def test_policy_normalizes_ko_reason_code_aliases(alias: str) -> None:
    client = ScriptedModelClient([decision_json(reason_codes=[alias])])

    result = await SingleCallPolicy(client).decide(snapshot(), catalog())

    assert result.attempts == 1
    assert result.decision.reason_codes == ("DAMAGE",)


@pytest.mark.asyncio
async def test_tactical_policy_executes_two_native_tools_in_order() -> None:
    client = ScriptedModelClient(["{}", decision_json()])
    battle = type(
        "Battle",
        (),
        {
            "active_pokemon": None,
            "opponent_active_pokemon": None,
        },
    )()

    result = await TacticalToolPolicy(client).decide(
        snapshot(),
        catalog(),
        battle=battle,
    )

    assert result.decision.action_id == "move:tackle"
    assert result.model_calls == 2
    assert result.expected_model_calls == 2
    assert result.attempts == 1
    assert result.tool_names == (TACTICAL_TOOL_NAME, ACTION_TOOL_NAME)
    assert result.tactical_analysis["schema"] == TACTICAL_ANALYSIS_SCHEMA
    assert result.tool_executions[0]["tool_name"] == TACTICAL_TOOL_NAME
    assert client.requests[0].tool.name == TACTICAL_TOOL_NAME
    assert client.requests[1].tool.name == ACTION_TOOL_NAME
    history = client.requests[1].tool_history
    assert len(history) == 1
    assert history[0].tool_name == TACTICAL_TOOL_NAME
    assert history[0].tool_call_id == "scripted-tool-call-1"
    assert TACTICAL_ANALYSIS_SCHEMA in history[0].result
    model_analysis = json.loads(history[0].result)
    assert model_analysis["view"] == TACTICAL_MODEL_VIEW
    assert "limitations" not in model_analysis
    execution = result.tool_executions[0]
    assert execution["model_result_characters"] < (
        execution["audit_result_characters"]
    )


@pytest.mark.asyncio
async def test_tactical_policy_retries_invalid_tool_arguments() -> None:
    client = ScriptedModelClient(
        [
            '{"unexpected":true}',
            "{}",
            decision_json(),
        ]
    )
    battle = type(
        "Battle",
        (),
        {
            "active_pokemon": None,
            "opponent_active_pokemon": None,
        },
    )()

    result = await TacticalToolPolicy(client).decide(
        snapshot(),
        catalog(),
        battle=battle,
    )

    assert result.model_calls == 3
    assert result.expected_model_calls == 2
    assert result.attempts == 2
    assert "accepts no arguments" in result.errors[0]


@pytest.mark.asyncio
async def test_tactical_policy_requires_live_battle_context() -> None:
    with pytest.raises(ValueError, match="current battle"):
        await TacticalToolPolicy(ScriptedModelClient([])).decide(
            snapshot(),
            catalog(),
        )


def test_policy_limits_repairs_to_one() -> None:
    with pytest.raises(ValueError, match="max_repairs"):
        SingleCallPolicy(ScriptedModelClient([]), max_repairs=2)


def test_policy_rejects_unknown_input_format() -> None:
    with pytest.raises(ValueError, match="input format"):
        SingleCallPolicy(ScriptedModelClient([]), input_format="missing")
