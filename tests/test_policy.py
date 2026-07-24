import json

import pytest
from poke_env.battle import Move
from poke_env.player.battle_order import SingleBattleOrder

from showdown_mind.actions import ActionCatalog, CatalogEntry
from showdown_mind.domain import BattleSnapshot, LegalAction
from showdown_mind.models import ScriptedModelClient
from showdown_mind.policy import PolicyFailure, SingleCallPolicy


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


@pytest.mark.asyncio
async def test_policy_accepts_one_valid_json_decision() -> None:
    client = ScriptedModelClient(
        [
            json.dumps(
                {
                    "action_id": "move:tackle",
                    "confidence": 0.8,
                    "reason_codes": ["DAMAGE"],
                    "short_rationale": "Best available damage.",
                }
            )
        ],
        model_id="test-model",
    )

    result = await SingleCallPolicy(client).decide(snapshot(), catalog())

    assert result.decision.action_id == "move:tackle"
    assert result.attempts == 1
    assert result.model_ids == ("test-model",)
    assert result.errors == ()


@pytest.mark.asyncio
async def test_policy_repairs_once_after_invalid_output() -> None:
    client = ScriptedModelClient(
        [
            "not json",
            '{"action_id": "move:tackle"}',
        ]
    )

    result = await SingleCallPolicy(client).decide(snapshot(), catalog())

    assert result.attempts == 2
    assert len(result.errors) == 1
    repair_payload = json.loads(client.requests[1].user_prompt)
    assert repair_payload["valid_action_ids"] == ["move:tackle"]
    assert repair_payload["battle"]["battle_id"] == "battle-test"


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
            '{"action_id": "move:tackle", "confidence": true}',
            '{"action_id": "move:tackle"}',
        ]
    )

    result = await SingleCallPolicy(client).decide(snapshot(), catalog())

    assert result.attempts == 2
    assert "confidence must be a number" in result.errors[0]


def test_policy_limits_repairs_to_one() -> None:
    with pytest.raises(ValueError, match="max_repairs"):
        SingleCallPolicy(ScriptedModelClient([]), max_repairs=2)
