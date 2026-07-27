from __future__ import annotations

import json

import pytest

from showdown_mind.domain import BattleSnapshot
from showdown_mind.models import ScriptedModelClient
from showdown_mind.planning import (
    BattlePlanner,
    PlannerFailure,
    neutral_plan,
)


def snapshot() -> BattleSnapshot:
    return BattleSnapshot.from_dict(
        {
            "schema_version": "1.0",
            "battle_id": "battle-test",
            "request_id": 1,
            "turn": 4,
            "battle_format": "gen9randombattle",
            "own_side": {
                "active": "Dragapult",
                "team": [{"species": "Dragapult"}, {"species": "Great Tusk"}],
            },
            "opponent_side": {
                "active": "Kingambit",
                "revealed_team": [{"species": "Kingambit"}],
            },
            "field": {},
            "resources": {},
            "legal_actions": [],
        }
    )


def valid_plan(**overrides: object) -> str:
    value: dict[str, object] = {
        "win_condition": "Preserve Dragapult for a late clean.",
        "preserve": ["dragapult"],
        "priority_targets": ["kingambit"],
        "tera_policy": "Hold Tera for the win condition.",
        "risk_posture": "balanced",
        "replan_triggers": ["preserve_fainted", "target_fainted"],
    }
    value.update(overrides)
    return json.dumps(value)


@pytest.mark.asyncio
async def test_planner_returns_validated_visible_plan() -> None:
    client = ScriptedModelClient([valid_plan()])
    result = await BattlePlanner(client).update(
        snapshot=snapshot(),
        context={"battle": snapshot().to_dict()},
        previous=None,
    )

    assert result.plan.version == 1
    assert result.plan.preserve == ("dragapult",)
    assert result.plan.priority_targets == ("kingambit",)
    assert client.requests[0].tool.name == "update_battle_plan"


@pytest.mark.asyncio
async def test_planner_cannot_disable_required_plan_invalidation() -> None:
    client = ScriptedModelClient(
        [
            valid_plan(
                replan_triggers=["opponent_tera"],
            )
        ]
    )

    result = await BattlePlanner(client).update(
        snapshot=snapshot(),
        context={"battle": snapshot().to_dict()},
        previous=None,
    )

    assert result.plan.replan_triggers == (
        "opponent_tera",
        "preserve_fainted",
        "target_fainted",
    )


@pytest.mark.asyncio
async def test_planner_repairs_non_visible_species_once() -> None:
    client = ScriptedModelClient(
        [
            valid_plan(preserve=["missingno"]),
            valid_plan(),
        ]
    )
    result = await BattlePlanner(client).update(
        snapshot=snapshot(),
        context={"battle": snapshot().to_dict()},
        previous=None,
    )

    assert result.model_calls == 2
    assert "non-visible species" in result.errors[0]


@pytest.mark.asyncio
async def test_planner_failure_is_explicit_after_repair() -> None:
    client = ScriptedModelClient(
        [
            valid_plan(priority_targets=["missingno"]),
            valid_plan(priority_targets=["missingno"]),
        ]
    )
    with pytest.raises(PlannerFailure):
        await BattlePlanner(client).update(
            snapshot=snapshot(),
            context={"battle": snapshot().to_dict()},
            previous=None,
        )


def test_neutral_plan_requires_no_hidden_species() -> None:
    plan = neutral_plan(7)
    assert plan.version == 0
    assert plan.preserve == ()
    assert plan.priority_targets == ()
    assert plan.risk_posture == "balanced"
