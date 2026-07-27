from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from poke_env.battle import Move, Pokemon
from poke_env.player.battle_order import SingleBattleOrder

from showdown_mind.actions import ActionCatalog, CatalogEntry
from showdown_mind.battle_memory import BattleMemory
from showdown_mind.controlled_policy import (
    ControlledAgentPolicy,
    ControlledBattleState,
    _plan_trigger,
)
from showdown_mind.domain import BattleSnapshot, LegalAction
from showdown_mind.models import ScriptedModelClient
from showdown_mind.planning import BattlePlan


def snapshot(*, turn: int = 1, request_id: int = 1) -> BattleSnapshot:
    return BattleSnapshot.from_dict(
        {
            "schema_version": "1.0",
            "battle_id": "battle-controlled",
            "request_id": request_id,
            "turn": turn,
            "battle_format": "gen9randombattle",
            "own_side": {
                "active": "Pikachu",
                "team": [
                    {
                        "species": "Pikachu",
                        "hp_fraction": 1.0,
                        "moves": ["Thunderbolt"],
                    }
                ],
                "side_conditions": {},
            },
            "opponent_side": {
                "active": "Gyarados",
                "revealed_team": [
                    {
                        "species": "Gyarados",
                        "hp_fraction": 1.0,
                        "moves": [],
                    }
                ],
                "side_conditions": {},
                "used_tera": False,
            },
            "field": {},
            "resources": {"can_tera": False},
            "legal_actions": [
                {
                    "action_id": "move:thunderbolt",
                    "kind": "move",
                    "label": "Use Thunderbolt",
                    "details": {"base_power": 90},
                }
            ],
        }
    )


def catalog() -> ActionCatalog:
    action = LegalAction(
        action_id="move:thunderbolt",
        kind="move",
        label="Use Thunderbolt",
        details={"base_power": 90},
    )
    return ActionCatalog(
        (
            CatalogEntry(
                action,
                SingleBattleOrder(Move("thunderbolt", gen=9)),
            ),
        )
    )


def battle(messages: list[list[str]]) -> SimpleNamespace:
    own = Pokemon(gen=9, species="Pikachu")
    opponent = Pokemon(gen=9, species="Gyarados")
    return SimpleNamespace(
        battle_tag="battle-controlled",
        player_role="p1",
        _replay_data=messages,
        active_pokemon=own,
        opponent_active_pokemon=opponent,
        side_conditions={},
        opponent_side_conditions={},
        fields={},
        weather={},
        gen=9,
    )


def plan_json() -> str:
    return json.dumps(
        {
            "win_condition": "Use Pikachu to pressure Gyarados.",
            "preserve": ["pikachu"],
            "priority_targets": ["gyarados"],
            "tera_policy": "Hold Tera.",
            "risk_posture": "balanced",
            "replan_triggers": ["preserve_fainted", "target_fainted"],
        }
    )


def action_json(
    *,
    prediction_kind: str = "attack",
    request_replan: bool = False,
    reason_codes: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "action_id": "move:thunderbolt",
            "confidence": 0.8,
            "opponent_prediction": {
                "kind": prediction_kind,
                "detail": "waterfall",
                "confidence": 0.6,
            },
            "request_replan": request_replan,
            "reason_codes": reason_codes or ["DAMAGE", "PLAN_ALIGNMENT"],
            "short_rationale": "Pressure the visible target with strong damage.",
        }
    )


class FailingTacticalAdvisor:
    def analyze(self, *args, **kwargs):
        raise RuntimeError("calculator unavailable")


@pytest.mark.asyncio
async def test_first_controlled_turn_plans_then_selects_one_action() -> None:
    client = ScriptedModelClient([plan_json(), action_json()])
    policy = ControlledAgentPolicy(client)

    result = await policy.decide(
        snapshot(),
        catalog(),
        battle=battle(
            [
                ["", "turn", "1"],
                ["", "switch", "p2a: Gyarados", "Gyarados, L80", "100/100"],
            ]
        ),
    )

    assert [request.tool.name for request in client.requests] == [
        "update_battle_plan",
        "choose_battle_action",
    ]
    assert result.model_calls == 2
    assert result.expected_model_calls == 2
    assert result.plan_trigger == "initial"
    assert result.battle_plan["preserve"] == ("pikachu",)
    assert result.decision.opponent_prediction["kind"] == "attack"
    assert result.tool_executions[0]["execution_kind"] == "internal"
    assert result.memory["recent_events"]
    assert result.policy_input["schema"] == "controlled-agent-v2"
    assert result.policy_input["memory"]["schema"] == (
        "battle-memory-decision-v1"
    )
    assert result.policy_input["tactical"]["view"] == "action-decision-v2"


@pytest.mark.asyncio
async def test_normal_followup_turn_uses_one_action_call_and_resolves_prediction() -> None:
    client = ScriptedModelClient(
        [plan_json(), action_json(), action_json(prediction_kind="switch")]
    )
    policy = ControlledAgentPolicy(client)
    live_battle = battle([["", "turn", "1"]])
    await policy.decide(snapshot(), catalog(), battle=live_battle)
    live_battle._replay_data.extend(
        [
            ["", "move", "p2a: Gyarados", "Waterfall", "p1a: Pikachu"],
            ["", "turn", "2"],
        ]
    )

    result = await policy.decide(
        snapshot(turn=2, request_id=2),
        catalog(),
        battle=live_battle,
    )

    assert result.model_calls == 1
    assert result.expected_model_calls == 1
    assert result.plan_trigger == ""
    assert result.memory["previous_prediction_resolution"]["matched"] is True
    assert len(client.requests) == 3


@pytest.mark.asyncio
async def test_planner_failure_uses_neutral_plan_but_action_continues() -> None:
    invalid_plan = json.dumps(
        {
            "win_condition": "Invalid hidden target.",
            "preserve": ["missingno"],
            "priority_targets": [],
            "tera_policy": "Hold.",
            "risk_posture": "balanced",
            "replan_triggers": ["plan_invalid"],
        }
    )
    client = ScriptedModelClient(
        [invalid_plan, invalid_plan, action_json()]
    )
    result = await ControlledAgentPolicy(client).decide(
        snapshot(),
        catalog(),
        battle=battle([["", "turn", "1"]]),
    )

    assert result.battle_plan["version"] == 0
    assert result.planner_model_calls == 2
    assert result.planner_errors
    assert result.planner_failed is True
    assert result.decision.action_id == "move:thunderbolt"
    assert result.model_calls == 3


@pytest.mark.asyncio
async def test_repaired_planner_error_is_not_a_final_failure() -> None:
    invalid_plan = json.dumps(
        {
            "win_condition": "Invalid.",
            "preserve": ["missingno"],
            "priority_targets": [],
            "tera_policy": "Hold.",
            "risk_posture": "balanced",
            "replan_triggers": [],
        }
    )
    client = ScriptedModelClient(
        [invalid_plan, plan_json(), action_json()]
    )

    result = await ControlledAgentPolicy(client).decide(
        snapshot(),
        catalog(),
        battle=battle([["", "turn", "1"]]),
    )

    assert result.planner_errors
    assert result.planner_failed is False
    assert result.battle_plan["version"] == 1


@pytest.mark.asyncio
async def test_enrichment_failure_is_recorded_but_does_not_block_action() -> None:
    client = ScriptedModelClient([plan_json(), action_json()])
    policy = ControlledAgentPolicy(
        client,
        tactical_advisor=FailingTacticalAdvisor(),
    )

    result = await policy.decide(
        snapshot(),
        catalog(),
        battle=battle([["", "turn", "1"]]),
    )

    assert result.decision.action_id == "move:thunderbolt"
    assert result.tactical_analysis["schema"] == (
        "tactical-analysis-unavailable-v1"
    )
    assert result.enrichment_errors == (
        "tactical_advisor:RuntimeError: calculator unavailable",
    )


@pytest.mark.asyncio
async def test_known_reason_code_overflow_is_normalized_without_retry() -> None:
    client = ScriptedModelClient(
        [
            plan_json(),
            action_json(
                reason_codes=[
                    "DAMAGE",
                    "STAB",
                    "TYPE_MATCHUP",
                    "SURVIVAL",
                ]
            ),
        ]
    )

    result = await ControlledAgentPolicy(client).decide(
        snapshot(),
        catalog(),
        battle=battle([["", "turn", "1"]]),
    )

    assert result.attempts == 1
    assert result.decision.reason_codes == (
        "DAMAGE",
        "STAB",
        "TYPE_MATCHUP",
    )
    assert result.decision_normalizations == (
        "reason_codes_truncated:4->3",
    )
    assert result.errors == ()


def test_plan_reacts_only_to_faints_that_invalidate_it() -> None:
    state = ControlledBattleState(
        memory=BattleMemory("battle-controlled"),
        plan=BattlePlan(
            version=1,
            created_turn=1,
            win_condition="Preserve Pikachu for Gyarados.",
            preserve=("pikachu",),
            priority_targets=("gyarados",),
            tera_policy="Hold.",
            risk_posture="balanced",
            replan_triggers=("preserve_fainted", "target_fainted"),
        ),
    )

    unrelated = _plan_trigger(
        state,
        new_events=({"kind": "faint", "actor": "own:eevee"},),
        changes=(),
    )
    relevant = _plan_trigger(
        state,
        new_events=({"kind": "faint", "actor": "opponent:gyarados"},),
        changes=(),
    )

    assert unrelated == ""
    assert relevant == "target_fainted"
