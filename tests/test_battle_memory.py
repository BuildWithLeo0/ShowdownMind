from __future__ import annotations

import json
from types import SimpleNamespace

from showdown_mind.battle_memory import (
    BattleEvent,
    BattleMemory,
    OpponentPrediction,
    PokeEnvEventAdapter,
)


def fake_battle(messages: list[list[str]]) -> SimpleNamespace:
    return SimpleNamespace(
        battle_tag="battle-gen9randombattle-1",
        player_role="p1",
        _replay_data=messages,
    )


def test_adapter_consumes_public_events_once_and_memory_reduces_them() -> None:
    battle = fake_battle(
        [
            ["", "turn", "1"],
            ["", "move", "p2a: Gholdengo", "Make It Rain", "p1a: Dragapult"],
            ["", "-damage", "p1a: Dragapult", "53/100"],
            ["", "-item", "p2a: Gholdengo", "Choice Scarf"],
        ]
    )
    memory = BattleMemory(battle.battle_tag)
    adapter = PokeEnvEventAdapter()

    first = adapter.consume(battle, memory)
    memory.consume(first)
    second = adapter.consume(battle, memory)

    assert second == ()
    assert memory.cursor == 4
    assert memory.revealed_moves["opponent:gholdengo"] == {"makeitrain"}
    assert memory.item_history["opponent:gholdengo"] == ["choicescarf"]
    assert memory.damage_evidence[0]["source_move_event_id"] == first[1].event_id


def test_prediction_is_resolved_by_next_opponent_action_kind() -> None:
    battle = fake_battle(
        [
            ["", "turn", "4"],
            ["", "move", "p2a: Corviknight", "Roost", "p2a: Corviknight"],
        ]
    )
    memory = BattleMemory(battle.battle_tag)
    memory.set_prediction(
        OpponentPrediction(
            kind="recovery",
            detail="roost",
            confidence=0.7,
            decision_turn=3,
        )
    )

    events = PokeEnvEventAdapter().consume(battle, memory)
    memory.consume(events)

    resolution = memory.latest_prediction_resolution
    assert resolution is not None
    assert resolution.actual_kind == "recovery"
    assert resolution.matched is True


def test_model_memory_preserves_a_false_prediction_result() -> None:
    memory = BattleMemory("battle-test")
    memory.set_prediction(
        OpponentPrediction(
            kind="switch",
            detail="expected a switch",
            confidence=0.5,
            decision_turn=1,
        )
    )
    memory.consume(
        (
            BattleEvent(
                event_id="battle-test:1",
                battle_id="battle-test",
                sequence=1,
                turn=1,
                kind="move_used",
                actor="opponent:gyarados",
                target="own:pikachu",
                payload={"move_id": "waterfall"},
            ),
        )
    )

    resolution = memory.model_view()["previous_prediction_resolution"]

    assert resolution["matched"] is False


def test_prediction_distinguishes_self_setup_from_opponent_debuff() -> None:
    memory = BattleMemory("battle-test")
    memory.set_prediction(
        OpponentPrediction(
            kind="status",
            detail="growl",
            confidence=0.5,
            decision_turn=1,
        )
    )
    memory.consume(
        (
            BattleEvent(
                event_id="battle-test:1",
                battle_id="battle-test",
                sequence=1,
                turn=1,
                kind="move_used",
                actor="opponent:eevee",
                target="own:pikachu",
                payload={"move_id": "growl"},
            ),
        )
    )

    assert memory.latest_prediction_resolution is not None
    assert memory.latest_prediction_resolution.actual_kind == "status"
    assert memory.latest_prediction_resolution.matched is True


def test_model_memory_excludes_unknown_protocol_noise() -> None:
    battle = fake_battle(
        [
            ["", "player", "p1", "ResearchPlayer"],
            ["", "turn", "1"],
            ["", "switch", "p2a: Pikachu", "Pikachu, L80", "100/100"],
        ]
    )
    memory = BattleMemory(battle.battle_tag)
    events = PokeEnvEventAdapter().consume(battle, memory)
    memory.consume(events)

    recent = memory.model_view()["recent_events"]
    assert all(event["kind"] != "unknown_public_event" for event in recent)
    assert any(event["kind"] == "switch" for event in recent)


def test_decision_memory_view_is_narrower_than_planner_view() -> None:
    battle = fake_battle(
        [
            ["", "turn", "1"],
            ["", "switch", "p2a: Pikachu", "Pikachu, L80", "100/100"],
            ["", "move", "p2a: Pikachu", "Thunderbolt", "p1a: Eevee"],
            ["", "turn", "2"],
        ]
    )
    memory = BattleMemory(battle.battle_tag)
    events = PokeEnvEventAdapter().consume(battle, memory)
    memory.consume(events)

    decision = memory.decision_view()
    planner = memory.model_view()

    assert decision["schema"] == "battle-memory-decision-v1"
    assert "opponent_behavior_counts" in decision
    assert "opponent_behavior" not in decision
    assert len(json.dumps(decision)) <= len(json.dumps(planner))
