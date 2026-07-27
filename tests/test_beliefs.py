from __future__ import annotations

import json

from showdown_mind.battle_memory import BattleEvent, BattleMemory
from showdown_mind.beliefs import (
    BeliefHypothesis,
    BeliefState,
    RandomBattlePriorLoader,
    RuleBeliefTracker,
)
from showdown_mind.domain import BattleSnapshot


def snapshot() -> BattleSnapshot:
    return BattleSnapshot.from_dict(
        {
            "schema_version": "1.0",
            "battle_id": "battle-test",
            "request_id": 1,
            "turn": 3,
            "battle_format": "gen9randombattle",
            "own_side": {"active": "dragapult", "team": []},
            "opponent_side": {
                "active": "Gholdengo",
                "revealed_team": [
                    {
                        "species": "Gholdengo",
                        "moves": ["Shadow Ball"],
                        "ability": "Good as Gold",
                        "tera_type": None,
                    }
                ],
            },
            "field": {},
            "resources": {},
            "legal_actions": [],
        }
    )


def test_rule_belief_filters_incompatible_public_sets(tmp_path) -> None:
    priors = {
        "gholdengo": {
            "sets": [
                {
                    "role": "Fast Attacker",
                    "movepool": ["Shadow Ball", "Make It Rain"],
                    "abilities": ["Good as Gold"],
                    "teraTypes": ["Steel"],
                },
                {
                    "role": "Setup",
                    "movepool": ["Hex", "Nasty Plot"],
                    "abilities": ["Good as Gold"],
                    "teraTypes": ["Flying"],
                },
            ]
        }
    }
    path = tmp_path / "sets.json"
    path.write_text(json.dumps(priors), encoding="utf-8")
    memory = BattleMemory("battle-test")
    memory.consume(
        (
            BattleEvent(
                event_id="battle-test:1",
                battle_id="battle-test",
                sequence=1,
                turn=2,
                kind="move_used",
                actor="opponent:gholdengo",
                target="own:dragapult",
                payload={"move_id": "shadowball"},
            ),
        )
    )

    belief = RuleBeliefTracker(RandomBattlePriorLoader(path)).update(
        snapshot(),
        memory,
    )

    values = {
        (item.kind, item.value, item.confidence)
        for item in belief.hypotheses
    }
    assert ("possible_move", "makeitrain", "likely") in values
    assert ("possible_move", "hex", "possible") not in values
    assert all(item.evidence_ids for item in belief.hypotheses)


def test_missing_prior_is_explicit(tmp_path) -> None:
    path = tmp_path / "sets.json"
    path.write_text("{}", encoding="utf-8")
    belief = RuleBeliefTracker(RandomBattlePriorLoader(path)).update(
        snapshot(),
        BattleMemory("battle-test"),
    )

    assert belief.unavailable_priors == ("gholdengo",)
    assert belief.hypotheses == ()


def test_incompatible_prior_is_not_reintroduced_as_a_guess(tmp_path) -> None:
    priors = {
        "gholdengo": {
            "sets": [
                {
                    "role": "Setup",
                    "movepool": ["Hex", "Nasty Plot"],
                    "abilities": ["Good as Gold"],
                    "teraTypes": ["Flying"],
                }
            ]
        }
    }
    path = tmp_path / "sets.json"
    path.write_text(json.dumps(priors), encoding="utf-8")

    belief = RuleBeliefTracker(RandomBattlePriorLoader(path)).update(
        snapshot(),
        BattleMemory("battle-test"),
    )

    assert belief.hypotheses == ()
    assert belief.unavailable_priors == (
        "gholdengo:no_compatible_public_prior",
    )


def test_model_view_prioritizes_active_subject_and_caps_repeated_evidence() -> None:
    hypotheses = tuple(
        BeliefHypothesis(
            subject=subject,
            kind="possible_move",
            value=f"move-{index}",
            confidence="possible",
            evidence_ids=("prior", "event-1", "event-2", "event-3"),
        )
        for subject in ("opponent:bench", "opponent:active")
        for index in range(4)
    )
    belief = BeliefState(
        battle_id="battle-test",
        updated_turn=3,
        hypotheses=hypotheses,
    )

    view = belief.model_view(
        active_subject="opponent:active",
        max_hypotheses=4,
    )

    assert len(view["hypotheses"]) == 4
    assert view["hypotheses"][0]["subject"] == "opponent:active"
    assert view["hypotheses"][0]["evidence_ids"] == [
        "prior",
        "event-2",
        "event-3",
    ]
