from __future__ import annotations

import json

import pytest

from showdown_mind.models import DeterministicModelClient
from showdown_mind.scenario_benchmark import (
    SCENARIO_BANK_SCHEMA,
    build_scenario_bank,
    evaluate_scenario_bank,
    load_scenario_bank,
)


def controlled_record(*, action_id: str = "move:thunderbolt") -> dict:
    legal_actions = [
        {
            "action_id": "move:tackle",
            "kind": "move",
            "base_power": 40,
        },
        {
            "action_id": "move:thunderbolt",
            "kind": "move",
            "base_power": 90,
        },
        {
            "action_id": "switch:eevee",
            "kind": "switch",
            "species": "eevee",
        },
    ]
    return {
        "battle_id": "battle-fixed-1",
        "request_id": 2,
        "turn": 1,
        "action_id": action_id,
        "fallback_used": False,
        "errors": ["TypeError: opponent_prediction must be an object"],
        "confidence": 0.8,
        "plan_trigger": "initial",
        "policy_input": {
            "schema": "controlled-agent-v2",
            "battle_input_format": "compact-v1",
            "battle": {
                "schema": "compact-v1",
                "turn": 1,
                "own": {"active": "pikachu"},
                "opponent": {"active": "gyarados"},
                "legal_actions": legal_actions,
            },
            "memory": {},
            "beliefs": {},
            "plan": {"win_condition": "Pressure Gyarados."},
            "tactical": {},
        },
        "tactical_analysis": {
            "best_damage_action_ids": ["move:thunderbolt"],
            "best_ko_action_ids": ["move:thunderbolt"],
            "safest_action_ids": ["move:thunderbolt"],
            "best_ko_probability": 1.0,
        },
    }


def write_source(tmp_path) -> tuple:
    decision_log = tmp_path / "run.jsonl"
    decision_log.write_text(
        json.dumps(controlled_record()) + "\n",
        encoding="utf-8",
    )
    decision_log.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "agent_wins": 0,
                "opponent_wins": 1,
                "draws": 0,
            }
        ),
        encoding="utf-8",
    )
    return decision_log, tmp_path / "bank.json"


def test_builds_and_validates_fixed_scenario_bank(tmp_path) -> None:
    decision_log, bank_path = write_source(tmp_path)

    summary = build_scenario_bank(
        [decision_log],
        output_path=bank_path,
    )
    bank = load_scenario_bank(bank_path)

    assert bank["schema"] == SCENARIO_BANK_SCHEMA
    assert summary.scenarios == 1
    scenario = bank["scenarios"][0]
    assert scenario["source"]["outcome"] == "loss"
    assert "protocol_regression" in scenario["tags"]
    assert scenario["calculator_recommended_action_ids"] == [
        "move:thunderbolt"
    ]
    assert scenario["acceptable_action_ids"] == []
    assert "opponent_prediction" not in json.dumps(scenario["policy_input"])


def test_scenario_builder_excludes_rejected_evaluation_attempts(tmp_path) -> None:
    rejected = tmp_path / "rejected.jsonl"
    rejected.write_text(
        json.dumps(controlled_record()) + "\n",
        encoding="utf-8",
    )
    rejected.with_suffix(".attempt.json").write_text(
        json.dumps({"status": "rejected"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no decision JSONL"):
        build_scenario_bank(
            [tmp_path],
            output_path=tmp_path / "bank.json",
        )


@pytest.mark.asyncio
async def test_evaluates_frozen_scenario_without_showdown_or_planner(tmp_path) -> None:
    decision_log, bank_path = write_source(tmp_path)
    build_scenario_bank([decision_log], output_path=bank_path)
    bank = load_scenario_bank(bank_path)

    report = await evaluate_scenario_bank(
        DeterministicModelClient(),
        bank,
    )

    assert report["metrics"]["scenarios"] == 1
    assert report["metrics"]["valid_decisions"] == 1
    assert report["metrics"]["model_calls"] == 1
    assert report["metrics"]["calculator_alignment_rate"] == 1.0
    assert report["metrics"]["historical_agreement_rate"] == 1.0
    assert report["metrics"]["curated_alignment_rate"] is None
    assert report["results"][0]["action_id"] == "move:thunderbolt"
