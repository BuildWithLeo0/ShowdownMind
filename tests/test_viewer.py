import base64
import json
import re

import pytest

from showdown_mind.viewer import ViewerError, build_replay_viewer


def decision(battle_id: str, *, turn: int = 1) -> dict:
    return {
        "battle_id": battle_id,
        "request_id": turn,
        "turn": turn,
        "action_id": "move:thunderbolt",
        "confidence": 0.8,
        "reason_codes": ["DAMAGE"],
        "short_rationale": "Reliable neutral damage.",
        "attempts": 1,
        "fallback_used": False,
        "errors": ["provider mentioned sk-abcdefghijk"],
        "model_ids": ["test-model"],
        "raw_responses": ["raw-response-must-not-be-embedded"],
        "policy_input": {"private-shape": "must-not-be-embedded"},
        "policy_input_format": "pruned-v1",
        "policy_input_hash": "sha256:test",
        "policy_input_characters": 123,
        "elapsed_seconds": 0.5,
        "usages": [
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
        ],
        "snapshot": {
            "turn": turn,
            "request_id": turn,
            "own_side": {
                "active": "pikachu",
                "team": [
                    {
                        "name": "Pikachu",
                        "species": "pikachu",
                        "active": True,
                        "hp_fraction": 1.0,
                    }
                ],
            },
            "opponent_side": {
                "active": "squirtle",
                "revealed_team": [
                    {
                        "name": "Squirtle",
                        "species": "squirtle",
                        "active": True,
                        "hp_fraction": 1.0,
                    }
                ],
            },
            "field": {},
            "resources": {"can_tera": True},
            "legal_actions": [
                {
                    "action_id": "move:thunderbolt",
                    "kind": "move",
                    "label": "Use Thunderbolt",
                    "details": {
                        "base_power": 90,
                        "accuracy": 1.0,
                        "type": "electric",
                    },
                }
            ],
        },
    }


def write_log(path, *records: dict) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def write_replay(path, battle_id: str, protocol: str | None = None) -> None:
    battle_protocol = protocol or "\n".join(
        [
            f">{battle_id}",
            "|init|battle",
            "|player|p1|ResearchPlayer 1|",
            "|player|p2|Opponent 1|",
            "|start",
            "|turn|1",
            "|move|p1a: Pikachu|Thunderbolt|p2a: Squirtle",
            "|win|ResearchPlayer 1",
        ]
    )
    path.write_text(
        (
            f"<!doctype html><title>{battle_id}</title>"
            '<script type="text/plain" class="battle-log-data">'
            f"{battle_protocol}</script>"
            "<script>window.replay=true</script>"
        ),
        encoding="utf-8",
    )


def decoded_payload(html: str) -> dict:
    match = re.search(
        r'<script type="application/octet-stream" id="viewer-data">([^<]+)</script>',
        html,
    )
    assert match
    return json.loads(base64.b64decode(match.group(1)).decode("utf-8"))


def test_builds_single_file_viewer_with_sanitized_decisions(tmp_path) -> None:
    battle_id = "battle-gen9randombattle-42"
    decision_log = tmp_path / "agent.jsonl"
    replay = tmp_path / f"ResearchPlayer 1 - {battle_id}.html"
    write_log(decision_log, decision(battle_id))
    write_replay(replay, battle_id)

    result = build_replay_viewer(
        decision_log,
        replay_path=replay,
    )

    output = decision_log.with_suffix(".viewer.html")
    assert result.output_path == str(output)
    assert result.decisions == 1
    html = output.read_text(encoding="utf-8")
    payload = decoded_payload(html)
    encoded_payload = json.dumps(payload)
    assert payload["battle_id"] == battle_id
    assert payload["decisions"][0]["action_id"] == "move:thunderbolt"
    assert isinstance(payload["decisions"][0]["replay_step"], int)
    assert payload["replay_sync"]["agent_side"] == "p1"
    assert payload["replay_sync"]["anchored_decisions"] == 1
    assert payload["decisions"][0]["errors"] == ["provider mentioned [REDACTED]"]
    assert "raw-response-must-not-be-embedded" not in encoded_payload
    assert "private-shape" not in encoded_payload


def test_discovers_and_prefers_research_player_replay(tmp_path) -> None:
    battle_id = "battle-gen9randombattle-77"
    decision_log = tmp_path / "agent.jsonl"
    other = tmp_path / f"MaxBasePowerPlay 1 - {battle_id}.html"
    research = tmp_path / f"ResearchPlayer 1 - {battle_id}.html"
    write_log(decision_log, decision(battle_id))
    write_replay(other, battle_id)
    write_replay(research, battle_id)

    result = build_replay_viewer(decision_log, replay_dir=tmp_path)

    assert result.replay_path == str(research)


def test_requires_battle_id_for_multi_battle_log(tmp_path) -> None:
    decision_log = tmp_path / "batch.jsonl"
    write_log(
        decision_log,
        decision("battle-gen9randombattle-1"),
        decision("battle-gen9randombattle-2"),
    )

    with pytest.raises(ViewerError, match="multiple battles"):
        build_replay_viewer(decision_log, replay_dir=tmp_path)


def test_selects_only_requested_battle_from_batch(tmp_path) -> None:
    first = "battle-gen9randombattle-1"
    second = "battle-gen9randombattle-2"
    decision_log = tmp_path / "batch.jsonl"
    replay = tmp_path / f"ResearchPlayer 1 - {second}.html"
    write_log(
        decision_log,
        decision(first),
        decision(second, turn=3),
    )
    write_replay(replay, second)

    result = build_replay_viewer(
        decision_log,
        replay_path=replay,
        battle_id=second,
    )

    payload = decoded_payload(
        decision_log.with_suffix(".viewer.html").read_text(encoding="utf-8")
    )
    assert result.battle_id == second
    assert result.decisions == 1
    assert payload["decisions"][0]["turn"] == 3


def test_rejects_replay_for_another_battle(tmp_path) -> None:
    decision_log = tmp_path / "agent.jsonl"
    replay = tmp_path / "wrong.html"
    write_log(decision_log, decision("battle-gen9randombattle-1"))
    write_replay(replay, "battle-gen9randombattle-2")

    with pytest.raises(ViewerError, match="does not match"):
        build_replay_viewer(decision_log, replay_path=replay)


def test_requires_force_to_replace_viewer(tmp_path) -> None:
    battle_id = "battle-gen9randombattle-9"
    decision_log = tmp_path / "agent.jsonl"
    replay = tmp_path / f"ResearchPlayer 1 - {battle_id}.html"
    output = decision_log.with_suffix(".viewer.html")
    write_log(decision_log, decision(battle_id))
    write_replay(replay, battle_id)
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(ViewerError, match="already exists"):
        build_replay_viewer(decision_log, replay_path=replay)

    result = build_replay_viewer(
        decision_log,
        replay_path=replay,
        force=True,
    )
    assert result.output_path == str(output)
    assert "ShowdownMind" in output.read_text(encoding="utf-8")


def test_anchors_second_same_turn_decision_to_forced_switch(tmp_path) -> None:
    battle_id = "battle-gen9randombattle-55"
    decision_log = tmp_path / "agent.jsonl"
    replay = tmp_path / f"ResearchPlayer 1 - {battle_id}.html"
    move_decision = decision(battle_id, turn=2)
    switch_decision = decision(battle_id, turn=2)
    switch_decision["request_id"] = 3
    switch_decision["action_id"] = "switch:slowbro"
    switch_decision["snapshot"]["request_id"] = 3
    switch_decision["snapshot"]["legal_actions"] = [
        {
            "action_id": "switch:slowbro",
            "kind": "switch",
            "label": "Switch to Slowbro",
            "details": {"species": "slowbro", "hp_fraction": 1.0},
        }
    ]
    protocol = "\n".join(
        [
            f">{battle_id}",
            "|init|battle",
            "|player|p1|ResearchPlayer 1|",
            "|player|p2|Opponent 1|",
            "|turn|2",
            "|move|p1a: Pikachu|Thunderbolt|p2a: Squirtle",
            "|move|p2a: Squirtle|Surf|p1a: Pikachu",
            "|faint|p1a: Pikachu",
            "|switch|p1a: Slowbro|Slowbro, L85|300/300",
            "|turn|3",
        ]
    )
    write_log(decision_log, move_decision, switch_decision)
    write_replay(replay, battle_id, protocol)

    build_replay_viewer(decision_log, replay_path=replay)

    payload = decoded_payload(
        decision_log.with_suffix(".viewer.html").read_text(encoding="utf-8")
    )
    anchors = [
        viewer_decision["replay_step"] for viewer_decision in payload["decisions"]
    ]
    switch_step = protocol.splitlines().index(
        "|switch|p1a: Slowbro|Slowbro, L85|300/300"
    ) + 1
    assert anchors[0] < anchors[1]
    assert anchors[1] == switch_step


def test_detects_research_player_on_player_two_side(tmp_path) -> None:
    battle_id = "battle-gen9randombattle-88"
    decision_log = tmp_path / "agent.jsonl"
    replay = tmp_path / f"ResearchPlayer 1 - {battle_id}.html"
    protocol = "\n".join(
        [
            f">{battle_id}",
            "|init|battle",
            "|player|p1|Opponent 1|",
            "|player|p2|ResearchPlayer 1|",
            "|turn|1",
            "|move|p2a: Pikachu|Thunderbolt|p1a: Squirtle",
        ]
    )
    write_log(decision_log, decision(battle_id))
    write_replay(replay, battle_id, protocol)

    build_replay_viewer(decision_log, replay_path=replay)

    payload = decoded_payload(
        decision_log.with_suffix(".viewer.html").read_text(encoding="utf-8")
    )
    assert payload["replay_sync"]["agent_side"] == "p2"
    assert isinstance(payload["decisions"][0]["replay_step"], int)


def test_rejects_replay_without_research_player_side(tmp_path) -> None:
    battle_id = "battle-gen9randombattle-91"
    decision_log = tmp_path / "agent.jsonl"
    replay = tmp_path / f"Agent - {battle_id}.html"
    protocol = "\n".join(
        [
            f">{battle_id}",
            "|player|p1|Player One|",
            "|player|p2|Player Two|",
            "|turn|1",
        ]
    )
    write_log(decision_log, decision(battle_id))
    write_replay(replay, battle_id, protocol)

    with pytest.raises(ViewerError, match="ResearchPlayer side"):
        build_replay_viewer(decision_log, replay_path=replay)


def test_missing_duplicate_action_anchor_keeps_manual_fallback(tmp_path) -> None:
    battle_id = "battle-gen9randombattle-92"
    decision_log = tmp_path / "agent.jsonl"
    replay = tmp_path / f"ResearchPlayer 1 - {battle_id}.html"
    first = decision(battle_id, turn=2)
    second = decision(battle_id, turn=2)
    second["action_id"] = "switch:slowbro"
    second["snapshot"]["legal_actions"] = []
    protocol = "\n".join(
        [
            f">{battle_id}",
            "|player|p1|ResearchPlayer 1|",
            "|player|p2|Opponent 1|",
            "|turn|2",
            "|move|p1a: Pikachu|Thunderbolt|p2a: Squirtle",
            "|turn|3",
        ]
    )
    write_log(decision_log, first, second)
    write_replay(replay, battle_id, protocol)

    build_replay_viewer(decision_log, replay_path=replay)

    payload = decoded_payload(
        decision_log.with_suffix(".viewer.html").read_text(encoding="utf-8")
    )
    assert isinstance(payload["decisions"][0]["replay_step"], int)
    assert payload["decisions"][1]["replay_step"] is None
    assert payload["replay_sync"]["anchored_decisions"] == 1
