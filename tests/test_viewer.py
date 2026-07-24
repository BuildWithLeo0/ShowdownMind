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


def write_replay(path, battle_id: str) -> None:
    path.write_text(
        f"<!doctype html><title>{battle_id}</title><script>window.replay=true</script>",
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
