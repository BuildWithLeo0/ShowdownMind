import json

from showdown_mind.domain import DecisionRecord
from showdown_mind.storage import JsonlDecisionWriter


def test_jsonl_writer_appends_decision_records(tmp_path) -> None:
    path = tmp_path / "nested" / "decisions.jsonl"
    writer = JsonlDecisionWriter(path)
    record = DecisionRecord(
        battle_id="battle-1",
        request_id=1,
        turn=1,
        snapshot_hash="sha256:test",
        snapshot={"battle_id": "battle-1"},
        action_id="move:tackle",
        fallback_used=False,
        attempts=1,
        model_ids=("model-1",),
        errors=(),
        raw_responses=('{"action_id":"move:tackle"}',),
        tool_name="choose_battle_action",
        tool_call_ids=("call-test",),
    )

    writer(record)
    writer(record)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["action_id"] == "move:tackle"
    assert rows[0]["model_ids"] == ["model-1"]
    assert rows[0]["tool_name"] == "choose_battle_action"
    assert rows[0]["tool_call_ids"] == ["call-test"]
