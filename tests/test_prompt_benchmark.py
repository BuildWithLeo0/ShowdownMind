import json

from showdown_mind.domain import BattleSnapshot, LegalAction
from showdown_mind.prompt_benchmark import benchmark_decision_log


def snapshot() -> BattleSnapshot:
    return BattleSnapshot(
        schema_version="1.0",
        battle_id="battle-1",
        request_id=1,
        turn=1,
        battle_format="gen9randombattle",
        own_side={"active": "pikachu", "team": [], "side_conditions": {}},
        opponent_side={
            "active": "eevee",
            "revealed_team": [],
            "side_conditions": {},
            "used_tera": False,
        },
        field={"weather": {}, "fields": {}},
        resources={
            "can_tera": False,
            "used_tera": False,
            "force_switch": False,
            "trapped": False,
        },
        legal_actions=(
            LegalAction(
                "move:tackle",
                "move",
                "Use tackle",
                {"move_id": "tackle", "base_power": 40},
            ),
        ),
    )


def test_prompt_benchmark_compares_logged_snapshots(tmp_path) -> None:
    path = tmp_path / "decisions.jsonl"
    row = json.dumps({"snapshot": snapshot().to_dict()})
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")

    result = benchmark_decision_log(path)

    assert result.decisions == 2
    assert result.pruned_characters < result.full_characters
    assert result.compact_characters < result.full_characters
    assert result.pruned_saved_characters > 0
    assert result.compact_saved_characters > 0
    assert result.pruned_reduction_percent > 0
    assert result.compact_reduction_percent > 0
