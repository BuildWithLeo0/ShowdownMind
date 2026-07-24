from showdown_mind.domain import BattleSnapshot, LegalAction
from showdown_mind.policy_input import compile_policy_input


def rich_snapshot() -> BattleSnapshot:
    own_pokemon = {
        "species": "pikachu",
        "name": "Sparky",
        "active": True,
        "fainted": False,
        "hp_fraction": 0.75,
        "status": None,
        "types": ["electric"],
        "boosts": {"spa": 1},
        "item": "lightball",
        "ability": "static",
        "moves": ["quickattack", "thunderbolt"],
        "tera_type": "electric",
        "information_scope": "own",
    }
    opponent = {
        "species": "gyarados",
        "name": "gyarados",
        "active": True,
        "fainted": False,
        "hp_fraction": 0.5,
        "status": "par",
        "types": ["water", "flying"],
        "boosts": {},
        "item": None,
        "ability": "intimidate",
        "moves": ["waterfall"],
        "tera_type": None,
        "information_scope": "revealed",
    }
    return BattleSnapshot(
        schema_version="1.0",
        battle_id="battle-secret-metadata",
        request_id=9,
        turn=4,
        battle_format="gen9randombattle",
        own_side={
            "active": "pikachu",
            "team": [own_pokemon],
            "side_conditions": {"stealthrock": 1},
        },
        opponent_side={
            "active": "gyarados",
            "revealed_team": [opponent],
            "side_conditions": {},
            "used_tera": False,
        },
        field={"weather": {}, "fields": {}},
        resources={
            "can_tera": True,
            "used_tera": False,
            "force_switch": False,
            "trapped": False,
        },
        legal_actions=(
            LegalAction(
                "move:thunderbolt",
                "move",
                "Use thunderbolt",
                {
                    "move_id": "thunderbolt",
                    "type": "electric",
                    "category": "special",
                    "base_power": 90,
                    "accuracy": 100,
                    "terastallize": False,
                },
            ),
        ),
    )


def test_compact_input_preserves_tactical_facts_and_removes_redundancy() -> None:
    compact = compile_policy_input(rich_snapshot(), "compact")

    assert compact.format_name == "compact-v1"
    assert compact.payload["own"]["team"][0] == {
        "species": "pikachu",
        "hp": 0.75,
        "types": ["electric"],
        "boosts": {"spa": 1},
        "item": "lightball",
        "ability": "static",
        "moves": ["quickattack", "thunderbolt"],
        "tera_type": "electric",
    }
    assert compact.payload["opponent"]["team"][0]["moves"] == ["waterfall"]
    assert compact.payload["legal_actions"][0]["base_power"] == 90
    assert "battle_id" not in compact.payload
    assert "label" not in compact.payload["legal_actions"][0]
    assert "used_tera" not in compact.payload["resources"]


def test_compact_input_is_smaller_and_deterministic() -> None:
    snapshot = rich_snapshot()
    full = compile_policy_input(snapshot, "full")
    first = compile_policy_input(snapshot, "compact")
    second = compile_policy_input(snapshot, "compact")

    assert full.payload == snapshot.to_dict()
    assert first.characters < full.characters
    assert first.fingerprint() == second.fingerprint()
    assert first.canonical_json() == second.canonical_json()


def test_pruned_input_preserves_readable_schema_and_removes_empty_values() -> None:
    pruned = compile_policy_input(rich_snapshot(), "pruned")

    assert pruned.format_name == "pruned-v1"
    assert pruned.payload["own_side"]["team"][0]["species"] == "pikachu"
    assert pruned.payload["legal_actions"][0]["details"]["base_power"] == 90
    assert "battle_id" not in pruned.payload
    assert "name" not in pruned.payload["own_side"]["team"][0]
    assert "status" not in pruned.payload["own_side"]["team"][0]
    assert "used_tera" not in pruned.payload["resources"]


def test_snapshot_round_trips_from_logged_dictionary() -> None:
    original = rich_snapshot()

    restored = BattleSnapshot.from_dict(original.to_dict())

    assert restored == original
