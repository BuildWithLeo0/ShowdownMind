from types import SimpleNamespace

from showdown_mind.domain import LegalAction
from showdown_mind.observation import BattleSnapshotBuilder


def pokemon(
    species: str,
    *,
    revealed: bool,
    move_id: str,
    item: str,
    ability: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        species=species,
        name=species,
        revealed=revealed,
        active=False,
        fainted=False,
        current_hp_fraction=1.0,
        status=None,
        types=[],
        boosts={},
        item=item,
        ability=ability,
        moves={move_id: SimpleNamespace(id=move_id)},
        tera_type=None,
    )


def visible_battle() -> SimpleNamespace:
    own = pokemon(
        "Pikachu",
        revealed=True,
        move_id="thunderbolt",
        item="lightball",
        ability="static",
    )
    shown = pokemon(
        "Gengar",
        revealed=True,
        move_id="shadowball",
        item="unknown_item",
        ability="cursedbody",
    )
    hidden = pokemon(
        "SECRET_SPECIES",
        revealed=False,
        move_id="SECRET_MOVE",
        item="SECRET_ITEM",
        ability="SECRET_ABILITY",
    )
    own.active = True
    shown.active = True
    return SimpleNamespace(
        battle_tag="battle-test-1",
        last_request={"rqid": 7, "SECRET_REQUEST_FIELD": "must-not-leak"},
        turn=3,
        format="gen9randombattle",
        team={"p1": own},
        opponent_team={"shown": shown, "hidden": hidden},
        active_pokemon=own,
        opponent_active_pokemon=shown,
        side_conditions={},
        opponent_side_conditions={},
        weather={},
        fields={},
        can_tera=True,
        used_tera=False,
        opponent_used_tera=False,
        force_switch=False,
        trapped=False,
    )


def test_snapshot_is_whitelist_only_and_excludes_hidden_opponent_data() -> None:
    snapshot = BattleSnapshotBuilder().build(
        visible_battle(),
        (LegalAction("move:thunderbolt", "move", "Use thunderbolt"),),
    )
    encoded = snapshot.canonical_json()

    assert snapshot.request_id == 7
    assert snapshot.opponent_side["revealed_team"][0]["species"] == "Gengar"
    assert "shadowball" in encoded
    assert "SECRET_" not in encoded
    assert "must-not-leak" not in encoded


def test_opponent_without_explicit_revealed_flag_is_hidden() -> None:
    battle = visible_battle()
    del battle.opponent_team["hidden"].revealed

    snapshot = BattleSnapshotBuilder().build(battle, ())

    assert len(snapshot.opponent_side["revealed_team"]) == 1
