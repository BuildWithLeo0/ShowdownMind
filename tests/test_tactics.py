from types import SimpleNamespace

from poke_env.battle import Move, Pokemon
from poke_env.player.battle_order import SingleBattleOrder

from showdown_mind.actions import ActionCatalog, CatalogEntry
from showdown_mind.domain import LegalAction
from showdown_mind.tactics import TACTICAL_ANALYSIS_SCHEMA, TacticalAdvisor


def move_entry(move_id: str, *, tera: bool = False) -> CatalogEntry:
    move = Move(move_id, gen=9)
    suffix = ":tera" if tera else ""
    return CatalogEntry(
        action=LegalAction(
            action_id=f"move:{move_id}{suffix}",
            kind="move",
            label=f"Use {move_id}",
        ),
        order=SingleBattleOrder(move, terastallize=tera),
    )


def test_advisor_ranks_player_visible_damage_facts() -> None:
    active = Pokemon(gen=9, species="Pikachu")
    opponent = Pokemon(gen=9, species="Gyarados")
    catalog = ActionCatalog(
        (
            move_entry("thunderbolt"),
            move_entry("quickattack"),
            move_entry("thunderwave"),
        )
    )

    result = TacticalAdvisor().analyze(
        SimpleNamespace(
            active_pokemon=active,
            opponent_active_pokemon=opponent,
        ),
        catalog,
    )

    assert result["schema"] == TACTICAL_ANALYSIS_SCHEMA
    assert result["speed_relation"] == "faster"
    assert result["best_damage_action_ids"] == ["move:thunderbolt"]
    actions = {action["action_id"]: action for action in result["actions"]}
    assert actions["move:thunderbolt"]["type_multiplier"] == 4.0
    assert actions["move:thunderbolt"]["stab_multiplier"] == 1.5
    assert actions["move:thunderbolt"]["relative_damage"] == 1.0
    assert actions["move:quickattack"]["priority"] == 1
    assert actions["move:thunderwave"]["damage_index"] == 0.0


def test_advisor_scores_legal_switches_without_hidden_team_data() -> None:
    active = Pokemon(gen=9, species="Charmander")
    opponent = Pokemon(gen=9, species="Blastoise")
    switch = Pokemon(gen=9, species="Venusaur")
    switch_entry = CatalogEntry(
        action=LegalAction(
            action_id="switch:venusaur",
            kind="switch",
            label="Switch to Venusaur",
        ),
        order=SingleBattleOrder(switch),
    )

    result = TacticalAdvisor().analyze(
        SimpleNamespace(
            active_pokemon=active,
            opponent_active_pokemon=opponent,
        ),
        ActionCatalog((switch_entry,)),
    )

    switch_result = result["actions"][0]
    assert result["best_switch_action_ids"] == ["switch:venusaur"]
    assert switch_result["offensive_type_multiplier"] == 2.0
    assert switch_result["defensive_weakness_multiplier"] == 0.5
    assert "moves" not in switch_result


def test_advisor_accounts_for_trick_room_in_move_order() -> None:
    active = Pokemon(gen=9, species="Pikachu")
    opponent = Pokemon(gen=9, species="Gyarados")
    catalog = ActionCatalog((move_entry("thunderbolt"),))

    result = TacticalAdvisor().analyze(
        SimpleNamespace(
            active_pokemon=active,
            opponent_active_pokemon=opponent,
            side_conditions={},
            opponent_side_conditions={},
            fields={"trick_room": 1},
        ),
        catalog,
    )

    assert result["speed_relation"] == "slower"
    assert result["speed_context"]["trick_room"] is True
    assert result["actions"][0]["move_order"] == "likely_second"
