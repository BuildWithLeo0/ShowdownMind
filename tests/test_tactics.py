import json
from types import SimpleNamespace

from poke_env.battle import Move, Pokemon
from poke_env.player.battle_order import SingleBattleOrder

from showdown_mind.actions import ActionCatalog, CatalogEntry
from showdown_mind.domain import LegalAction
from showdown_mind.tactics import (
    TACTICAL_ANALYSIS_SCHEMA,
    TACTICAL_MODEL_VIEW,
    TacticalAdvisor,
    compact_tactical_analysis_for_model,
)


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


def healthy_pokemon(species: str) -> Pokemon:
    pokemon = Pokemon(gen=9, species=species)
    pokemon.set_hp_status("100/100", store=True)
    return pokemon


def switch_entry(pokemon: Pokemon) -> CatalogEntry:
    return CatalogEntry(
        action=LegalAction(
            action_id=f"switch:{pokemon.species}",
            kind="switch",
            label=f"Switch to {pokemon.species}",
        ),
        order=SingleBattleOrder(pokemon),
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


def test_advisor_estimates_counterplay_from_revealed_moves_only() -> None:
    active = healthy_pokemon("Pikachu")
    opponent = healthy_pokemon("Gyarados")
    opponent.moved("earthquake")

    result = TacticalAdvisor().analyze(
        SimpleNamespace(
            active_pokemon=active,
            opponent_active_pokemon=opponent,
        ),
        ActionCatalog(
            (
                move_entry("thunderbolt"),
                move_entry("protect"),
            )
        ),
    )

    actions = {action["action_id"]: action for action in result["actions"]}
    attack = actions["move:thunderbolt"]["counterplay"]
    protect = actions["move:protect"]["counterplay"]
    assert attack["basis"] == "revealed_opponent_moves"
    assert attack["worst_move_id"] == "earthquake"
    assert attack["player_acts_before_reply"] is True
    assert attack["estimated_counter_ko_probability"] > 0
    assert protect["estimated_counter_ko_probability"] == 0.0
    assert protect["risk"] == "survives_known_reply"
    assert result["safest_action_ids"] == ["move:protect"]


def test_advisor_uses_public_candidate_moves_without_treating_them_as_revealed() -> None:
    active = healthy_pokemon("Pikachu")
    opponent = healthy_pokemon("Gyarados")

    result = TacticalAdvisor().analyze(
        SimpleNamespace(
            active_pokemon=active,
            opponent_active_pokemon=opponent,
        ),
        ActionCatalog((move_entry("thunderbolt"),)),
        opponent_candidate_move_ids=("earthquake",),
    )

    counterplay = result["actions"][0]["counterplay"]
    compact = compact_tactical_analysis_for_model(result)

    assert result["opponent_candidate_move_ids"] == ["earthquake"]
    assert counterplay["basis"] == "revealed_and_public_prior_moves"
    assert counterplay["revealed_moves_considered"] == 0
    assert counterplay["candidate_moves_considered"] == 1
    assert counterplay["worst_move_id"] == "earthquake"
    assert compact["opponent_move_information"] == {
        "public_candidate_move_ids": ["earthquake"],
        "candidate_moves_are_hypotheses": True,
    }


def test_advisor_applies_visible_weather_and_screen_modifiers() -> None:
    active = healthy_pokemon("Charizard")
    opponent = healthy_pokemon("Venusaur")

    result = TacticalAdvisor().analyze(
        SimpleNamespace(
            active_pokemon=active,
            opponent_active_pokemon=opponent,
            opponent_side_conditions={"light_screen": 1},
            weather={"sunnyday": 1},
        ),
        ActionCatalog((move_entry("flamethrower"),)),
    )

    action = result["actions"][0]
    assert action["battle_modifier"] == 0.75
    assert action["modifier_sources"] == [
        "sun_fire_boost",
        "light_screen",
    ]


def test_advisor_applies_burn_and_terrain_modifiers() -> None:
    burned = healthy_pokemon("Garchomp")
    burned.status = "brn"
    target = healthy_pokemon("Blissey")
    burned_result = TacticalAdvisor().analyze(
        SimpleNamespace(
            active_pokemon=burned,
            opponent_active_pokemon=target,
        ),
        ActionCatalog((move_entry("earthquake"),)),
    )
    assert burned_result["actions"][0]["battle_modifier"] == 0.5
    assert burned_result["actions"][0]["modifier_sources"] == ["burn"]

    electric = healthy_pokemon("Pikachu")
    water_target = healthy_pokemon("Blastoise")
    terrain_result = TacticalAdvisor().analyze(
        SimpleNamespace(
            active_pokemon=electric,
            opponent_active_pokemon=water_target,
            fields={"electric_terrain": 1},
        ),
        ActionCatalog((move_entry("thunderbolt"),)),
    )
    assert terrain_result["actions"][0]["battle_modifier"] == 1.3
    assert terrain_result["actions"][0]["modifier_sources"] == [
        "electric_terrain"
    ]


def test_repeated_protect_reports_reduced_success_probability() -> None:
    active = healthy_pokemon("Pikachu")
    active.moved("protect")
    opponent = healthy_pokemon("Gyarados")
    opponent.moved("earthquake")

    result = TacticalAdvisor().analyze(
        SimpleNamespace(
            active_pokemon=active,
            opponent_active_pokemon=opponent,
        ),
        ActionCatalog((move_entry("protect"),)),
    )

    counterplay = result["actions"][0]["counterplay"]
    assert counterplay["protect_success_probability"] == 0.3333
    assert counterplay["estimated_counter_ko_probability"] == 0.6667


def test_advisor_applies_entry_hazards_before_switch_counterplay() -> None:
    active = healthy_pokemon("Pikachu")
    opponent = healthy_pokemon("Blastoise")
    opponent.moved("surf")
    candidate = healthy_pokemon("Charizard")

    result = TacticalAdvisor().analyze(
        SimpleNamespace(
            active_pokemon=active,
            opponent_active_pokemon=opponent,
            side_conditions={"stealth_rock": 1},
        ),
        ActionCatalog((switch_entry(candidate),)),
    )

    switch = result["actions"][0]
    assert switch["entry_hazards"]["damage_fraction"] == 0.5
    assert switch["entry_hazards"]["post_entry_hp_fraction"] == 0.5
    assert switch["counterplay"]["worst_move_id"] == "surf"
    assert switch["counterplay"]["player_acts_before_reply"] is True


def test_heavy_duty_boots_prevent_entry_hazard_effects() -> None:
    active = healthy_pokemon("Pikachu")
    opponent = healthy_pokemon("Blastoise")
    candidate = healthy_pokemon("Charizard")
    candidate.item = "heavydutyboots"

    result = TacticalAdvisor().analyze(
        SimpleNamespace(
            active_pokemon=active,
            opponent_active_pokemon=opponent,
            side_conditions={
                "spikes": 3,
                "stealth_rock": 1,
                "sticky_web": 1,
            },
        ),
        ActionCatalog((switch_entry(candidate),)),
    )

    entry = result["actions"][0]["entry_hazards"]
    assert entry["damage_fraction"] == 0.0
    assert entry["post_entry_hp_fraction"] == 1.0
    assert entry["effects"] == ["heavy_duty_boots"]


def test_model_view_is_smaller_but_keeps_decision_facts() -> None:
    active = healthy_pokemon("Pikachu")
    opponent = healthy_pokemon("Gyarados")
    opponent.moved("earthquake")
    full = TacticalAdvisor().analyze(
        SimpleNamespace(
            active_pokemon=active,
            opponent_active_pokemon=opponent,
        ),
        ActionCatalog(
            (
                move_entry("thunderbolt"),
                move_entry("protect"),
            )
        ),
    )

    compact = compact_tactical_analysis_for_model(full)

    assert compact["schema"] == TACTICAL_ANALYSIS_SCHEMA
    assert compact["view"] == TACTICAL_MODEL_VIEW
    assert compact["safest_action_ids"] == ["move:protect"]
    action = compact["actions"][0]
    assert action["action_id"] == "move:thunderbolt"
    assert action["counterplay"]["worst_move_id"] == "earthquake"
    assert "attack_stat_estimate" not in action
    assert "raw_incoming_ko_probability" not in action["counterplay"]
    assert len(json.dumps(compact)) < len(json.dumps(full)) * 0.7
