from types import SimpleNamespace

import pytest
from poke_env.battle import Move, Pokemon
from poke_env.player.battle_order import SingleBattleOrder

from showdown_mind.actions import ActionCatalog, UnknownActionError


def test_catalog_uses_stable_semantic_ids_and_resolves_orders() -> None:
    earthquake = SingleBattleOrder(Move("earthquake", gen=9))
    tera_earthquake = SingleBattleOrder(
        Move("earthquake", gen=9),
        terastallize=True,
    )
    switch = SingleBattleOrder(Pokemon(gen=9, species="Pikachu"))
    battle = SimpleNamespace(
        valid_orders=[switch, tera_earthquake, earthquake],
    )

    catalog = ActionCatalog.from_battle(battle)

    assert [action.action_id for action in catalog.actions] == [
        "move:earthquake",
        "move:earthquake:tera",
        "switch:pikachu",
    ]
    assert catalog.resolve("move:earthquake") is earthquake
    assert catalog.resolve("switch:pikachu") is switch


def test_catalog_rejects_unknown_action() -> None:
    battle = SimpleNamespace(valid_orders=[SingleBattleOrder(Move("tackle", gen=9))])
    catalog = ActionCatalog.from_battle(battle)

    with pytest.raises(UnknownActionError, match="not-real"):
        catalog.resolve("not-real")
