from __future__ import annotations

from typing import Any

from poke_env.battle import Move, Pokemon
from poke_env.player.battle_order import SingleBattleOrder

from showdown_mind.actions import ActionCatalog

TACTICAL_ANALYSIS_SCHEMA = "tactical-analysis-v1"

ENTRY_HAZARDS = {
    "ceaselessedge",
    "spikes",
    "stealthrock",
    "stickyweb",
    "stoneaxe",
    "toxicspikes",
}
HAZARD_REMOVAL = {
    "courtchange",
    "defog",
    "mortalspin",
    "rapidspin",
    "tidyup",
}


class TacticalAdvisor:
    """Calculate player-visible tactical facts for the current legal actions."""

    def analyze(self, battle: Any, catalog: ActionCatalog) -> dict[str, Any]:
        active = getattr(battle, "active_pokemon", None)
        opponent = getattr(battle, "opponent_active_pokemon", None)
        own_conditions = getattr(battle, "side_conditions", None) or {}
        opponent_conditions = (
            getattr(battle, "opponent_side_conditions", None) or {}
        )
        trick_room = _has_named_effect(
            getattr(battle, "fields", None) or {},
            "trick_room",
        )
        speed_relation = _speed_relation(
            active,
            opponent,
            left_conditions=own_conditions,
            right_conditions=opponent_conditions,
            reverse=trick_room,
        )
        actions: list[dict[str, Any]] = []

        for action in catalog.actions:
            order = catalog.resolve(action.action_id)
            if isinstance(order, SingleBattleOrder) and isinstance(order.order, Move):
                actions.append(
                    self._analyze_move(
                        action.action_id,
                        order,
                        active,
                        opponent,
                        speed_relation,
                    )
                )
            elif isinstance(order, SingleBattleOrder) and isinstance(
                order.order, Pokemon
            ):
                actions.append(
                    self._analyze_switch(
                        action.action_id,
                        order.order,
                        opponent,
                        own_conditions,
                        opponent_conditions,
                        trick_room,
                    )
                )
            else:
                actions.append(
                    {
                        "action_id": action.action_id,
                        "kind": action.kind,
                        "note": "No tactical estimate is available.",
                    }
                )

        damaging = [
            action
            for action in actions
            if action.get("kind") == "move"
            and float(action.get("damage_index") or 0.0) > 0
        ]
        best_damage = max(
            (float(action["damage_index"]) for action in damaging),
            default=0.0,
        )
        for action in damaging:
            action["relative_damage"] = round(
                float(action["damage_index"]) / best_damage
                if best_damage > 0
                else 0.0,
                4,
            )

        switches = [
            action for action in actions if action.get("kind") == "switch"
        ]
        best_switch_score = max(
            (float(action["matchup_score"]) for action in switches),
            default=None,
        )
        return {
            "schema": TACTICAL_ANALYSIS_SCHEMA,
            "active": _species(active),
            "opponent_active": _species(opponent),
            "speed_relation": speed_relation,
            "speed_context": {
                "trick_room": trick_room,
                "own_tailwind": _has_named_effect(
                    own_conditions,
                    "tailwind",
                ),
                "opponent_tailwind": _has_named_effect(
                    opponent_conditions,
                    "tailwind",
                ),
            },
            "best_damage_action_ids": [
                action["action_id"]
                for action in damaging
                if float(action["damage_index"]) == best_damage
            ],
            "best_switch_action_ids": [
                action["action_id"]
                for action in switches
                if best_switch_score is not None
                and float(action["matchup_score"]) == best_switch_score
            ],
            "actions": actions,
            "limitations": (
                "Damage values are relative estimates based on public species, "
                "types, boosts, move data, and HP. Hidden EVs, unrevealed items, "
                "abilities, and moves are not used. Variable-power moves are "
                "marked dynamic and are not ranked in v1."
            ),
        }

    @staticmethod
    def _analyze_move(
        action_id: str,
        order: SingleBattleOrder,
        active: Any,
        opponent: Any,
        speed_relation: str,
    ) -> dict[str, Any]:
        move = order.order
        category = _enum_name(getattr(move, "category", None))
        base_power = max(0.0, _number(getattr(move, "base_power", 0.0)))
        dynamic_power = base_power == 0 and category in {"physical", "special"}
        accuracy = _accuracy(getattr(move, "accuracy", 1.0))
        expected_hits = max(1.0, _number(getattr(move, "expected_hits", 1.0)))
        type_multiplier = _damage_multiplier(opponent, move)
        stab = _stab_multiplier(active, move, bool(order.terastallize))
        stat_ratio = _offensive_stat_ratio(active, opponent, move)
        damage_index = None
        if not dynamic_power:
            damage_index = (
                base_power
                * accuracy
                * expected_hits
                * type_multiplier
                * stab
                * stat_ratio
            )
        priority = int(_number(getattr(move, "priority", 0)))
        return {
            "action_id": action_id,
            "kind": "move",
            "move_id": str(getattr(move, "id", "")),
            "category": category,
            "base_power": round(base_power, 3),
            "dynamic_power": dynamic_power,
            "accuracy": round(accuracy, 4),
            "expected_hits": round(expected_hits, 3),
            "stab_multiplier": round(stab, 3),
            "type_multiplier": round(type_multiplier, 3),
            "stat_ratio": round(stat_ratio, 4),
            "damage_index": (
                round(damage_index, 4) if damage_index is not None else None
            ),
            "priority": priority,
            "move_order": _move_order(priority, speed_relation),
            "terastallize": bool(order.terastallize),
            "role_tags": _move_role_tags(move),
        }

    @staticmethod
    def _analyze_switch(
        action_id: str,
        candidate: Pokemon,
        opponent: Any,
        own_conditions: dict[Any, Any],
        opponent_conditions: dict[Any, Any],
        trick_room: bool,
    ) -> dict[str, Any]:
        candidate_types = [
            value for value in (getattr(candidate, "types", None) or []) if value
        ]
        opponent_types = [
            value for value in (getattr(opponent, "types", None) or []) if value
        ]
        offensive = max(
            (_damage_multiplier(opponent, value) for value in candidate_types),
            default=1.0,
        )
        defensive = max(
            (_damage_multiplier(candidate, value) for value in opponent_types),
            default=1.0,
        )
        speed_relation = _speed_relation(
            candidate,
            opponent,
            left_conditions=own_conditions,
            right_conditions=opponent_conditions,
            reverse=trick_room,
        )
        speed_adjustment = {
            "faster": 0.25,
            "slower": -0.25,
            "tie": 0.0,
            "unknown": 0.0,
        }[speed_relation]
        hp_fraction = _number(
            getattr(candidate, "current_hp_fraction", 0.0)
        )
        opponent_hp = _number(
            getattr(opponent, "current_hp_fraction", 0.0)
        )
        matchup_score = (
            offensive
            - defensive
            + speed_adjustment
            + 0.4 * (hp_fraction - opponent_hp)
        )
        return {
            "action_id": action_id,
            "kind": "switch",
            "species": _species(candidate),
            "hp_fraction": round(hp_fraction, 4),
            "offensive_type_multiplier": round(offensive, 3),
            "defensive_weakness_multiplier": round(defensive, 3),
            "speed_relation": speed_relation,
            "matchup_score": round(matchup_score, 4),
        }


def _offensive_stat_ratio(active: Any, opponent: Any, move: Move) -> float:
    category = _enum_name(getattr(move, "category", None))
    if category == "physical":
        attack = _estimated_stat(active, "atk")
        defense = _estimated_stat(opponent, "def")
    elif category == "special":
        attack = _estimated_stat(active, "spa")
        defense = _estimated_stat(opponent, "spd")
    else:
        return 0.0
    return attack / max(defense, 1.0)


def _estimated_stat(pokemon: Any, stat: str) -> float:
    if pokemon is None:
        return 100.0
    base_stats = getattr(pokemon, "base_stats", None) or {}
    base = _number(base_stats.get(stat, 100))
    boosts = getattr(pokemon, "boosts", None) or {}
    stage = max(-6, min(6, int(_number(boosts.get(stat, 0)))))
    boost = (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
    level = _number(getattr(pokemon, "level", 100)) or 100
    unboosted = ((2 * base + 31) * level / 100) + 5
    return max(1.0, unboosted * boost)


def _speed_relation(
    left: Any,
    right: Any,
    *,
    left_conditions: dict[Any, Any] | None = None,
    right_conditions: dict[Any, Any] | None = None,
    reverse: bool = False,
) -> str:
    if left is None or right is None:
        return "unknown"
    left_speed = _effective_speed(left, left_conditions or {})
    right_speed = _effective_speed(right, right_conditions or {})
    if left_speed > right_speed:
        return "slower" if reverse else "faster"
    if left_speed < right_speed:
        return "faster" if reverse else "slower"
    return "tie"


def _effective_speed(pokemon: Any, side_conditions: dict[Any, Any]) -> float:
    speed = _estimated_stat(pokemon, "spe")
    if _enum_name(getattr(pokemon, "status", None)) == "par":
        speed *= 0.5
    if _has_named_effect(side_conditions, "tailwind"):
        speed *= 2
    return speed


def _move_order(priority: int, speed_relation: str) -> str:
    if priority > 0:
        return "likely_first"
    if priority < 0:
        return "likely_second"
    return {
        "faster": "likely_first",
        "slower": "likely_second",
        "tie": "speed_tie",
        "unknown": "unknown",
    }[speed_relation]


def _stab_multiplier(active: Any, move: Move, terastallize: bool) -> float:
    move_type = getattr(move, "type", None)
    original_types = getattr(active, "types", None) or []
    original_stab = move_type is not None and move_type in original_types
    tera_type = getattr(active, "tera_type", None)
    if terastallize and move_type is not None and move_type == tera_type:
        return 2.0 if original_stab else 1.5
    return 1.5 if original_stab else 1.0


def _damage_multiplier(target: Any, type_or_move: Any) -> float:
    if target is None or not hasattr(target, "damage_multiplier"):
        return 1.0
    try:
        return max(0.0, _number(target.damage_multiplier(type_or_move)))
    except (AttributeError, KeyError, TypeError, ValueError):
        return 1.0


def _move_role_tags(move: Move) -> list[str]:
    move_id = str(getattr(move, "id", ""))
    tags: list[str] = []
    if move_id in ENTRY_HAZARDS:
        tags.append("entry_hazard")
    if move_id in HAZARD_REMOVAL:
        tags.append("hazard_removal")
    boosts = getattr(move, "boosts", None) or {}
    if any(_number(value) > 0 for value in boosts.values()):
        tags.append("setup")
    if _number(getattr(move, "heal", 0)) > 0:
        tags.append("healing")
    if _number(getattr(move, "recoil", 0)) > 0:
        tags.append("recoil")
    if _number(getattr(move, "drain", 0)) > 0:
        tags.append("drain")
    return tags


def _accuracy(value: Any) -> float:
    if value is True or value is None:
        return 1.0
    number = _number(value)
    if number > 1:
        number /= 100
    return max(0.0, min(1.0, number))


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "name", value)).lower()


def _has_named_effect(values: dict[Any, Any], name: str) -> bool:
    return any(_enum_name(value) == name for value in values)


def _species(pokemon: Any) -> str | None:
    if pokemon is None:
        return None
    value = str(getattr(pokemon, "species", ""))
    return value or None
