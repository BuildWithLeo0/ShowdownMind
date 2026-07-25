from __future__ import annotations

import math
from typing import Any

from poke_env.battle import Move, Pokemon
from poke_env.data import GenData
from poke_env.player.battle_order import SingleBattleOrder

from showdown_mind.actions import ActionCatalog

TACTICAL_ANALYSIS_SCHEMA = "tactical-analysis-v2"

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
        battle_gen = int(
            _number(getattr(battle, "gen", None))
            or _number(getattr(active, "gen", None))
            or 9
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
                        own_conditions,
                        opponent_conditions,
                        battle_gen,
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
        ko_estimates = [
            float(action["estimated_ko_probability"])
            for action in damaging
            if action.get("estimated_ko_probability") is not None
        ]
        best_ko_probability = max(ko_estimates) if ko_estimates else None

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
            "best_ko_action_ids": [
                action["action_id"]
                for action in damaging
                if best_ko_probability is not None
                and best_ko_probability > 0
                and action.get("estimated_ko_probability") is not None
                and float(action["estimated_ko_probability"])
                == best_ko_probability
            ],
            "best_ko_probability": (
                round(best_ko_probability, 4)
                if best_ko_probability is not None
                else None
            ),
            "best_switch_action_ids": [
                action["action_id"]
                for action in switches
                if best_switch_score is not None
                and float(action["matchup_score"]) == best_switch_score
            ],
            "actions": actions,
            "limitations": (
                "Damage ranges and KO probabilities are approximations based on "
                "public species, level, types, boosts, move data, and HP. They "
                "use exact visible stats when available; missing stats assume 31 "
                "IVs, zero EVs, and a neutral nature. Hidden items, abilities, "
                "weather modifiers, screens, critical hits, and other special "
                "effects are omitted. Supported variable-power moves are "
                "estimated; unknown dynamic moves remain unranked. Defensive Tera "
                "compares only the opponent's currently visible STAB types, not "
                "unrevealed moves."
            ),
        }

    @staticmethod
    def _analyze_move(
        action_id: str,
        order: SingleBattleOrder,
        active: Any,
        opponent: Any,
        speed_relation: str,
        own_conditions: dict[Any, Any],
        opponent_conditions: dict[Any, Any],
        battle_gen: int,
    ) -> dict[str, Any]:
        move = order.order
        category = _enum_name(getattr(move, "category", None))
        raw_base_power = max(0.0, _number(getattr(move, "base_power", 0.0)))
        effective_power, power_source = _effective_move_power(
            move,
            active,
            opponent,
            own_conditions=own_conditions,
            opponent_conditions=opponent_conditions,
            speed_relation=speed_relation,
        )
        dynamic_power = power_source != "fixed"
        accuracy = _accuracy(getattr(move, "accuracy", 1.0))
        expected_hits = max(1.0, _number(getattr(move, "expected_hits", 1.0)))
        move_type = _effective_move_type(
            active,
            move,
            terastallize=bool(order.terastallize),
        )
        type_multiplier = _damage_multiplier(opponent, move_type or move)
        stab = _stab_multiplier(
            active,
            move_type,
            bool(order.terastallize),
        )
        attack_stat, defense_stat, stat_source = _offensive_stats(
            active,
            opponent,
            move,
            terastallize=(
                bool(order.terastallize)
                or bool(getattr(active, "is_terastallized", False))
            ),
        )
        effective_category = (
            "physical"
            if stat_source == "tera_blast_physical"
            else "special"
            if stat_source == "tera_blast_special"
            else category
        )
        stat_ratio = (
            attack_stat / max(defense_stat, 1.0)
            if effective_category in {"physical", "special"}
            else 0.0
        )
        damage_range = _estimated_damage_fraction_range(
            active,
            opponent,
            power=effective_power,
            attack_stat=attack_stat,
            defense_stat=defense_stat,
            stab=stab,
            type_multiplier=type_multiplier,
            expected_hits=expected_hits,
        )
        ko_probability = _estimated_ko_probability(
            active,
            opponent,
            power=effective_power,
            attack_stat=attack_stat,
            defense_stat=defense_stat,
            stab=stab,
            type_multiplier=type_multiplier,
            expected_hits=expected_hits,
            accuracy=accuracy,
        )
        damage_index = None
        if damage_range is not None:
            damage_index = accuracy * sum(damage_range) / 2
        priority = int(_number(getattr(move, "priority", 0)))
        result = {
            "action_id": action_id,
            "kind": "move",
            "move_id": str(getattr(move, "id", "")),
            "move_type": _enum_name(move_type),
            "category": category,
            "effective_category": effective_category,
            "base_power": round(raw_base_power, 3),
            "effective_power": (
                round(effective_power, 3)
                if effective_power is not None
                else None
            ),
            "power_source": power_source,
            "dynamic_power": dynamic_power,
            "accuracy": round(accuracy, 4),
            "expected_hits": round(expected_hits, 3),
            "stab_multiplier": round(stab, 3),
            "type_multiplier": round(type_multiplier, 3),
            "attack_stat_estimate": round(attack_stat, 3),
            "defense_stat_estimate": round(defense_stat, 3),
            "stat_source": stat_source,
            "stat_ratio": round(stat_ratio, 4),
            "estimated_damage_fraction_range": (
                [round(value, 4) for value in damage_range]
                if damage_range is not None
                else None
            ),
            "estimated_ko_probability": (
                round(ko_probability, 4)
                if ko_probability is not None
                else None
            ),
            "damage_index": (
                round(damage_index, 4) if damage_index is not None else None
            ),
            "priority": priority,
            "move_order": _move_order(priority, speed_relation),
            "terastallize": bool(order.terastallize),
            "role_tags": _move_role_tags(move),
        }
        if order.terastallize:
            result["defensive_tera"] = _defensive_tera_estimate(
                active,
                opponent,
                battle_gen,
            )
        return result

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


def _effective_move_power(
    move: Move,
    active: Any,
    opponent: Any,
    *,
    own_conditions: dict[Any, Any],
    opponent_conditions: dict[Any, Any],
    speed_relation: str,
) -> tuple[float | None, str]:
    move_id = str(getattr(move, "id", ""))
    category = _enum_name(getattr(move, "category", None))
    raw_power = max(0.0, _number(getattr(move, "base_power", 0.0)))

    if move_id in {"heavyslam", "heatcrash"}:
        user_weight = _positive_number(getattr(active, "weight", None))
        target_weight = _positive_number(getattr(opponent, "weight", None))
        if user_weight is None or target_weight is None:
            return None, "unknown_weight_ratio"
        ratio = user_weight / target_weight
        if ratio >= 5:
            return 120.0, "weight_ratio"
        if ratio >= 4:
            return 100.0, "weight_ratio"
        if ratio >= 3:
            return 80.0, "weight_ratio"
        if ratio >= 2:
            return 60.0, "weight_ratio"
        return 40.0, "weight_ratio"

    if move_id in {"lowkick", "grassknot"}:
        target_weight = _positive_number(getattr(opponent, "weight", None))
        if target_weight is None:
            return None, "unknown_target_weight"
        if target_weight < 10:
            return 20.0, "target_weight"
        if target_weight < 25:
            return 40.0, "target_weight"
        if target_weight < 50:
            return 60.0, "target_weight"
        if target_weight < 100:
            return 80.0, "target_weight"
        if target_weight < 200:
            return 100.0, "target_weight"
        return 120.0, "target_weight"

    if move_id in {"electroball", "gyroball"}:
        user_speed = _positive_number(_effective_speed(active, own_conditions))
        target_speed = _positive_number(
            _effective_speed(opponent, opponent_conditions)
        )
        if user_speed is None or target_speed is None:
            return None, "unknown_speed_ratio"
        if move_id == "gyroball":
            return (
                float(min(150, math.floor(25 * target_speed / user_speed) + 1)),
                "speed_ratio",
            )
        ratio = user_speed / target_speed
        if ratio >= 4:
            return 150.0, "speed_ratio"
        if ratio >= 3:
            return 120.0, "speed_ratio"
        if ratio >= 2:
            return 80.0, "speed_ratio"
        if ratio >= 1:
            return 60.0, "speed_ratio"
        return 40.0, "speed_ratio"

    if move_id in {"storedpower", "powertrip"}:
        boosts = getattr(active, "boosts", None) or {}
        positive_stages = sum(
            max(0, int(_number(value))) for value in boosts.values()
        )
        return float(20 + 20 * positive_stages), "positive_boosts"

    if move_id == "punishment":
        boosts = getattr(opponent, "boosts", None) or {}
        positive_stages = sum(
            max(0, int(_number(value))) for value in boosts.values()
        )
        return float(min(200, 60 + 20 * positive_stages)), "target_boosts"

    if move_id in {"eruption", "waterspout", "dragonenergy"}:
        hp_fraction = _known_hp_fraction(active)
        if hp_fraction is None:
            return None, "unknown_user_hp"
        return float(max(1, math.floor(150 * hp_fraction))), "user_hp"

    if move_id in {"flail", "reversal"}:
        hp_fraction = _known_hp_fraction(active)
        if hp_fraction is None:
            return None, "unknown_user_hp"
        if hp_fraction <= 1 / 48:
            power = 200
        elif hp_fraction <= 4 / 48:
            power = 150
        elif hp_fraction <= 9 / 48:
            power = 100
        elif hp_fraction <= 16 / 48:
            power = 80
        elif hp_fraction <= 32 / 48:
            power = 40
        else:
            power = 20
        return float(power), "user_hp"

    if move_id in {"crushgrip", "wringout"}:
        hp_fraction = _known_hp_fraction(opponent)
        if hp_fraction is None:
            return None, "unknown_target_hp"
        return float(max(1, math.floor(120 * hp_fraction) + 1)), "target_hp"

    if move_id == "hardpress":
        hp_fraction = _known_hp_fraction(opponent)
        if hp_fraction is None:
            return None, "unknown_target_hp"
        return float(max(1, math.floor(100 * hp_fraction))), "target_hp"

    if move_id == "facade" and getattr(active, "status", None) is not None:
        return 140.0, "user_status"

    if move_id in {"hex", "infernalparade"} and (
        getattr(opponent, "status", None) is not None
    ):
        return raw_power * 2, "target_status"

    if move_id in {"venoshock", "barbbarrage"} and (
        _enum_name(getattr(opponent, "status", None)) in {"psn", "tox"}
    ):
        return raw_power * 2, "target_poisoned"

    if move_id == "brine":
        hp_fraction = _known_hp_fraction(opponent)
        if hp_fraction is not None and hp_fraction <= 0.5:
            return raw_power * 2, "target_low_hp"

    if move_id == "acrobatics" and not getattr(active, "item", None):
        return 110.0, "no_held_item"

    if move_id == "knockoff":
        item = getattr(opponent, "item", None)
        if item and str(item) != GenData.UNKNOWN_ITEM:
            return raw_power * 1.5, "known_item_assumed_removable"

    if move_id in {"fishiousrend", "boltbeak"}:
        multiplier = 2 if speed_relation == "faster" else 1
        return raw_power * multiplier, "estimated_turn_order"

    if move_id == "payback":
        multiplier = 2 if speed_relation == "slower" else 1
        return raw_power * multiplier, "estimated_turn_order"

    if raw_power > 0 or category not in {"physical", "special"}:
        return raw_power, "fixed"
    return None, "unknown_dynamic"


def _offensive_stats(
    active: Any,
    opponent: Any,
    move: Move,
    *,
    terastallize: bool,
) -> tuple[float, float, str]:
    move_id = str(getattr(move, "id", ""))
    category = _enum_name(getattr(move, "category", None))
    if move_id == "bodypress":
        return (
            _estimated_stat(active, "def"),
            _estimated_stat(opponent, "def"),
            "user_def_vs_target_def",
        )
    if move_id == "foulplay":
        return (
            _estimated_stat(opponent, "atk"),
            _estimated_stat(opponent, "def"),
            "target_atk_vs_target_def",
        )
    if move_id in {"psyshock", "psystrike", "secretsword"}:
        return (
            _estimated_stat(active, "spa"),
            _estimated_stat(opponent, "def"),
            "user_spa_vs_target_def",
        )
    if move_id == "terablast" and terastallize:
        physical_attack = _estimated_stat(active, "atk")
        special_attack = _estimated_stat(active, "spa")
        if physical_attack > special_attack:
            return (
                physical_attack,
                _estimated_stat(opponent, "def"),
                "tera_blast_physical",
            )
        return (
            special_attack,
            _estimated_stat(opponent, "spd"),
            "tera_blast_special",
        )
    if category == "physical":
        return (
            _estimated_stat(active, "atk"),
            _estimated_stat(opponent, "def"),
            "user_atk_vs_target_def",
        )
    if category == "special":
        return (
            _estimated_stat(active, "spa"),
            _estimated_stat(opponent, "spd"),
            "user_spa_vs_target_spd",
        )
    return 0.0, 1.0, "status"


def _estimated_damage_fraction_range(
    active: Any,
    opponent: Any,
    *,
    power: float | None,
    attack_stat: float,
    defense_stat: float,
    stab: float,
    type_multiplier: float,
    expected_hits: float,
) -> tuple[float, float] | None:
    if power is None:
        return None
    if power <= 0 or type_multiplier <= 0:
        return (0.0, 0.0)
    level = _number(getattr(active, "level", 100)) or 100
    target_hp = _estimated_max_hp(opponent)
    base_damage = (
        (((2 * level / 5) + 2) * power * attack_stat / max(defense_stat, 1.0))
        / 50
    ) + 2
    modifier = stab * type_multiplier * expected_hits
    return (
        max(0.0, base_damage * modifier * 0.85 / target_hp),
        max(0.0, base_damage * modifier / target_hp),
    )


def _estimated_ko_probability(
    active: Any,
    opponent: Any,
    *,
    power: float | None,
    attack_stat: float,
    defense_stat: float,
    stab: float,
    type_multiplier: float,
    expected_hits: float,
    accuracy: float,
) -> float | None:
    if power is None:
        return None
    remaining_hp = _known_hp_fraction(opponent)
    if remaining_hp is None:
        return None
    if power <= 0 or type_multiplier <= 0:
        return 0.0
    level = _number(getattr(active, "level", 100)) or 100
    target_hp = _estimated_max_hp(opponent)
    base_damage = (
        (((2 * level / 5) + 2) * power * attack_stat / max(defense_stat, 1.0))
        / 50
    ) + 2
    modifier = stab * type_multiplier * expected_hits
    ko_rolls = sum(
        1
        for roll in (0.85 + index * 0.01 for index in range(16))
        if (base_damage * modifier * roll / target_hp) >= remaining_hp
    )
    return accuracy * ko_rolls / 16


def _estimated_stat(pokemon: Any, stat: str) -> float:
    if pokemon is None:
        return 100.0
    boosts = getattr(pokemon, "boosts", None) or {}
    stage = max(-6, min(6, int(_number(boosts.get(stat, 0)))))
    boost = (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
    observed_stats = getattr(pokemon, "stats", None) or {}
    observed = _positive_number(observed_stats.get(stat))
    if observed is not None:
        return max(1.0, observed * boost)
    base_stats = getattr(pokemon, "base_stats", None) or {}
    base = _number(base_stats.get(stat, 100))
    level = _number(getattr(pokemon, "level", 100)) or 100
    unboosted = ((2 * base + 31) * level / 100) + 5
    return max(1.0, unboosted * boost)


def _estimated_max_hp(pokemon: Any) -> float:
    if pokemon is None:
        return 300.0
    base_stats = getattr(pokemon, "base_stats", None) or {}
    base_hp = _number(base_stats.get("hp", 100))
    if base_hp == 1:
        return 1.0
    level = _number(getattr(pokemon, "level", 100)) or 100
    return max(1.0, ((2 * base_hp + 31) * level / 100) + level + 10)


def _known_hp_fraction(pokemon: Any) -> float | None:
    if pokemon is None:
        return None
    value = getattr(pokemon, "current_hp_fraction", None)
    if value is None:
        return None
    fraction = max(0.0, min(1.0, _number(value)))
    if fraction == 0 and _enum_name(getattr(pokemon, "status", None)) != "fnt":
        return None
    return fraction


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


def _effective_move_type(
    active: Any,
    move: Move,
    *,
    terastallize: bool,
) -> Any:
    move_type = getattr(move, "type", None)
    is_tera = terastallize or bool(
        getattr(active, "is_terastallized", False)
    )
    if str(getattr(move, "id", "")) == "terablast" and is_tera:
        return getattr(active, "tera_type", None) or move_type
    return move_type


def _stab_multiplier(
    active: Any,
    move_type: Any,
    terastallize: bool,
) -> float:
    original_types = (
        getattr(active, "base_types", None)
        or getattr(active, "types", None)
        or []
    )
    original_stab = move_type is not None and move_type in original_types
    tera_type = getattr(active, "tera_type", None)
    is_tera = terastallize or bool(
        getattr(active, "is_terastallized", False)
    )
    if is_tera and move_type is not None and move_type == tera_type:
        return 2.0 if original_stab else 1.5
    return 1.5 if original_stab else 1.0


def _damage_multiplier(target: Any, type_or_move: Any) -> float:
    if target is None or not hasattr(target, "damage_multiplier"):
        return 1.0
    try:
        return max(0.0, _number(target.damage_multiplier(type_or_move)))
    except (AttributeError, KeyError, TypeError, ValueError):
        return 1.0


def _defensive_tera_estimate(
    active: Any,
    opponent: Any,
    battle_gen: int,
) -> dict[str, Any]:
    tera_type = getattr(active, "tera_type", None)
    opponent_types = [
        value
        for value in (getattr(opponent, "types", None) or [])
        if value is not None
    ]
    original_types = [
        value
        for value in (
            getattr(active, "base_types", None)
            or getattr(active, "types", None)
            or []
        )
        if value is not None
    ]
    if tera_type is None or not opponent_types or not original_types:
        return {
            "available": False,
            "reason": "Missing a Tera type or visible combatant types.",
        }

    type_chart = GenData.from_gen(battle_gen).type_chart
    before = max(
        _type_multiplier_against_types(
            attacking_type,
            original_types,
            type_chart,
        )
        for attacking_type in opponent_types
    )
    if _enum_name(tera_type) == "stellar":
        after = before
    else:
        after = max(
            _type_multiplier_against_types(
                attacking_type,
                [tera_type],
                type_chart,
            )
            for attacking_type in opponent_types
        )
    if after < before:
        verdict = "improves"
    elif after > before:
        verdict = "worsens"
    else:
        verdict = "neutral"
    return {
        "available": True,
        "basis": "opponent_visible_stab_types",
        "opponent_visible_types": [
            _enum_name(value) for value in opponent_types
        ],
        "tera_type": _enum_name(tera_type),
        "max_stab_multiplier_before": round(before, 3),
        "max_stab_multiplier_after": round(after, 3),
        "pressure_ratio_after_vs_before": (
            round(after / before, 4) if before > 0 else None
        ),
        "verdict": verdict,
    }


def _type_multiplier_against_types(
    attacking_type: Any,
    defending_types: list[Any],
    type_chart: dict[str, dict[str, float]],
) -> float:
    if not defending_types or not hasattr(attacking_type, "damage_multiplier"):
        return 1.0
    try:
        return max(
            0.0,
            _number(
                attacking_type.damage_multiplier(
                    defending_types[0],
                    defending_types[1] if len(defending_types) > 1 else None,
                    type_chart=type_chart,
                )
            ),
        )
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


def _positive_number(value: Any) -> float | None:
    number = _number(value)
    return number if number > 0 else None


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
