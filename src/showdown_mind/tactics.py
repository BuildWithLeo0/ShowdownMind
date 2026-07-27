from __future__ import annotations

import math
from typing import Any

from poke_env.battle import Move, Pokemon, PokemonType
from poke_env.data import GenData
from poke_env.player.battle_order import SingleBattleOrder

from showdown_mind.actions import ActionCatalog

TACTICAL_ANALYSIS_SCHEMA = "tactical-analysis-v2.1"
TACTICAL_MODEL_VIEW = "model-compact-v1"

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

    def analyze(
        self,
        battle: Any,
        catalog: ActionCatalog,
        *,
        opponent_candidate_move_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        active = getattr(battle, "active_pokemon", None)
        opponent = getattr(battle, "opponent_active_pokemon", None)
        own_conditions = getattr(battle, "side_conditions", None) or {}
        opponent_conditions = (
            getattr(battle, "opponent_side_conditions", None) or {}
        )
        fields = getattr(battle, "fields", None) or {}
        weather = getattr(battle, "weather", None) or {}
        trick_room = _has_named_effect(
            fields,
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
                        fields,
                        weather,
                        battle_gen,
                        opponent_candidate_move_ids,
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
                        fields,
                        weather,
                        trick_room,
                        battle_gen,
                        opponent_candidate_move_ids,
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
        counterplay_estimates = [
            (
                action["action_id"],
                float(
                    action["counterplay"][
                        "estimated_counter_ko_probability"
                    ]
                ),
            )
            for action in actions
            if action.get("counterplay", {}).get(
                "estimated_counter_ko_probability"
            )
            is not None
        ]
        lowest_counter_ko_probability = min(
            (probability for _, probability in counterplay_estimates),
            default=None,
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
                "weather": [
                    _enum_name(value) for value in weather
                ],
                "terrain": [
                    _enum_name(value)
                    for value in fields
                    if str(_enum_name(value) or "").endswith("_terrain")
                ],
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
            "safest_action_ids": [
                action_id
                for action_id, probability in counterplay_estimates
                if lowest_counter_ko_probability is not None
                and probability == lowest_counter_ko_probability
            ],
            "lowest_counter_ko_probability": (
                round(lowest_counter_ko_probability, 4)
                if lowest_counter_ko_probability is not None
                else None
            ),
            "best_switch_action_ids": [
                action["action_id"]
                for action in switches
                if best_switch_score is not None
                and float(action["matchup_score"]) == best_switch_score
            ],
            "actions": actions,
            "opponent_candidate_move_ids": list(
                opponent_candidate_move_ids[:8]
            ),
            "limitations": (
                "Damage ranges and KO probabilities are approximations based on "
                "public species, level, types, boosts, move data, and HP. They "
                "use exact visible stats when available; missing stats assume 31 "
                "IVs, zero EVs, and a neutral nature. Hidden items, abilities, "
                "critical hits, and unmodeled special effects are omitted. "
                "Weather, terrain, burn, screens, and entry hazards are included "
                "when visible. Supported variable-power moves are estimated; "
                "unknown dynamic moves remain unranked. Counterplay uses revealed "
                "opponent moves plus any supplied public Random Battle move "
                "candidates; candidates are hypotheses, not hidden facts. "
                "Defensive Tera compares only the "
                "opponent's currently visible STAB types, not unrevealed moves."
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
        fields: dict[Any, Any],
        weather: dict[Any, Any],
        battle_gen: int,
        opponent_candidate_move_ids: tuple[str, ...],
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
        damage_modifier, modifier_sources = _battle_damage_modifier(
            attacker=active,
            defender=opponent,
            move=move,
            move_type=move_type,
            category=effective_category,
            defender_conditions=opponent_conditions,
            fields=fields,
            weather=weather,
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
            damage_modifier=damage_modifier,
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
            damage_modifier=damage_modifier,
        )
        damage_index = None
        if damage_range is not None:
            damage_index = accuracy * sum(damage_range) / 2
        priority = int(_number(getattr(move, "priority", 0)))
        self_hp_effect = _self_hp_effect(
            active,
            opponent,
            move,
            damage_range,
            accuracy,
        )
        counterplay = _counterplay_estimate(
            defender=active,
            attacker=opponent,
            own_move=move,
            own_priority=priority,
            outgoing_ko_probability=ko_probability,
            speed_relation=speed_relation,
            defender_conditions=own_conditions,
            attacker_conditions=opponent_conditions,
            fields=fields,
            weather=weather,
            battle_gen=battle_gen,
            remaining_hp_fraction=self_hp_effect["post_action_hp_fraction"],
            terastallize=bool(order.terastallize),
            candidate_move_ids=opponent_candidate_move_ids,
        )
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
            "battle_modifier": round(damage_modifier, 4),
            "modifier_sources": modifier_sources,
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
            "self_hp_effect": self_hp_effect,
            "counterplay": counterplay,
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
        fields: dict[Any, Any],
        weather: dict[Any, Any],
        trick_room: bool,
        battle_gen: int,
        opponent_candidate_move_ids: tuple[str, ...],
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
        entry_hazards = _entry_hazard_estimate(
            candidate,
            own_conditions,
            fields,
            battle_gen,
        )
        counterplay = _counterplay_estimate(
            defender=candidate,
            attacker=opponent,
            own_move=None,
            own_priority=None,
            outgoing_ko_probability=None,
            speed_relation=speed_relation,
            defender_conditions=own_conditions,
            attacker_conditions=opponent_conditions,
            fields=fields,
            weather=weather,
            battle_gen=battle_gen,
            remaining_hp_fraction=entry_hazards["post_entry_hp_fraction"],
            terastallize=False,
            candidate_move_ids=opponent_candidate_move_ids,
        )
        counter_ko_probability = _number(
            counterplay.get("estimated_counter_ko_probability")
        )
        entry_damage = _number(entry_hazards.get("damage_fraction"))
        matchup_score = (
            offensive
            - defensive
            + speed_adjustment
            + 0.4 * (hp_fraction - opponent_hp)
            - 2 * counter_ko_probability
            - entry_damage
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
            "entry_hazards": entry_hazards,
            "counterplay": counterplay,
        }


def compact_tactical_analysis_for_model(
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Keep decision-relevant fields while the audit log retains full details."""
    compact = {
        "schema": analysis.get("schema"),
        "view": TACTICAL_MODEL_VIEW,
        "speed_relation": analysis.get("speed_relation"),
        "best_damage_action_ids": analysis.get(
            "best_damage_action_ids",
            [],
        ),
        "best_ko_action_ids": analysis.get("best_ko_action_ids", []),
        "best_ko_probability": analysis.get("best_ko_probability"),
        "safest_action_ids": analysis.get("safest_action_ids", []),
        "lowest_counter_ko_probability": analysis.get(
            "lowest_counter_ko_probability"
        ),
        "best_switch_action_ids": analysis.get(
            "best_switch_action_ids",
            [],
        ),
        "opponent_move_information": {
            "public_candidate_move_ids": analysis.get(
                "opponent_candidate_move_ids",
                [],
            ),
            "candidate_moves_are_hypotheses": bool(
                analysis.get("opponent_candidate_move_ids")
            ),
        },
        "actions": [
            _compact_tactical_action(action)
            for action in analysis.get("actions", [])
        ],
    }
    return compact


def _compact_tactical_action(action: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "action_id": action.get("action_id"),
        "kind": action.get("kind"),
    }
    if action.get("kind") == "move":
        _copy_present(
            compact,
            action,
            (
                "move_id",
                "type_multiplier",
                "estimated_damage_fraction_range",
                "estimated_ko_probability",
                "relative_damage",
                "move_order",
            ),
        )
        if action.get("power_source") != "fixed":
            _copy_present(
                compact,
                action,
                ("effective_power", "power_source"),
            )
        if _number(action.get("stab_multiplier")) > 1:
            compact["stab_multiplier"] = action["stab_multiplier"]
        if action.get("terastallize"):
            compact["terastallize"] = True
        if action.get("role_tags"):
            compact["role_tags"] = action["role_tags"]
        if action.get("modifier_sources"):
            compact["modifier_sources"] = action["modifier_sources"]
        self_hp_effect = action.get("self_hp_effect") or {}
        if (
            self_hp_effect.get("expected_net_change_fraction")
            or self_hp_effect.get("self_ko_risk")
        ):
            compact["self_hp_effect"] = {
                key: self_hp_effect[key]
                for key in (
                    "expected_net_change_fraction",
                    "post_action_hp_fraction",
                    "self_ko_risk",
                )
                if key in self_hp_effect
            }
        defensive_tera = action.get("defensive_tera") or {}
        if defensive_tera:
            compact["defensive_tera"] = {
                key: defensive_tera[key]
                for key in (
                    "available",
                    "tera_type",
                    "max_stab_multiplier_before",
                    "max_stab_multiplier_after",
                    "verdict",
                )
                if key in defensive_tera
            }
    elif action.get("kind") == "switch":
        _copy_present(
            compact,
            action,
            (
                "species",
                "offensive_type_multiplier",
                "defensive_weakness_multiplier",
                "speed_relation",
                "matchup_score",
            ),
        )
        entry_hazards = action.get("entry_hazards") or {}
        if entry_hazards.get("damage_fraction") or entry_hazards.get("effects"):
            compact["entry_hazards"] = {
                key: entry_hazards[key]
                for key in (
                    "damage_fraction",
                    "post_entry_hp_fraction",
                    "effects",
                )
                if key in entry_hazards
            }
    else:
        _copy_present(compact, action, ("note",))

    if action.get("counterplay"):
        compact["counterplay"] = _compact_counterplay(
            action["counterplay"]
        )
    return compact


def _compact_counterplay(counterplay: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "risk",
        "worst_move_id",
        "incoming_damage_fraction_range",
        "estimated_counter_ko_probability",
        "player_acts_before_reply",
        "protect_success_probability",
        "unscored_move_ids",
    )
    compact = {
        key: counterplay[key]
        for key in keys
        if key in counterplay
        and counterplay[key] not in (None, [], {})
    }
    if counterplay.get("available") is False:
        compact["available"] = False
    return compact


def _copy_present(
    target: dict[str, Any],
    source: dict[str, Any],
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if key in source and source[key] is not None:
            target[key] = source[key]


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


def _battle_damage_modifier(
    *,
    attacker: Any,
    defender: Any,
    move: Move,
    move_type: Any,
    category: str | None,
    defender_conditions: dict[Any, Any],
    fields: dict[Any, Any],
    weather: dict[Any, Any],
    defender_types: list[Any] | None = None,
) -> tuple[float, list[str]]:
    multiplier = 1.0
    sources: list[str] = []
    move_id = str(getattr(move, "id", ""))
    move_type_name = _enum_name(move_type)
    attacker_grounded = _is_grounded(attacker, fields)
    defender_grounded = _is_grounded(
        defender,
        fields,
        types=defender_types,
    )

    if _has_any_named_effect(weather, {"raindance", "primordialsea"}):
        if move_type_name == "water":
            multiplier *= 1.5
            sources.append("rain_water_boost")
        elif move_type_name == "fire":
            multiplier *= 0.0 if _has_named_effect(
                weather, "primordialsea"
            ) else 0.5
            sources.append("rain_fire_reduction")
    elif _has_any_named_effect(weather, {"sunnyday", "desolateland"}):
        if move_type_name == "fire":
            multiplier *= 1.5
            sources.append("sun_fire_boost")
        elif move_type_name == "water":
            multiplier *= 0.0 if _has_named_effect(
                weather, "desolateland"
            ) else 0.5
            sources.append("sun_water_reduction")

    if (
        category == "special"
        and _has_named_effect(weather, "sandstorm")
        and _has_type(defender_types or getattr(defender, "types", None), "rock")
    ):
        multiplier *= 2 / 3
        sources.append("sandstorm_rock_spd")
    if (
        category == "physical"
        and _has_any_named_effect(weather, {"snow", "snowscape"})
        and _has_type(defender_types or getattr(defender, "types", None), "ice")
    ):
        multiplier *= 2 / 3
        sources.append("snow_ice_def")

    if (
        move_type_name == "electric"
        and attacker_grounded
        and _has_named_effect(fields, "electric_terrain")
    ):
        multiplier *= 1.3
        sources.append("electric_terrain")
    elif (
        move_type_name == "grass"
        and attacker_grounded
        and _has_named_effect(fields, "grassy_terrain")
    ):
        multiplier *= 1.3
        sources.append("grassy_terrain")
    elif (
        move_type_name == "psychic"
        and attacker_grounded
        and _has_named_effect(fields, "psychic_terrain")
    ):
        multiplier *= 1.3
        sources.append("psychic_terrain")

    if (
        move_type_name == "dragon"
        and defender_grounded
        and _has_named_effect(fields, "misty_terrain")
    ):
        multiplier *= 0.5
        sources.append("misty_terrain")
    if (
        move_id in {"earthquake", "bulldoze", "magnitude"}
        and defender_grounded
        and _has_named_effect(fields, "grassy_terrain")
    ):
        multiplier *= 0.5
        sources.append("grassy_terrain_ground_move_reduction")
    if (
        int(_number(getattr(move, "priority", 0))) > 0
        and defender_grounded
        and _has_named_effect(fields, "psychic_terrain")
    ):
        multiplier = 0.0
        sources.append("psychic_terrain_priority_block")

    attacker_ability = str(getattr(attacker, "ability", "") or "")
    if attacker_ability != "infiltrator":
        if _has_named_effect(defender_conditions, "aurora_veil"):
            multiplier *= 0.5
            sources.append("aurora_veil")
        elif category == "physical" and _has_named_effect(
            defender_conditions,
            "reflect",
        ):
            multiplier *= 0.5
            sources.append("reflect")
        elif category == "special" and _has_named_effect(
            defender_conditions,
            "light_screen",
        ):
            multiplier *= 0.5
            sources.append("light_screen")

    if (
        category == "physical"
        and _enum_name(getattr(attacker, "status", None)) == "brn"
        and move_id != "facade"
        and attacker_ability != "guts"
    ):
        multiplier *= 0.5
        sources.append("burn")

    return multiplier, sources


def _self_hp_effect(
    active: Any,
    opponent: Any,
    move: Move,
    damage_range: tuple[float, float] | None,
    accuracy: float,
) -> dict[str, Any]:
    current_hp = _known_hp_fraction(active)
    if current_hp is None:
        return {
            "available": False,
            "post_action_hp_fraction": None,
        }

    midpoint_damage = (
        sum(damage_range) / 2 if damage_range is not None else 0.0
    )
    target_hp = _known_hp_fraction(opponent)
    inflicted_damage = (
        min(midpoint_damage, target_hp)
        if target_hp is not None
        else midpoint_damage
    )
    drain = max(0.0, _number(getattr(move, "drain", 0.0)))
    recoil = max(0.0, _number(getattr(move, "recoil", 0.0)))
    direct_heal = max(0.0, _number(getattr(move, "heal", 0.0)))
    healing = direct_heal + inflicted_damage * drain * accuracy
    recoil_damage = inflicted_damage * recoil * accuracy
    post_action_hp = max(
        0.0,
        min(1.0, current_hp + healing - recoil_damage),
    )
    return {
        "available": True,
        "current_hp_fraction": round(current_hp, 4),
        "expected_healing_fraction": round(healing, 4),
        "expected_recoil_fraction": round(recoil_damage, 4),
        "expected_net_change_fraction": round(
            post_action_hp - current_hp,
            4,
        ),
        "post_action_hp_fraction": round(post_action_hp, 4),
        "self_ko_risk": current_hp > 0 and post_action_hp <= 0,
    }


def _counterplay_estimate(
    *,
    defender: Any,
    attacker: Any,
    own_move: Move | None,
    own_priority: int | None,
    outgoing_ko_probability: float | None,
    speed_relation: str,
    defender_conditions: dict[Any, Any],
    attacker_conditions: dict[Any, Any],
    fields: dict[Any, Any],
    weather: dict[Any, Any],
    battle_gen: int,
    remaining_hp_fraction: float | None,
    terastallize: bool,
    candidate_move_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    revealed_moves = [
        move
        for move in (getattr(attacker, "moves", None) or {}).values()
        if isinstance(move, Move)
    ]
    revealed_ids = {
        str(getattr(move, "id", "")) for move in revealed_moves
    }
    candidate_moves: list[Move] = []
    for move_id in candidate_move_ids[:8]:
        if move_id in revealed_ids:
            continue
        try:
            candidate_moves.append(Move(move_id, gen=battle_gen))
        except (KeyError, ValueError):
            continue
    considered_moves = revealed_moves + candidate_moves
    if remaining_hp_fraction is not None and remaining_hp_fraction <= 0:
        return {
            "available": True,
            "revealed_moves_considered": len(revealed_moves),
            "candidate_moves_considered": len(candidate_moves),
            "estimated_counter_ko_probability": 1.0,
            "survival_probability": 0.0,
            "risk": "self_ko" if own_move is not None else "entry_hazard_ko",
        }
    if not considered_moves:
        return {
            "available": False,
            "revealed_moves_considered": 0,
            "candidate_moves_considered": 0,
            "unscored_move_ids": [],
            "risk": "unknown_no_candidate_or_revealed_moves",
        }

    type_chart = GenData.from_gen(battle_gen).type_chart
    defender_types = _defender_types_after_action(
        defender,
        terastallize=terastallize,
    )
    attacker_speed_relation = _reverse_speed_relation(speed_relation)
    protected = bool(
        own_move is not None
        and getattr(own_move, "is_protect_move", False)
    )
    protect_success_probability = (
        1
        / (
            3
            ** max(
                0,
                int(_number(getattr(defender, "protect_counter", 0))),
            )
        )
        if protected
        else 0.0
    )
    estimates: list[dict[str, Any]] = []
    unscored_move_ids: list[str] = []

    for move in considered_moves:
        move_id = str(getattr(move, "id", ""))
        category = _enum_name(getattr(move, "category", None))
        effective_power, power_source = _effective_move_power(
            move,
            attacker,
            defender,
            own_conditions=attacker_conditions,
            opponent_conditions=defender_conditions,
            speed_relation=attacker_speed_relation,
        )
        if effective_power is None or effective_power <= 0:
            unscored_move_ids.append(move_id)
            continue
        move_type = _effective_move_type(
            attacker,
            move,
            terastallize=False,
        )
        type_multiplier = _type_multiplier_against_types(
            move_type,
            defender_types,
            type_chart,
        )
        stab = _stab_multiplier(attacker, move_type, False)
        attack_stat, defense_stat, stat_source = _offensive_stats(
            attacker,
            defender,
            move,
            terastallize=bool(
                getattr(attacker, "is_terastallized", False)
            ),
        )
        effective_category = (
            "physical"
            if stat_source == "tera_blast_physical"
            else "special"
            if stat_source == "tera_blast_special"
            else category
        )
        battle_modifier, modifier_sources = _battle_damage_modifier(
            attacker=attacker,
            defender=defender,
            move=move,
            move_type=move_type,
            category=effective_category,
            defender_conditions=defender_conditions,
            fields=fields,
            weather=weather,
            defender_types=defender_types,
        )
        expected_hits = max(
            1.0,
            _number(getattr(move, "expected_hits", 1.0)),
        )
        damage_range = _estimated_damage_fraction_range(
            attacker,
            defender,
            power=effective_power,
            attack_stat=attack_stat,
            defense_stat=defense_stat,
            stab=stab,
            type_multiplier=type_multiplier,
            expected_hits=expected_hits,
            damage_modifier=battle_modifier,
        )
        raw_ko_probability = _estimated_ko_probability(
            attacker,
            defender,
            power=effective_power,
            attack_stat=attack_stat,
            defense_stat=defense_stat,
            stab=stab,
            type_multiplier=type_multiplier,
            expected_hits=expected_hits,
            accuracy=_accuracy(getattr(move, "accuracy", 1.0)),
            damage_modifier=battle_modifier,
            remaining_hp_fraction=remaining_hp_fraction,
        )
        if raw_ko_probability is None:
            unscored_move_ids.append(move_id)
            continue
        opponent_priority = int(_number(getattr(move, "priority", 0)))
        acts_before_reply = (
            True
            if own_priority is None
            else _acts_before(
                own_priority,
                opponent_priority,
                speed_relation,
            )
        )
        counter_ko_probability = raw_ko_probability
        if protected and not bool(getattr(move, "breaks_protect", False)):
            counter_ko_probability *= 1 - protect_success_probability
        elif acts_before_reply is True and outgoing_ko_probability is not None:
            counter_ko_probability *= 1 - outgoing_ko_probability
        estimates.append(
            {
                "move_id": move_id,
                "power_source": power_source,
                "estimated_damage_fraction_range": (
                    [round(value, 4) for value in damage_range]
                    if damage_range is not None
                    else None
                ),
                "raw_ko_probability": round(raw_ko_probability, 4),
                "estimated_counter_ko_probability": round(
                    counter_ko_probability,
                    4,
                ),
                "player_acts_before_reply": acts_before_reply,
                "type_multiplier": round(type_multiplier, 3),
                "battle_modifier": round(battle_modifier, 4),
                "modifier_sources": modifier_sources,
            }
        )

    if not estimates:
        return {
            "available": False,
            "revealed_moves_considered": len(revealed_moves),
            "candidate_moves_considered": len(candidate_moves),
            "unscored_move_ids": sorted(set(unscored_move_ids)),
            "risk": "unknown_no_scorable_candidate_or_revealed_damage",
        }

    worst = max(
        estimates,
        key=lambda estimate: (
            float(estimate["estimated_counter_ko_probability"]),
            (
                float(estimate["estimated_damage_fraction_range"][1])
                if estimate["estimated_damage_fraction_range"] is not None
                else 0.0
            ),
        ),
    )
    counter_ko_probability = float(
        worst["estimated_counter_ko_probability"]
    )
    if counter_ko_probability >= 0.5:
        risk = "likely_counter_ko"
    elif counter_ko_probability > 0:
        risk = "possible_counter_ko"
    else:
        risk = "survives_known_reply"
    return {
        "available": True,
        "basis": (
            "revealed_and_public_prior_moves"
            if candidate_moves
            else "revealed_opponent_moves"
        ),
        "revealed_moves_considered": len(revealed_moves),
        "candidate_moves_considered": len(candidate_moves),
        "scored_moves": len(estimates),
        "unscored_move_ids": sorted(set(unscored_move_ids)),
        "worst_move_id": worst["move_id"],
        "incoming_damage_fraction_range": worst[
            "estimated_damage_fraction_range"
        ],
        "raw_incoming_ko_probability": worst["raw_ko_probability"],
        "estimated_counter_ko_probability": round(
            counter_ko_probability,
            4,
        ),
        "survival_probability": round(1 - counter_ko_probability, 4),
        "player_acts_before_reply": worst["player_acts_before_reply"],
        "type_multiplier": worst["type_multiplier"],
        "battle_modifier": worst["battle_modifier"],
        "modifier_sources": worst["modifier_sources"],
        "protect_success_probability": (
            round(protect_success_probability, 4)
            if protected
            else None
        ),
        "risk": risk,
    }


def _entry_hazard_estimate(
    candidate: Any,
    side_conditions: dict[Any, Any],
    fields: dict[Any, Any],
    battle_gen: int,
) -> dict[str, Any]:
    current_hp = _known_hp_fraction(candidate)
    item = str(getattr(candidate, "item", "") or "")
    ability = str(getattr(candidate, "ability", "") or "")
    if item == "heavydutyboots":
        return {
            "damage_fraction": 0.0,
            "post_entry_hp_fraction": current_hp,
            "effects": ["heavy_duty_boots"],
        }

    type_chart = GenData.from_gen(battle_gen).type_chart
    candidate_types = _defender_types_after_action(
        candidate,
        terastallize=False,
    )
    grounded = _is_grounded(candidate, fields, types=candidate_types)
    damage = 0.0
    effects: list[str] = []

    if _has_named_effect(side_conditions, "stealth_rock"):
        rock_multiplier = _type_multiplier_against_types(
            PokemonType.ROCK,
            candidate_types,
            type_chart,
        )
        damage += rock_multiplier / 8
        effects.append("stealth_rock")
    if _has_named_effect(side_conditions, "g_max_steelsurge"):
        steel_multiplier = _type_multiplier_against_types(
            PokemonType.STEEL,
            candidate_types,
            type_chart,
        )
        damage += steel_multiplier / 8
        effects.append("g_max_steelsurge")

    spikes_layers = _condition_level(side_conditions, "spikes")
    if grounded and spikes_layers:
        damage += {1: 1 / 8, 2: 1 / 6, 3: 1 / 4}[
            min(3, spikes_layers)
        ]
        effects.append(f"spikes_{min(3, spikes_layers)}")
    if ability == "magicguard":
        damage = 0.0
        effects.append("magic_guard")

    toxic_layers = _condition_level(side_conditions, "toxic_spikes")
    if grounded and toxic_layers:
        if _has_type(candidate_types, "poison"):
            effects.append("absorbs_toxic_spikes")
        elif not _has_type(candidate_types, "steel"):
            effects.append(
                "toxic_poison_risk" if toxic_layers >= 2 else "poison_risk"
            )
    if grounded and _has_named_effect(side_conditions, "sticky_web"):
        effects.append("sticky_web_speed_drop")

    post_entry_hp = (
        max(0.0, current_hp - damage)
        if current_hp is not None
        else None
    )
    return {
        "damage_fraction": round(damage, 4),
        "post_entry_hp_fraction": (
            round(post_entry_hp, 4)
            if post_entry_hp is not None
            else None
        ),
        "grounded": grounded,
        "effects": effects,
    }


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
    damage_modifier: float = 1.0,
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
    modifier = stab * type_multiplier * expected_hits * damage_modifier
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
    damage_modifier: float = 1.0,
    remaining_hp_fraction: float | None = None,
) -> float | None:
    if power is None:
        return None
    remaining_hp = (
        remaining_hp_fraction
        if remaining_hp_fraction is not None
        else _known_hp_fraction(opponent)
    )
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
    modifier = stab * type_multiplier * expected_hits * damage_modifier
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


def _defender_types_after_action(
    pokemon: Any,
    *,
    terastallize: bool,
) -> list[Any]:
    current_types = [
        value
        for value in (getattr(pokemon, "types", None) or [])
        if value is not None
    ]
    tera_type = getattr(pokemon, "tera_type", None)
    is_tera = terastallize or bool(
        getattr(pokemon, "is_terastallized", False)
    )
    if not is_tera or tera_type is None:
        return current_types
    if _enum_name(tera_type) == "stellar":
        return [
            value
            for value in (
                getattr(pokemon, "base_types", None) or current_types
            )
            if value is not None
        ]
    return [tera_type]


def _is_grounded(
    pokemon: Any,
    fields: dict[Any, Any],
    *,
    types: list[Any] | None = None,
) -> bool:
    if pokemon is None:
        return True
    if _has_named_effect(fields, "gravity"):
        return True
    item = str(getattr(pokemon, "item", "") or "")
    if item == "ironball":
        return True
    effects = getattr(pokemon, "effects", None) or {}
    if _has_any_named_effect(effects, {"magnetrise", "telekinesis"}):
        return False
    ability = str(getattr(pokemon, "ability", "") or "")
    if ability == "levitate" or item == "airballoon":
        return False
    return not _has_type(
        types or getattr(pokemon, "types", None),
        "flying",
    )


def _acts_before(
    own_priority: int,
    opponent_priority: int,
    speed_relation: str,
) -> bool | None:
    if own_priority > opponent_priority:
        return True
    if own_priority < opponent_priority:
        return False
    if speed_relation == "faster":
        return True
    if speed_relation == "slower":
        return False
    return None


def _reverse_speed_relation(speed_relation: str) -> str:
    return {
        "faster": "slower",
        "slower": "faster",
        "tie": "tie",
        "unknown": "unknown",
    }[speed_relation]


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


def _has_any_named_effect(
    values: dict[Any, Any],
    names: set[str],
) -> bool:
    return any(_enum_name(value) in names for value in values)


def _has_type(types: Any, name: str) -> bool:
    return any(
        _enum_name(value) == name
        for value in (types or [])
        if value is not None
    )


def _condition_level(values: dict[Any, Any], name: str) -> int:
    return max(
        (
            max(1, int(_number(level)))
            for condition, level in values.items()
            if _enum_name(condition) == name
        ),
        default=0,
    )


def _species(pokemon: Any) -> str | None:
    if pokemon is None:
        return None
    value = str(getattr(pokemon, "species", ""))
    return value or None
