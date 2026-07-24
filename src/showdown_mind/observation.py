from __future__ import annotations

from typing import Any

from showdown_mind.domain import BattleSnapshot, LegalAction


class BattleSnapshotBuilder:
    """Build a whitelist-only player view from a poke-env Battle."""

    schema_version = "1.0"

    def build(
        self,
        battle: Any,
        legal_actions: tuple[LegalAction, ...],
    ) -> BattleSnapshot:
        own_team = [
            self._serialize_pokemon(pokemon, own=True)
            for pokemon in battle.team.values()
        ]
        opponent_team = [
            self._serialize_pokemon(pokemon, own=False)
            for pokemon in battle.opponent_team.values()
            if getattr(pokemon, "revealed", False)
        ]

        own_team.sort(key=lambda value: (value["species"], value["name"] or ""))
        opponent_team.sort(key=lambda value: (value["species"], value["name"] or ""))

        request_id = int((getattr(battle, "last_request", {}) or {}).get("rqid", 0))
        return BattleSnapshot(
            schema_version=self.schema_version,
            battle_id=str(battle.battle_tag),
            request_id=request_id,
            turn=int(battle.turn),
            battle_format=str(battle.format or ""),
            own_side={
                "active": self._species(battle.active_pokemon),
                "team": own_team,
                "side_conditions": self._serialize_counter_map(battle.side_conditions),
            },
            opponent_side={
                "active": self._species(battle.opponent_active_pokemon),
                "revealed_team": opponent_team,
                "side_conditions": self._serialize_counter_map(
                    battle.opponent_side_conditions
                ),
                "used_tera": bool(battle.opponent_used_tera),
            },
            field={
                "weather": self._serialize_counter_map(battle.weather),
                "fields": self._serialize_counter_map(battle.fields),
            },
            resources={
                "can_tera": bool(battle.can_tera),
                "used_tera": bool(battle.used_tera),
                "force_switch": bool(battle.force_switch),
                "trapped": bool(battle.trapped),
            },
            legal_actions=legal_actions,
        )

    def _serialize_pokemon(self, pokemon: Any, *, own: bool) -> dict[str, Any]:
        moves = sorted(
            {
                str(getattr(move, "id", move))
                for move in (getattr(pokemon, "moves", {}) or {}).values()
            }
        )
        item = self._known_text(getattr(pokemon, "item", None))
        ability = self._known_text(getattr(pokemon, "ability", None))
        tera_type = self._enum_name(getattr(pokemon, "tera_type", None))

        return {
            "species": str(getattr(pokemon, "species", "")),
            "name": self._known_text(getattr(pokemon, "name", None)),
            "active": bool(getattr(pokemon, "active", False)),
            "fainted": bool(getattr(pokemon, "fainted", False)),
            "hp_fraction": round(
                float(getattr(pokemon, "current_hp_fraction", 0.0)),
                4,
            ),
            "status": self._enum_name(getattr(pokemon, "status", None)),
            "types": [
                self._enum_name(value)
                for value in (getattr(pokemon, "types", []) or [])
            ],
            "boosts": {
                str(key): int(value)
                for key, value in (getattr(pokemon, "boosts", {}) or {}).items()
                if value
            },
            "item": item,
            "ability": ability,
            "moves": moves,
            "tera_type": tera_type,
            "information_scope": "own" if own else "revealed",
        }

    @staticmethod
    def _serialize_counter_map(values: dict[Any, Any]) -> dict[str, int]:
        return {
            BattleSnapshotBuilder._enum_name(key) or str(key): int(value)
            for key, value in sorted(
                values.items(),
                key=lambda item: str(item[0]),
            )
        }

    @staticmethod
    def _enum_name(value: Any) -> str | None:
        if value is None:
            return None
        return str(getattr(value, "name", value)).lower()

    @staticmethod
    def _known_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        if not text or text.startswith("unknown"):
            return None
        return text

    @staticmethod
    def _species(pokemon: Any) -> str | None:
        if pokemon is None:
            return None
        return str(getattr(pokemon, "species", "")) or None
