from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from showdown_mind.domain import BattleSnapshot, LegalAction

POLICY_INPUT_FORMATS = ("pruned", "compact", "full")


@dataclass(frozen=True)
class CompiledPolicyInput:
    format_name: str
    payload: dict[str, Any]

    def canonical_json(self) -> str:
        return json.dumps(
            self.payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.canonical_json().encode()).hexdigest()
        return f"sha256:{digest}"

    @property
    def characters(self) -> int:
        return len(self.canonical_json())


def compile_policy_input(
    snapshot: BattleSnapshot,
    format_name: str = "pruned",
) -> CompiledPolicyInput:
    if format_name == "full":
        return CompiledPolicyInput("full-v1", snapshot.to_dict())
    if format_name == "pruned":
        return CompiledPolicyInput("pruned-v1", _pruned_snapshot(snapshot))
    if format_name == "compact":
        return CompiledPolicyInput("compact-v1", _compact_snapshot(snapshot))
    choices = ", ".join(POLICY_INPUT_FORMATS)
    raise ValueError(
        f"Unknown policy input format {format_name!r}; choose one of: {choices}"
    )


def _pruned_snapshot(snapshot: BattleSnapshot) -> dict[str, Any]:
    payload = snapshot.to_dict()
    for key in ("schema_version", "battle_id", "request_id"):
        payload.pop(key)
    for side_key, team_key in (
        ("own_side", "team"),
        ("opponent_side", "revealed_team"),
    ):
        side = payload.get(side_key, {})
        for pokemon in side.get(team_key, []):
            pokemon.pop("name", None)
            pokemon.pop("information_scope", None)
    return {
        "schema": "pruned-v1",
        **_prune(payload),
    }


def _compact_snapshot(snapshot: BattleSnapshot) -> dict[str, Any]:
    return {
        "schema": "compact-v1",
        "turn": snapshot.turn,
        "format": snapshot.battle_format,
        "own": _compact_side(snapshot.own_side, team_key="team"),
        "opponent": _compact_side(
            snapshot.opponent_side,
            team_key="revealed_team",
        ),
        "field": _prune(snapshot.field),
        "resources": _prune(snapshot.resources),
        "legal_actions": [_compact_action(action) for action in snapshot.legal_actions],
    }


def _compact_side(side: dict[str, Any], *, team_key: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "active": side.get("active"),
        "team": [_compact_pokemon(pokemon) for pokemon in side.get(team_key, [])],
        "conditions": side.get("side_conditions", {}),
    }
    if "used_tera" in side:
        value["used_tera"] = side["used_tera"]
    return _prune(value)


def _compact_pokemon(pokemon: dict[str, Any]) -> dict[str, Any]:
    return _prune(
        {
            "species": pokemon.get("species"),
            "hp": pokemon.get("hp_fraction"),
            "fainted": pokemon.get("fainted"),
            "status": pokemon.get("status"),
            "types": pokemon.get("types"),
            "boosts": pokemon.get("boosts"),
            "item": pokemon.get("item"),
            "ability": pokemon.get("ability"),
            "moves": pokemon.get("moves"),
            "tera_type": pokemon.get("tera_type"),
        }
    )


def _compact_action(action: LegalAction) -> dict[str, Any]:
    return _prune(
        {
            "action_id": action.action_id,
            "kind": action.kind,
            **action.details,
        }
    )


def _prune(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: compact_value
            for key, item in value.items()
            if not _is_empty(compact_value := _prune(item))
        }
    if isinstance(value, list):
        return [
            compact_value
            for item in value
            if not _is_empty(compact_value := _prune(item))
        ]
    return value


def _is_empty(value: Any) -> bool:
    return value is None or value is False or value == "" or value == [] or value == {}
