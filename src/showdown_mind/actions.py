from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from poke_env.battle import Move, Pokemon
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder

from showdown_mind.domain import LegalAction


class UnknownActionError(ValueError):
    """Raised when a policy chooses an action outside the current catalog."""


@dataclass(frozen=True)
class CatalogEntry:
    action: LegalAction
    order: BattleOrder


class ActionCatalog:
    """A safe, model-facing view over poke-env battle orders."""

    def __init__(self, entries: tuple[CatalogEntry, ...]):
        action_ids = [entry.action.action_id for entry in entries]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Action catalog contains duplicate action IDs")
        self._entries = entries
        self._orders = {entry.action.action_id: entry.order for entry in entries}

    @classmethod
    def from_battle(cls, battle: Any) -> ActionCatalog:
        entries: list[CatalogEntry] = []
        occurrences: dict[str, int] = {}
        for order in battle.valid_orders:
            action = _describe_order(order)
            occurrences[action.action_id] = occurrences.get(action.action_id, 0) + 1
            if occurrences[action.action_id] > 1:
                action = LegalAction(
                    action_id=f"{action.action_id}:{occurrences[action.action_id]}",
                    kind=action.kind,
                    label=action.label,
                    details=action.details,
                )
            entries.append(CatalogEntry(action=action, order=order))
        return cls(tuple(sorted(entries, key=lambda entry: entry.action.action_id)))

    @property
    def actions(self) -> tuple[LegalAction, ...]:
        return tuple(entry.action for entry in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def contains(self, action_id: str) -> bool:
        return action_id in self._orders

    def resolve(self, action_id: str) -> BattleOrder:
        try:
            return self._orders[action_id]
        except KeyError as exc:
            valid = ", ".join(self._orders)
            raise UnknownActionError(
                f"Unknown action {action_id!r}; valid actions: {valid}"
            ) from exc


def _describe_order(order: BattleOrder) -> LegalAction:
    if not isinstance(order, SingleBattleOrder):
        return LegalAction("other:order", "other", "Submit valid battle order")

    value = order.order
    if isinstance(value, Move):
        modifiers: list[str] = []
        id_modifiers: list[str] = []
        if order.terastallize:
            modifiers.append("Terastallize")
            id_modifiers.append("tera")
        if order.mega:
            modifiers.append("Mega Evolve")
            id_modifiers.append("mega")
        if order.z_move:
            modifiers.append("Z-Move")
            id_modifiers.append("zmove")
        if order.dynamax:
            modifiers.append("Dynamax")
            id_modifiers.append("dynamax")

        suffix = f" + {' + '.join(modifiers)}" if modifiers else ""
        action_id = f"move:{value.id}"
        if id_modifiers:
            action_id = f"{action_id}:{':'.join(id_modifiers)}"
        if order.move_target:
            action_id = f"{action_id}:target:{order.move_target}"
        return LegalAction(
            action_id=action_id,
            kind="move",
            label=f"Use {value.id}{suffix}",
            details={
                "move_id": value.id,
                "type": _enum_name(value.type),
                "category": _enum_name(value.category),
                "base_power": value.base_power,
                "accuracy": value.accuracy,
                "terastallize": order.terastallize,
            },
        )

    if isinstance(value, Pokemon):
        return LegalAction(
            action_id=f"switch:{_slug(value.species)}",
            kind="switch",
            label=f"Switch to {value.species}",
            details={
                "species": value.species,
                "hp_fraction": round(value.current_hp_fraction, 4),
                "status": _enum_name(value.status),
            },
        )

    return LegalAction(
        action_id=f"default:{_slug(str(value))}",
        kind="default",
        label="Wait for the server",
    )


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "name", value)).lower()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower()) or "unknown"
