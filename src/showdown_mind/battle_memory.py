from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from poke_env.battle import Move
from poke_env.data import to_id_str


EVENT_SCHEMA = "battle-event-v1"
MEMORY_SCHEMA = "battle-memory-v1"
PREDICTION_KINDS = (
    "attack",
    "switch",
    "setup",
    "status",
    "recovery",
    "protect",
    "unknown",
)


@dataclass(frozen=True)
class BattleEvent:
    event_id: str
    battle_id: str
    sequence: int
    turn: int
    kind: str
    actor: str | None
    target: str | None
    payload: dict[str, Any] = field(default_factory=dict)
    schema: str = EVENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpponentPrediction:
    kind: str
    detail: str
    confidence: float
    decision_turn: int

    def __post_init__(self) -> None:
        if self.kind not in PREDICTION_KINDS:
            raise ValueError(f"unknown prediction kind: {self.kind}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("prediction confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionResolution:
    predicted: OpponentPrediction
    actual_kind: str
    actual_detail: str
    observed_turn: int
    matched: bool
    evidence_event_id: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["predicted"] = self.predicted.to_dict()
        return value


@dataclass
class BattleMemory:
    battle_id: str
    cursor: int = 0
    current_turn: int = 0
    events: list[BattleEvent] = field(default_factory=list)
    revealed_moves: dict[str, set[str]] = field(default_factory=dict)
    item_history: dict[str, list[str]] = field(default_factory=dict)
    ability_history: dict[str, list[str]] = field(default_factory=dict)
    fainted: set[str] = field(default_factory=set)
    tera_history: list[dict[str, Any]] = field(default_factory=list)
    speed_evidence: list[dict[str, Any]] = field(default_factory=list)
    damage_evidence: list[dict[str, Any]] = field(default_factory=list)
    opponent_action_history: list[dict[str, Any]] = field(default_factory=list)
    opponent_action_counts: dict[str, int] = field(default_factory=dict)
    pending_prediction: OpponentPrediction | None = None
    latest_prediction_resolution: PredictionResolution | None = None
    schema: str = MEMORY_SCHEMA
    _moves_by_turn: dict[int, list[BattleEvent]] = field(default_factory=dict)
    _latest_move: BattleEvent | None = None

    def consume(self, events: tuple[BattleEvent, ...]) -> None:
        for event in events:
            self.events.append(event)
            self.current_turn = max(self.current_turn, event.turn)
            self._reduce(event)

    def set_prediction(self, prediction: OpponentPrediction) -> None:
        self.pending_prediction = prediction

    def audit_view(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "battle_id": self.battle_id,
            "cursor": self.cursor,
            "current_turn": self.current_turn,
            "events": [event.to_dict() for event in self.events],
            "revealed_moves": _sorted_mapping(self.revealed_moves),
            "item_history": self.item_history,
            "ability_history": self.ability_history,
            "fainted": sorted(self.fainted),
            "tera_history": self.tera_history,
            "speed_evidence": self.speed_evidence,
            "damage_evidence": self.damage_evidence,
            "opponent_action_history": self.opponent_action_history,
            "opponent_action_counts": self.opponent_action_counts,
            "pending_prediction": (
                self.pending_prediction.to_dict()
                if self.pending_prediction is not None
                else None
            ),
            "latest_prediction_resolution": (
                self.latest_prediction_resolution.to_dict()
                if self.latest_prediction_resolution is not None
                else None
            ),
        }

    def model_view(self) -> dict[str, Any]:
        important_events = [
            _model_event(event)
            for event in self.events
            if event.kind != "unknown_public_event"
        ][-6:]
        return _prune(
            {
                "schema": "battle-memory-model-v1",
                "revealed_moves": _sorted_mapping(self.revealed_moves),
                "item_history": self.item_history,
                "ability_history": self.ability_history,
                "fainted": sorted(self.fainted),
                "tera_history": self.tera_history[-4:],
                "speed_evidence": self.speed_evidence[-4:],
                "damage_evidence": self.damage_evidence[-4:],
                "opponent_behavior": {
                    "counts": self.opponent_action_counts,
                },
                "recent_events": important_events,
                "previous_prediction_resolution": (
                    self.latest_prediction_resolution.to_dict()
                    if self.latest_prediction_resolution is not None
                    else None
                ),
            }
        )

    def decision_view(self) -> dict[str, Any]:
        """Return only the evidence needed to choose the current action."""
        important_events = [
            _model_event(event)
            for event in self.events
            if event.kind != "unknown_public_event"
        ][-4:]
        return _prune(
            {
                "schema": "battle-memory-decision-v1",
                "revealed_moves": _sorted_mapping(self.revealed_moves),
                "item_history": self.item_history,
                "ability_history": self.ability_history,
                "fainted": sorted(self.fainted),
                "tera_history": self.tera_history[-2:],
                "speed_evidence": self.speed_evidence[-2:],
                "damage_evidence": self.damage_evidence[-2:],
                "opponent_behavior_counts": self.opponent_action_counts,
                "recent_events": important_events,
                "previous_prediction_resolution": (
                    self.latest_prediction_resolution.to_dict()
                    if self.latest_prediction_resolution is not None
                    else None
                ),
            }
        )

    def _reduce(self, event: BattleEvent) -> None:
        if event.kind == "turn_started":
            self.current_turn = event.turn
            self._latest_move = None
            return
        if event.kind == "move_used":
            if event.actor:
                move_id = str(event.payload.get("move_id") or "")
                if move_id:
                    self.revealed_moves.setdefault(event.actor, set()).add(move_id)
                self._moves_by_turn.setdefault(event.turn, []).append(event)
                self._record_speed_evidence(event.turn)
                if event.actor.startswith("opponent:"):
                    action_kind = _classify_move(move_id)
                    self._record_opponent_action(
                        event=event,
                        kind=action_kind,
                        detail=move_id,
                    )
            self._latest_move = event
            return
        if event.kind == "switch":
            if event.actor and event.actor.startswith("opponent:"):
                self._record_opponent_action(
                    event=event,
                    kind="switch",
                    detail=str(event.payload.get("species") or ""),
                )
            return
        if event.kind == "item_revealed" and event.actor:
            _append_unique(
                self.item_history.setdefault(event.actor, []),
                str(event.payload.get("item_id") or ""),
            )
            return
        if event.kind == "item_consumed" and event.actor:
            _append_unique(
                self.item_history.setdefault(event.actor, []),
                str(event.payload.get("item_id") or ""),
            )
            return
        if event.kind == "ability_revealed" and event.actor:
            _append_unique(
                self.ability_history.setdefault(event.actor, []),
                str(event.payload.get("ability_id") or ""),
            )
            return
        if event.kind == "faint" and event.actor:
            self.fainted.add(event.actor)
            return
        if event.kind == "tera" and event.actor:
            self.tera_history.append(
                {
                    "event_id": event.event_id,
                    "turn": event.turn,
                    "subject": event.actor,
                    "tera_type": event.payload.get("tera_type"),
                }
            )
            return
        if event.kind == "damage" and event.actor:
            source = event.payload.get("source") or []
            self.damage_evidence.append(
                {
                    "event_id": event.event_id,
                    "turn": event.turn,
                    "target": event.actor,
                    "hp_status": event.payload.get("hp_status"),
                    "source_move_event_id": (
                        self._latest_move.event_id
                        if self._latest_move is not None and not source
                        else None
                    ),
                    "source": source,
                }
            )

    def _record_speed_evidence(self, turn: int) -> None:
        moves = self._moves_by_turn.get(turn, [])
        if len(moves) < 2:
            return
        first, second = moves[-2:]
        if not first.actor or not second.actor:
            return
        if first.actor.split(":", 1)[0] == second.actor.split(":", 1)[0]:
            return
        first_move = _safe_move(str(first.payload.get("move_id") or ""))
        second_move = _safe_move(str(second.payload.get("move_id") or ""))
        if first_move is None or second_move is None:
            return
        first_priority = int(getattr(first_move, "priority", 0))
        second_priority = int(getattr(second_move, "priority", 0))
        if first_priority != second_priority:
            return
        evidence = {
            "event_ids": [first.event_id, second.event_id],
            "turn": turn,
            "acted_first": first.actor,
            "acted_second": second.actor,
            "priority": first_priority,
            "reliability": "conditional",
        }
        if not self.speed_evidence or self.speed_evidence[-1] != evidence:
            self.speed_evidence.append(evidence)

    def _record_opponent_action(
        self,
        *,
        event: BattleEvent,
        kind: str,
        detail: str,
    ) -> None:
        action = {
            "event_id": event.event_id,
            "turn": event.turn,
            "kind": kind,
            "detail": detail,
        }
        self.opponent_action_history.append(action)
        self.opponent_action_counts[kind] = (
            self.opponent_action_counts.get(kind, 0) + 1
        )
        prediction = self.pending_prediction
        if prediction is not None and event.turn >= prediction.decision_turn:
            self.latest_prediction_resolution = PredictionResolution(
                predicted=prediction,
                actual_kind=kind,
                actual_detail=detail,
                observed_turn=event.turn,
                matched=prediction.kind == kind,
                evidence_event_id=event.event_id,
            )
            self.pending_prediction = None


class PokeEnvEventAdapter:
    """Convert new player-visible poke-env replay messages into stable events."""

    def consume(
        self,
        battle: Any,
        memory: BattleMemory,
    ) -> tuple[BattleEvent, ...]:
        replay_data = getattr(battle, "_replay_data", None)
        if not isinstance(replay_data, list):
            raise RuntimeError(
                "poke-env battle does not expose the pinned replay-data interface"
            )
        new_messages = replay_data[memory.cursor :]
        current_turn = memory.current_turn
        events: list[BattleEvent] = []
        player_role = str(getattr(battle, "player_role", "") or "")
        for offset, raw in enumerate(new_messages, start=memory.cursor):
            if not isinstance(raw, list) or len(raw) < 2:
                continue
            message = [str(part) for part in raw]
            if message[1] == "turn" and len(message) > 2:
                current_turn = _safe_int(message[2], current_turn)
            events.append(
                _parse_event(
                    battle_id=memory.battle_id,
                    sequence=offset,
                    turn=current_turn,
                    message=message,
                    player_role=player_role,
                )
            )
        memory.cursor = len(replay_data)
        memory.current_turn = current_turn
        return tuple(events)


def _parse_event(
    *,
    battle_id: str,
    sequence: int,
    turn: int,
    message: list[str],
    player_role: str,
) -> BattleEvent:
    message_type = message[1]
    actor: str | None = None
    target: str | None = None
    payload: dict[str, Any] = {}
    kind = "unknown_public_event"

    if message_type == "turn" and len(message) > 2:
        kind = "turn_started"
        turn = _safe_int(message[2], turn)
    elif message_type == "move" and len(message) > 3:
        kind = "move_used"
        actor = _subject(message[2], player_role)
        target = _subject(message[4], player_role) if len(message) > 4 else None
        payload["move_id"] = to_id_str(message[3])
    elif message_type in {"switch", "drag"} and len(message) > 4:
        kind = "switch"
        actor = _subject(message[2], player_role)
        payload = {
            "species": to_id_str(message[3].split(",", 1)[0]),
            "hp_status": message[4],
            "forced": message_type == "drag",
        }
    elif message_type == "-damage" and len(message) > 3:
        kind = "damage"
        actor = _subject(message[2], player_role)
        payload = {"hp_status": message[3], "source": message[4:]}
    elif message_type == "-heal" and len(message) > 3:
        kind = "heal"
        actor = _subject(message[2], player_role)
        payload = {"hp_status": message[3], "source": message[4:]}
    elif message_type == "faint" and len(message) > 2:
        kind = "faint"
        actor = _subject(message[2], player_role)
    elif message_type == "-item" and len(message) > 3:
        kind = "item_revealed"
        actor = _subject(message[2], player_role)
        payload["item_id"] = to_id_str(message[3])
    elif message_type == "-enditem" and len(message) > 3:
        kind = "item_consumed"
        actor = _subject(message[2], player_role)
        payload["item_id"] = to_id_str(message[3])
    elif message_type == "-ability" and len(message) > 3:
        kind = "ability_revealed"
        actor = _subject(message[2], player_role)
        payload["ability_id"] = to_id_str(message[3])
    elif message_type == "-status" and len(message) > 3:
        kind = "status"
        actor = _subject(message[2], player_role)
        payload["status"] = to_id_str(message[3])
    elif message_type in {"-boost", "-unboost"} and len(message) > 4:
        kind = "boost"
        actor = _subject(message[2], player_role)
        amount = _safe_int(message[4], 0)
        if message_type == "-unboost":
            amount *= -1
        payload = {"stat": to_id_str(message[3]), "amount": amount}
    elif message_type == "-terastallize" and len(message) > 3:
        kind = "tera"
        actor = _subject(message[2], player_role)
        payload["tera_type"] = to_id_str(message[3])
    else:
        payload = {"message_type": message_type, "parts": message[2:]}

    return BattleEvent(
        event_id=f"{battle_id}:{sequence}",
        battle_id=battle_id,
        sequence=sequence,
        turn=turn,
        kind=kind,
        actor=actor,
        target=target,
        payload=payload,
    )


def _subject(identifier: str, player_role: str) -> str | None:
    if not identifier:
        return None
    role = identifier[:2]
    label = identifier.split(":", 1)[-1].strip()
    subject_id = to_id_str(label)
    if not subject_id:
        return None
    side = "own" if player_role and role == player_role else "opponent"
    return f"{side}:{subject_id}"


def _classify_move(move_id: str) -> str:
    move = _safe_move(move_id)
    if move is None:
        return "unknown"
    if bool(getattr(move, "is_protect_move", False)):
        return "protect"
    if float(getattr(move, "heal", 0.0) or 0.0) > 0:
        return "recovery"
    if getattr(move, "category", None) is not None:
        category = str(getattr(getattr(move, "category"), "name", "")).lower()
        if category in {"physical", "special"} and int(
            getattr(move, "base_power", 0) or 0
        ) > 0:
            return "attack"
    boosts = getattr(move, "boosts", None) or {}
    self_boost = getattr(move, "self_boost", None) or {}
    target = str(
        getattr(getattr(move, "target", None), "name", "")
    ).lower()
    if (
        target == "self"
        and any(float(value) > 0 for value in boosts.values())
    ) or any(float(value) > 0 for value in self_boost.values()):
        return "setup"
    return "status"


def _safe_move(move_id: str) -> Move | None:
    if not move_id:
        return None
    try:
        return Move(move_id, gen=9)
    except (KeyError, ValueError):
        return None


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sorted_mapping(values: dict[str, set[str]]) -> dict[str, list[str]]:
    return {key: sorted(items) for key, items in sorted(values.items()) if items}


def _model_event(event: BattleEvent) -> dict[str, Any]:
    return _prune(
        {
            "event_id": event.event_id,
            "turn": event.turn,
            "kind": event.kind,
            "actor": event.actor,
            "target": event.target,
            "payload": event.payload,
        }
    )


def _prune(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: compact
            for key, item in value.items()
            if not _empty(compact := _prune(item))
        }
    if isinstance(value, list):
        return [
            compact
            for item in value
            if not _empty(compact := _prune(item))
        ]
    return value


def _empty(value: Any) -> bool:
    return value is None or value == "" or (
        isinstance(value, (dict, list, tuple, set)) and not value
    )
