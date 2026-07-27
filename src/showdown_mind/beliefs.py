from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from poke_env.data import to_id_str

from showdown_mind.battle_memory import BattleMemory
from showdown_mind.domain import BattleSnapshot
from showdown_mind.paths import SHOWDOWN_DIR


BELIEF_SCHEMA = "belief-state-v1"
CONFIDENCE_ORDER = {"likely": 0, "possible": 1, "unsupported": 2}


@dataclass(frozen=True)
class BeliefHypothesis:
    subject: str
    kind: str
    value: str
    confidence: str
    evidence_ids: tuple[str, ...]
    contradiction_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.confidence not in CONFIDENCE_ORDER:
            raise ValueError(f"unknown belief confidence: {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeliefState:
    battle_id: str
    updated_turn: int
    hypotheses: tuple[BeliefHypothesis, ...] = ()
    unavailable_priors: tuple[str, ...] = ()
    schema: str = BELIEF_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "battle_id": self.battle_id,
            "updated_turn": self.updated_turn,
            "hypotheses": [
                hypothesis.to_dict() for hypothesis in self.hypotheses
            ],
            "unavailable_priors": list(self.unavailable_priors),
        }

    def model_view(
        self,
        *,
        active_subject: str | None = None,
        max_hypotheses: int = 16,
        active_only: bool = False,
    ) -> dict[str, Any]:
        grouped: dict[tuple[str, str], list[BeliefHypothesis]] = {}
        for hypothesis in self.hypotheses:
            if (
                active_only
                and active_subject is not None
                and hypothesis.subject != active_subject
            ):
                continue
            grouped.setdefault(
                (hypothesis.subject, hypothesis.kind),
                [],
            ).append(hypothesis)
        selected: list[BeliefHypothesis] = []
        group_keys = sorted(
            grouped,
            key=lambda key: (
                0 if key[0] == active_subject else 1,
                key[0],
                key[1],
            ),
        )
        for key in group_keys:
            values = sorted(
                grouped[key],
                key=lambda value: (
                    CONFIDENCE_ORDER[value.confidence],
                    value.value,
                ),
            )
            selected.extend(values[:3])
            if len(selected) >= max_hypotheses:
                selected = selected[:max_hypotheses]
                break
        return {
            "schema": "belief-model-v1",
            "hypotheses": [_model_hypothesis(value) for value in selected],
            "unavailable_priors": list(self.unavailable_priors),
        }

    def possible_move_ids(self, subject: str) -> tuple[str, ...]:
        return tuple(
            hypothesis.value
            for hypothesis in self.hypotheses
            if hypothesis.subject == subject
            and hypothesis.kind == "possible_move"
            and hypothesis.confidence in {"likely", "possible"}
        )


class RandomBattlePriorLoader:
    def __init__(self, path: Path | None = None):
        self.path = path or (
            SHOWDOWN_DIR / "data" / "random-battles" / "gen9" / "sets.json"
        )
        self._sets: dict[str, Any] | None = None

    @property
    def source_id(self) -> str:
        return "pokemon-showdown:gen9randombattle:sets.json"

    def sets_for(self, species: str) -> tuple[dict[str, Any], ...]:
        data = self._load().get(to_id_str(species), {})
        values = data.get("sets", []) if isinstance(data, dict) else []
        return tuple(value for value in values if isinstance(value, dict))

    def _load(self) -> dict[str, Any]:
        if self._sets is not None:
            return self._sets
        if not self.path.is_file():
            self._sets = {}
            return self._sets
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"random battle prior must be an object: {self.path}")
        self._sets = value
        return self._sets


@dataclass
class RuleBeliefTracker:
    priors: RandomBattlePriorLoader = field(
        default_factory=RandomBattlePriorLoader
    )

    def update(
        self,
        snapshot: BattleSnapshot,
        memory: BattleMemory,
    ) -> BeliefState:
        hypotheses: list[BeliefHypothesis] = []
        unavailable: list[str] = []
        opponent = snapshot.opponent_side
        for pokemon in opponent.get("revealed_team", []):
            if not isinstance(pokemon, dict):
                continue
            species = to_id_str(str(pokemon.get("species") or ""))
            if not species:
                continue
            subject = _memory_subject(memory, species)
            sets = list(self.priors.sets_for(species))
            if not sets:
                unavailable.append(species)
                continue
            revealed_moves = {
                to_id_str(str(move))
                for move in pokemon.get("moves", [])
                if move
            }
            known_ability = to_id_str(str(pokemon.get("ability") or ""))
            known_tera = to_id_str(str(pokemon.get("tera_type") or ""))
            compatible = [
                candidate
                for candidate in sets
                if _compatible(
                    candidate,
                    revealed_moves=revealed_moves,
                    known_ability=known_ability,
                    known_tera=known_tera,
                )
            ]
            if not compatible:
                unavailable.append(f"{species}:no_compatible_public_prior")
                continue
            candidates = compatible
            prior_id = f"prior:gen9randombattle:{species}"
            evidence_ids = (prior_id,) + tuple(
                _subject_evidence_ids(memory, subject)
            )
            hypotheses.extend(
                _candidate_hypotheses(
                    subject=subject,
                    candidates=candidates,
                    revealed_moves=revealed_moves,
                    evidence_ids=evidence_ids,
                )
            )
        hypotheses.sort(
            key=lambda value: (
                value.subject,
                value.kind,
                CONFIDENCE_ORDER[value.confidence],
                value.value,
            )
        )
        return BeliefState(
            battle_id=snapshot.battle_id,
            updated_turn=snapshot.turn,
            hypotheses=tuple(hypotheses),
            unavailable_priors=tuple(sorted(set(unavailable))),
        )


def belief_changes(
    previous: BeliefState | None,
    current: BeliefState,
) -> tuple[dict[str, Any], ...]:
    before = (
        {
            (item.subject, item.kind, item.value): item.confidence
            for item in previous.hypotheses
        }
        if previous is not None
        else {}
    )
    after = {
        (item.subject, item.kind, item.value): item.confidence
        for item in current.hypotheses
    }
    changes: list[dict[str, Any]] = []
    for key in sorted(set(before) | set(after)):
        old = before.get(key)
        new = after.get(key)
        if old == new:
            continue
        subject, kind, value = key
        changes.append(
            {
                "subject": subject,
                "kind": kind,
                "value": value,
                "before": old,
                "after": new,
            }
        )
    return tuple(changes)


def _memory_subject(memory: BattleMemory, species: str) -> str:
    suffix = f":{species}"
    candidates = sorted(
        subject
        for subject in (
            set(memory.revealed_moves)
            | set(memory.item_history)
            | set(memory.ability_history)
        )
        if subject.startswith("opponent:") and subject.endswith(suffix)
    )
    return candidates[0] if candidates else f"opponent:{species}"


def _subject_evidence_ids(
    memory: BattleMemory,
    subject: str,
) -> tuple[str, ...]:
    return tuple(
        event.event_id
        for event in memory.events
        if event.actor == subject
        and event.kind
        in {
            "move_used",
            "item_revealed",
            "item_consumed",
            "ability_revealed",
            "tera",
        }
    )


def _compatible(
    candidate: dict[str, Any],
    *,
    revealed_moves: set[str],
    known_ability: str,
    known_tera: str,
) -> bool:
    movepool = {to_id_str(str(value)) for value in candidate.get("movepool", [])}
    abilities = {
        to_id_str(str(value)) for value in candidate.get("abilities", [])
    }
    tera_types = {
        to_id_str(str(value)) for value in candidate.get("teraTypes", [])
    }
    if revealed_moves and not revealed_moves.issubset(movepool):
        return False
    if known_ability and abilities and known_ability not in abilities:
        return False
    if known_tera and tera_types and known_tera not in tera_types:
        return False
    return True


def _candidate_hypotheses(
    *,
    subject: str,
    candidates: list[dict[str, Any]],
    revealed_moves: set[str],
    evidence_ids: tuple[str, ...],
) -> list[BeliefHypothesis]:
    values_by_kind: dict[str, list[set[str]]] = {
        "possible_move": [],
        "possible_ability": [],
        "possible_tera_type": [],
        "possible_role": [],
    }
    for candidate in candidates:
        values_by_kind["possible_move"].append(
            {
                to_id_str(str(value))
                for value in candidate.get("movepool", [])
                if to_id_str(str(value)) not in revealed_moves
            }
        )
        values_by_kind["possible_ability"].append(
            {
                to_id_str(str(value))
                for value in candidate.get("abilities", [])
            }
        )
        values_by_kind["possible_tera_type"].append(
            {
                to_id_str(str(value))
                for value in candidate.get("teraTypes", [])
            }
        )
        role = to_id_str(str(candidate.get("role") or ""))
        values_by_kind["possible_role"].append({role} if role else set())

    hypotheses: list[BeliefHypothesis] = []
    for kind, candidate_sets in values_by_kind.items():
        union = set().union(*candidate_sets) if candidate_sets else set()
        for value in sorted(union):
            present_count = sum(value in items for items in candidate_sets)
            confidence = (
                "likely"
                if candidate_sets and present_count == len(candidate_sets)
                else "possible"
            )
            hypotheses.append(
                BeliefHypothesis(
                    subject=subject,
                    kind=kind,
                    value=value,
                    confidence=confidence,
                    evidence_ids=evidence_ids,
                )
            )
    return hypotheses


def _model_hypothesis(value: BeliefHypothesis) -> dict[str, Any]:
    evidence = list(value.evidence_ids)
    if len(evidence) > 3:
        evidence = [evidence[0], *evidence[-2:]]
    return {
        "subject": value.subject,
        "kind": value.kind,
        "value": value.value,
        "confidence": value.confidence,
        "evidence_ids": evidence,
        "contradiction_ids": list(value.contradiction_ids[-2:]),
    }
