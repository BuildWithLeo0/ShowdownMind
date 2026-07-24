from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LegalAction:
    action_id: str
    kind: str
    label: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BattleSnapshot:
    schema_version: str
    battle_id: str
    request_id: int
    turn: int
    battle_format: str
    own_side: dict[str, Any]
    opponent_side: dict[str, Any]
    field: dict[str, Any]
    resources: dict[str, Any]
    legal_actions: tuple[LegalAction, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["legal_actions"] = [action.to_dict() for action in self.legal_actions]
        return value

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


@dataclass(frozen=True)
class PolicyDecision:
    action_id: str
    confidence: float | None = None
    reason_codes: tuple[str, ...] = ()
    short_rationale: str = ""


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class PolicyResult:
    decision: PolicyDecision
    attempts: int
    raw_responses: tuple[str, ...]
    model_ids: tuple[str, ...]
    response_ids: tuple[str, ...]
    usages: tuple[TokenUsage, ...]
    errors: tuple[str, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class DecisionRecord:
    battle_id: str
    request_id: int
    turn: int
    snapshot_hash: str
    snapshot: dict[str, Any]
    action_id: str
    fallback_used: bool
    attempts: int
    model_ids: tuple[str, ...]
    errors: tuple[str, ...]
    raw_responses: tuple[str, ...]
    response_ids: tuple[str, ...] = ()
    usages: tuple[TokenUsage, ...] = ()
    confidence: float | None = None
    reason_codes: tuple[str, ...] = ()
    short_rationale: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
