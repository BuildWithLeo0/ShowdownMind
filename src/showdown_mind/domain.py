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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BattleSnapshot:
        return cls(
            schema_version=str(value["schema_version"]),
            battle_id=str(value["battle_id"]),
            request_id=int(value["request_id"]),
            turn=int(value["turn"]),
            battle_format=str(value["battle_format"]),
            own_side=dict(value["own_side"]),
            opponent_side=dict(value["opponent_side"]),
            field=dict(value["field"]),
            resources=dict(value["resources"]),
            legal_actions=tuple(
                LegalAction(
                    action_id=str(action["action_id"]),
                    kind=str(action["kind"]),
                    label=str(action["label"]),
                    details=dict(action.get("details", {})),
                )
                for action in value["legal_actions"]
            ),
        )

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
    tool_call_ids: tuple[str, ...]
    usages: tuple[TokenUsage, ...]
    errors: tuple[str, ...]
    elapsed_seconds: float
    policy_input_format: str
    policy_input_hash: str
    policy_input_characters: int
    policy_input: dict[str, Any]


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
    policy_input_format: str = ""
    policy_input_hash: str = ""
    policy_input_characters: int = 0
    policy_input: dict[str, Any] = field(default_factory=dict)
    response_ids: tuple[str, ...] = ()
    tool_name: str = ""
    tool_call_ids: tuple[str, ...] = ()
    usages: tuple[TokenUsage, ...] = ()
    confidence: float | None = None
    reason_codes: tuple[str, ...] = ()
    short_rationale: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
