from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from poke_env.battle import Move
from poke_env.player.battle_order import SingleBattleOrder

from showdown_mind.actions import ActionCatalog, CatalogEntry
from showdown_mind.domain import BattleSnapshot, LegalAction
from showdown_mind.models import ModelClient
from showdown_mind.policy import SingleCallPolicy


@dataclass(frozen=True)
class ModelCheckResult:
    model_id: str
    prompt_format: str
    model_input_characters: int
    action_id: str
    confidence: float | None
    reason_codes: tuple[str, ...]
    short_rationale: str
    tool_call_id: str | None
    attempts: int
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_model_check(
    model_client: ModelClient,
    *,
    prompt_format: str = "pruned",
) -> ModelCheckResult:
    catalog = _check_catalog()
    snapshot = BattleSnapshot(
        schema_version="1.0",
        battle_id="model-connectivity-check",
        request_id=1,
        turn=1,
        battle_format="gen9randombattle",
        own_side={
            "active": "pikachu",
            "team": [],
            "side_conditions": {},
        },
        opponent_side={
            "active": "gyarados",
            "revealed_team": [],
            "side_conditions": {},
            "used_tera": False,
        },
        field={"weather": {}, "fields": {}},
        resources={
            "can_tera": False,
            "used_tera": False,
            "force_switch": False,
            "trapped": False,
        },
        legal_actions=catalog.actions,
    )
    result = await SingleCallPolicy(
        model_client,
        input_format=prompt_format,
    ).decide(snapshot, catalog)
    return ModelCheckResult(
        model_id=result.model_ids[-1],
        prompt_format=result.policy_input_format,
        model_input_characters=result.policy_input_characters,
        action_id=result.decision.action_id,
        confidence=result.decision.confidence,
        reason_codes=result.decision.reason_codes,
        short_rationale=result.decision.short_rationale,
        tool_call_id=(result.tool_call_ids[-1] if result.tool_call_ids else None),
        attempts=result.attempts,
        input_tokens=sum(usage.input_tokens for usage in result.usages),
        output_tokens=sum(usage.output_tokens for usage in result.usages),
        total_tokens=sum(usage.total_tokens for usage in result.usages),
    )


def _check_catalog() -> ActionCatalog:
    actions = (
        LegalAction(
            action_id="move:thunderbolt",
            kind="move",
            label="Use thunderbolt",
            details={
                "type": "electric",
                "category": "special",
                "base_power": 90,
                "accuracy": 100,
            },
        ),
        LegalAction(
            action_id="move:quickattack",
            kind="move",
            label="Use quickattack",
            details={
                "type": "normal",
                "category": "physical",
                "base_power": 40,
                "accuracy": 100,
            },
        ),
    )
    orders = (
        SingleBattleOrder(Move("thunderbolt", gen=9)),
        SingleBattleOrder(Move("quickattack", gen=9)),
    )
    return ActionCatalog(
        tuple(
            CatalogEntry(action=action, order=order)
            for action, order in zip(actions, orders, strict=True)
        )
    )
