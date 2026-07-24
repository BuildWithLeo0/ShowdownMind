from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder

from showdown_mind.actions import ActionCatalog
from showdown_mind.domain import DecisionRecord, TokenUsage
from showdown_mind.models import ACTION_TOOL_NAME
from showdown_mind.observation import BattleSnapshotBuilder
from showdown_mind.policy import PolicyFailure, SingleCallPolicy

DecisionSink = Callable[[DecisionRecord], None]


class ResearchPlayer(Player):
    """A Policy-first player with strict action validation and fallback."""

    def __init__(
        self,
        policy: SingleCallPolicy,
        *,
        fallback_seed: str = "showdown-mind",
        decision_sink: DecisionSink | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._policy = policy
        self._fallback_seed = fallback_seed
        self._snapshot_builder = BattleSnapshotBuilder()
        self._decision_sink = decision_sink
        self.decision_records: list[DecisionRecord] = []
        self._request_cache: dict[tuple[str, int, int], BattleOrder] = {}

    async def choose_move(self, battle: Any) -> BattleOrder:
        catalog = ActionCatalog.from_battle(battle)
        if not len(catalog):
            return self.choose_default_move()

        snapshot = self._snapshot_builder.build(battle, catalog.actions)
        request_key = (
            snapshot.battle_id,
            snapshot.request_id,
            snapshot.turn,
        )
        cached_order = self._request_cache.get(request_key)
        if cached_order is not None:
            return cached_order

        fallback_used = False
        attempts = 0
        errors: tuple[str, ...] = ()
        raw_responses: tuple[str, ...] = ()
        model_ids: tuple[str, ...] = ()
        response_ids: tuple[str, ...] = ()
        tool_call_ids: tuple[str, ...] = ()
        tool_names: tuple[str, ...] = ()
        tool_executions: tuple[dict[str, Any], ...] = ()
        tactical_analysis: dict[str, Any] = {}
        usages: tuple[TokenUsage, ...] = ()
        model_calls = 0
        expected_model_calls = 1
        confidence: float | None = None
        reason_codes: tuple[str, ...] = ()
        short_rationale = ""
        elapsed_seconds = 0.0
        policy_input_format = ""
        policy_input_hash = ""
        policy_input_characters = 0
        policy_input: dict[str, Any] = {}

        try:
            result = await self._policy.decide(
                snapshot,
                catalog,
                battle=battle,
            )
            action_id = result.decision.action_id
            attempts = result.attempts
            errors = result.errors
            raw_responses = result.raw_responses
            model_ids = result.model_ids
            response_ids = result.response_ids
            tool_call_ids = result.tool_call_ids
            tool_names = result.tool_names
            tool_executions = result.tool_executions
            tactical_analysis = result.tactical_analysis
            usages = result.usages
            model_calls = result.model_calls
            expected_model_calls = result.expected_model_calls
            confidence = result.decision.confidence
            reason_codes = result.decision.reason_codes
            short_rationale = result.decision.short_rationale
            elapsed_seconds = result.elapsed_seconds
            policy_input_format = result.policy_input_format
            policy_input_hash = result.policy_input_hash
            policy_input_characters = result.policy_input_characters
            policy_input = result.policy_input
        except PolicyFailure as exc:
            fallback_used = True
            errors = exc.errors
            raw_responses = exc.raw_responses
            model_ids = exc.model_ids
            response_ids = exc.response_ids
            tool_call_ids = exc.tool_call_ids
            tool_names = exc.tool_names
            tool_executions = exc.tool_executions
            tactical_analysis = exc.tactical_analysis
            usages = exc.usages
            model_calls = exc.model_calls
            expected_model_calls = exc.expected_model_calls
            attempts = exc.attempts
            elapsed_seconds = exc.elapsed_seconds
            policy_input_format = exc.policy_input.format_name
            policy_input_hash = exc.policy_input.fingerprint()
            policy_input_characters = exc.policy_input.characters
            policy_input = exc.policy_input.payload
            action_id = deterministic_fallback_action_id(
                seed=self._fallback_seed,
                battle_id=snapshot.battle_id,
                request_id=snapshot.request_id,
                catalog=catalog,
            )
            reason_codes = ("FALLBACK",)
            short_rationale = "模型未能返回有效工具调用，系统使用确定性合法备用动作。"

        record = DecisionRecord(
            battle_id=snapshot.battle_id,
            request_id=snapshot.request_id,
            turn=snapshot.turn,
            snapshot_hash=snapshot.fingerprint(),
            snapshot=snapshot.to_dict(),
            action_id=action_id,
            fallback_used=fallback_used,
            attempts=attempts,
            model_ids=model_ids,
            errors=errors,
            raw_responses=raw_responses,
            policy_input_format=policy_input_format,
            policy_input_hash=policy_input_hash,
            policy_input_characters=policy_input_characters,
            policy_input=policy_input,
            response_ids=response_ids,
            tool_name=ACTION_TOOL_NAME,
            tool_names=tool_names,
            tool_call_ids=tool_call_ids,
            tool_executions=tool_executions,
            tactical_analysis=tactical_analysis,
            model_calls=model_calls,
            expected_model_calls=expected_model_calls,
            usages=usages,
            confidence=confidence,
            reason_codes=reason_codes,
            short_rationale=short_rationale,
            elapsed_seconds=elapsed_seconds,
        )
        self.decision_records.append(record)
        if self._decision_sink is not None:
            self._decision_sink(record)
        order = catalog.resolve(action_id)
        self._request_cache[request_key] = order
        return order


def deterministic_fallback_action_id(
    *,
    seed: str,
    battle_id: str,
    request_id: int,
    catalog: ActionCatalog,
) -> str:
    material = f"{seed}:{battle_id}:{request_id}".encode()
    index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(catalog)
    return catalog.actions[index].action_id
