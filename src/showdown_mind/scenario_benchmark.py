from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from showdown_mind.controlled_policy import (
    CONTROLLED_INPUT_SCHEMA,
    CONTROLLED_SYSTEM_PROMPT,
    _controlled_action_tool,
    _parse_controlled_decision,
    _repair_action_request,
)
from showdown_mind.domain import TokenUsage
from showdown_mind.models import (
    ACTION_TOOL_NAME,
    ModelCallError,
    ModelClient,
    ModelRequest,
)
from showdown_mind.policy_input import CompiledPolicyInput


SCENARIO_BANK_SCHEMA = "showdown-mind-scenario-bank-v1"
SCENARIO_REPORT_SCHEMA = "showdown-mind-scenario-report-v1"


@dataclass(frozen=True)
class ScenarioBankSummary:
    path: str
    scenarios: int
    tagged_protocol_regression: int
    tagged_source_loss: int
    calculator_references: int
    curated_references: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _FrozenAction:
    action_id: str


class _FrozenCatalog:
    def __init__(self, action_ids: list[str]):
        if not action_ids or len(action_ids) != len(set(action_ids)):
            raise ValueError("scenario legal actions must be unique and non-empty")
        self.actions = tuple(_FrozenAction(action_id) for action_id in action_ids)
        self._action_ids = frozenset(action_ids)

    def contains(self, action_id: str) -> bool:
        return action_id in self._action_ids


def build_scenario_bank(
    sources: list[Path],
    *,
    output_path: Path,
    limit: int = 20,
    force: bool = False,
) -> ScenarioBankSummary:
    if limit <= 0:
        raise ValueError("scenario limit must be positive")
    if output_path.exists() and not force:
        raise ValueError(f"scenario bank already exists: {output_path}")
    logs = _discover_logs(sources)
    if not logs:
        raise ValueError("no decision JSONL logs were found")
    candidates: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for log_path in logs:
        outcome = _source_outcome(log_path)
        for line_number, record in _read_records(log_path):
            scenario = _scenario_from_record(
                record,
                log_path=log_path,
                line_number=line_number,
                outcome=outcome,
            )
            if scenario is None:
                continue
            context_hash = str(scenario["context_hash"])
            if context_hash in seen_hashes:
                continue
            seen_hashes.add(context_hash)
            candidates.append(scenario)
    selected = _select_scenarios(candidates, limit=limit)
    if not selected:
        raise ValueError("no controlled-agent decision contexts were found")
    bank = {
        "schema": SCENARIO_BANK_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "description": (
            "Frozen player-visible controlled-Agent decisions. Historical and "
            "calculator actions are regression signals, not ground truth."
        ),
        "scenarios": selected,
    }
    validate_scenario_bank(bank)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return scenario_bank_summary(bank, path=output_path)


def load_scenario_bank(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"scenario bank does not exist: {path}")
    try:
        bank = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"scenario bank is not valid JSON: {path}") from exc
    validate_scenario_bank(bank)
    return bank


def validate_scenario_bank(bank: Any) -> None:
    if not isinstance(bank, dict) or bank.get("schema") != SCENARIO_BANK_SCHEMA:
        raise ValueError(f"scenario bank schema must be {SCENARIO_BANK_SCHEMA}")
    scenarios = bank.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenario bank must contain at least one scenario")
    scenario_ids: set[str] = set()
    context_hashes: set[str] = set()
    for index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenario {index} must be an object")
        scenario_id = str(scenario.get("id") or "")
        if not scenario_id or scenario_id in scenario_ids:
            raise ValueError(f"scenario {index} has a missing or duplicate id")
        scenario_ids.add(scenario_id)
        policy_input = scenario.get("policy_input")
        if (
            not isinstance(policy_input, dict)
            or policy_input.get("schema") != CONTROLLED_INPUT_SCHEMA
        ):
            raise ValueError(
                f"scenario {scenario_id} must contain a controlled Agent input"
            )
        legal_actions = _legal_actions(policy_input)
        action_ids = [action["action_id"] for action in legal_actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError(f"scenario {scenario_id} has duplicate legal actions")
        expected_hash = _context_hash(policy_input)
        if scenario.get("context_hash") != expected_hash:
            raise ValueError(f"scenario {scenario_id} context hash is invalid")
        if expected_hash in context_hashes:
            raise ValueError(f"scenario {scenario_id} duplicates another context")
        context_hashes.add(expected_hash)
        for field in (
            "historical_action_id",
            "calculator_recommended_action_ids",
            "acceptable_action_ids",
        ):
            values = (
                [scenario.get(field)]
                if field == "historical_action_id"
                else scenario.get(field, [])
            )
            if not isinstance(values, list):
                raise ValueError(f"scenario {scenario_id} {field} must be a list")
            unknown = [
                action_id
                for action_id in values
                if action_id and action_id not in action_ids
            ]
            if unknown:
                raise ValueError(
                    f"scenario {scenario_id} {field} contains illegal actions"
                )


def scenario_bank_summary(
    bank: dict[str, Any],
    *,
    path: Path,
) -> ScenarioBankSummary:
    scenarios = bank["scenarios"]
    return ScenarioBankSummary(
        path=str(path),
        scenarios=len(scenarios),
        tagged_protocol_regression=sum(
            "protocol_regression" in scenario.get("tags", [])
            for scenario in scenarios
        ),
        tagged_source_loss=sum(
            "source_loss" in scenario.get("tags", []) for scenario in scenarios
        ),
        calculator_references=sum(
            bool(scenario.get("calculator_recommended_action_ids"))
            for scenario in scenarios
        ),
        curated_references=sum(
            bool(scenario.get("acceptable_action_ids"))
            for scenario in scenarios
        ),
    )


async def evaluate_scenario_bank(
    model_client: ModelClient,
    bank: dict[str, Any],
    *,
    output_path: Path | None = None,
    limit: int | None = None,
    timeout_seconds: float = 45.0,
    max_repairs: int = 1,
) -> dict[str, Any]:
    validate_scenario_bank(bank)
    if timeout_seconds <= 0:
        raise ValueError("scenario timeout must be positive")
    if max_repairs not in (0, 1):
        raise ValueError("max_repairs must be 0 or 1")
    scenarios = list(bank["scenarios"])
    if limit is not None:
        if limit <= 0:
            raise ValueError("scenario evaluation limit must be positive")
        scenarios = scenarios[:limit]
    results = []
    for scenario in scenarios:
        results.append(
            await _evaluate_scenario(
                model_client,
                scenario,
                timeout_seconds=timeout_seconds,
                max_repairs=max_repairs,
            )
        )
    report = _scenario_report(bank, results)
    if output_path is not None:
        if output_path.exists():
            raise ValueError(f"scenario report already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return report


async def _evaluate_scenario(
    model_client: ModelClient,
    scenario: dict[str, Any],
    *,
    timeout_seconds: float,
    max_repairs: int,
) -> dict[str, Any]:
    policy_input = CompiledPolicyInput(
        CONTROLLED_INPUT_SCHEMA,
        scenario["policy_input"],
    )
    action_ids = [
        str(action["action_id"])
        for action in _legal_actions(policy_input.payload)
    ]
    catalog = _FrozenCatalog(action_ids)
    tool = _controlled_action_tool(catalog)  # type: ignore[arg-type]
    request = ModelRequest(
        system_prompt=CONTROLLED_SYSTEM_PROMPT,
        user_prompt=policy_input.canonical_json(),
        tool=tool,
    )
    raw_responses: list[str] = []
    usages: list[TokenUsage] = []
    errors: list[str] = []
    normalizations: list[str] = []
    model_ids: list[str] = []
    started = time.monotonic()
    decision = None
    calls = 0
    attempts = 0
    for attempt in range(max_repairs + 1):
        attempts = attempt + 1
        calls += 1
        attempt_normalizations: list[str] = []
        try:
            response = await asyncio.wait_for(
                model_client.complete(request),
                timeout=timeout_seconds,
            )
            raw_responses.append(response.content)
            model_ids.append(response.model_id)
            if response.usage is not None:
                usages.append(response.usage)
            decision = _parse_controlled_decision(
                response.content,
                catalog,  # type: ignore[arg-type]
                normalizations=attempt_normalizations,
            )
        except (TimeoutError, ModelCallError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt >= max_repairs:
                break
            continue
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(error)
            if attempt >= max_repairs:
                break
            request = _repair_action_request(
                policy_input,
                invalid_response=response.content,
                error=error,
                tool=tool,
            )
            continue
        normalizations.extend(attempt_normalizations)
        break

    selected = decision.action_id if decision is not None else ""
    calculator_ids = list(scenario.get("calculator_recommended_action_ids") or [])
    acceptable_ids = list(scenario.get("acceptable_action_ids") or [])
    return {
        "scenario_id": scenario["id"],
        "status": "valid" if decision is not None else "invalid",
        "action_id": selected,
        "historical_action_id": scenario.get("historical_action_id", ""),
        "historical_agreement": (
            bool(selected) and selected == scenario.get("historical_action_id")
        ),
        "calculator_alignment": (
            bool(selected) and bool(calculator_ids) and selected in calculator_ids
        ),
        "calculator_reference_available": bool(calculator_ids),
        "curated_alignment": (
            bool(selected) and bool(acceptable_ids) and selected in acceptable_ids
        ),
        "curated_reference_available": bool(acceptable_ids),
        "attempts": attempts,
        "model_calls": calls,
        "model_ids": model_ids,
        "errors": errors,
        "normalizations": normalizations,
        "raw_responses": raw_responses,
        "reason_codes": list(decision.reason_codes) if decision else [],
        "short_rationale": decision.short_rationale if decision else "",
        "opponent_prediction": (
            decision.opponent_prediction if decision else {}
        ),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "usage": {
            "input_tokens": sum(usage.input_tokens for usage in usages),
            "output_tokens": sum(usage.output_tokens for usage in usages),
            "total_tokens": sum(usage.total_tokens for usage in usages),
        },
    }


def _scenario_report(
    bank: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [result for result in results if result["status"] == "valid"]
    calculator = [
        result for result in valid if result["calculator_reference_available"]
    ]
    curated = [
        result for result in valid if result["curated_reference_available"]
    ]
    return {
        "schema": SCENARIO_REPORT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "bank_schema": bank["schema"],
        "bank_created_at": bank.get("created_at"),
        "metrics": {
            "scenarios": len(results),
            "valid_decisions": len(valid),
            "protocol_success_rate": _rate(len(valid), len(results)),
            "retry_decisions": sum(result["attempts"] > 1 for result in results),
            "normalized_decisions": sum(
                bool(result["normalizations"]) for result in results
            ),
            "historical_agreement_rate": _rate(
                sum(result["historical_agreement"] for result in valid),
                len(valid),
            ),
            "calculator_reference_scenarios": len(calculator),
            "calculator_alignment_rate": _rate(
                sum(result["calculator_alignment"] for result in calculator),
                len(calculator),
            ),
            "curated_reference_scenarios": len(curated),
            "curated_alignment_rate": _rate(
                sum(result["curated_alignment"] for result in curated),
                len(curated),
            ),
            "model_calls": sum(result["model_calls"] for result in results),
            "input_tokens": sum(
                result["usage"]["input_tokens"] for result in results
            ),
            "output_tokens": sum(
                result["usage"]["output_tokens"] for result in results
            ),
            "total_tokens": sum(
                result["usage"]["total_tokens"] for result in results
            ),
            "elapsed_seconds": round(
                sum(result["elapsed_seconds"] for result in results),
                6,
            ),
        },
        "caveat": (
            "Historical agreement and calculator alignment are regression "
            "signals. Only curated acceptable actions are quality labels."
        ),
        "results": results,
    }


def _discover_logs(sources: list[Path]) -> list[Path]:
    logs: set[Path] = set()
    for source in sources:
        if source.is_file() and source.suffix == ".jsonl":
            if _accepted_source(source):
                logs.add(source)
        elif source.is_dir():
            logs.update(
                path
                for path in source.rglob("*.jsonl")
                if _accepted_source(path)
            )
        else:
            raise ValueError(f"scenario source does not exist: {source}")
    return sorted(logs, key=lambda path: str(path))


def _accepted_source(log_path: Path) -> bool:
    attempt_path = log_path.with_suffix(".attempt.json")
    if not attempt_path.is_file():
        return True
    try:
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(attempt, dict) and attempt.get("status") == "accepted"


def _read_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid decision JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"decision row must be an object: {path}:{line_number}")
        records.append((line_number, value))
    return records


def _scenario_from_record(
    record: dict[str, Any],
    *,
    log_path: Path,
    line_number: int,
    outcome: str,
) -> dict[str, Any] | None:
    policy_input = record.get("policy_input")
    if (
        not isinstance(policy_input, dict)
        or policy_input.get("schema") != CONTROLLED_INPUT_SCHEMA
        or record.get("fallback_used")
    ):
        return None
    try:
        legal_actions = _legal_actions(policy_input)
    except ValueError:
        return None
    action_ids = [action["action_id"] for action in legal_actions]
    historical_action = str(record.get("action_id") or "")
    if historical_action not in action_ids:
        return None
    context_hash = _context_hash(policy_input)
    tags = _scenario_tags(record, legal_actions=legal_actions, outcome=outcome)
    calculator_ids = [
        action_id
        for action_id in _calculator_recommendations(record)
        if action_id in action_ids
    ]
    scenario_id = (
        f"{str(record.get('battle_id') or log_path.stem)}"
        f"-t{int(record.get('turn', 0)):02d}"
        f"-r{int(record.get('request_id', 0)):03d}"
        f"-{context_hash[-8:]}"
    )
    return {
        "id": scenario_id,
        "context_hash": context_hash,
        "source": {
            "decision_log": str(log_path),
            "line": line_number,
            "battle_id": str(record.get("battle_id") or ""),
            "turn": int(record.get("turn", 0)),
            "request_id": int(record.get("request_id", 0)),
            "outcome": outcome,
        },
        "tags": tags,
        "policy_input": policy_input,
        "historical_action_id": historical_action,
        "calculator_recommended_action_ids": calculator_ids,
        "acceptable_action_ids": [],
        "label_note": (
            "Uncurated. Historical and calculator actions are regression "
            "references only."
        ),
    }


def _select_scenarios(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -_scenario_priority(item),
            item["id"],
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    battle_counts: dict[str, int] = {}

    def add(scenario: dict[str, Any], *, enforce_battle_limit: bool) -> bool:
        battle_id = str(scenario.get("source", {}).get("battle_id") or "")
        if (
            enforce_battle_limit
            and battle_counts.get(battle_id, 0) >= 2
        ):
            return False
        selected.append(scenario)
        selected_ids.add(scenario["id"])
        battle_counts[battle_id] = battle_counts.get(battle_id, 0) + 1
        return True

    def take_matching(
        tag: str,
        maximum: int,
        *,
        exclude_tag: str = "",
    ) -> None:
        for scenario in ordered:
            if len(selected) >= limit or maximum <= 0:
                return
            if (
                scenario["id"] in selected_ids
                or tag not in scenario["tags"]
                or (exclude_tag and exclude_tag in scenario["tags"])
            ):
                continue
            if add(scenario, enforce_battle_limit=True):
                maximum -= 1

    take_matching("protocol_regression", min(6, limit))
    take_matching(
        "source_loss",
        min(8, max(0, limit - len(selected))),
        exclude_tag="forced_switch",
    )
    take_matching("forced_switch", min(2, max(0, limit - len(selected))))
    take_matching("ko_opportunity", min(4, max(0, limit - len(selected))))
    for scenario in ordered:
        if len(selected) >= limit:
            break
        if scenario["id"] not in selected_ids:
            add(scenario, enforce_battle_limit=True)
    for scenario in ordered:
        if len(selected) >= limit:
            break
        if scenario["id"] not in selected_ids:
            add(scenario, enforce_battle_limit=False)
    return selected


def _scenario_priority(scenario: dict[str, Any]) -> int:
    weights = {
        "protocol_regression": 40,
        "source_loss": 30,
        "forced_switch": 20,
        "ko_opportunity": 15,
        "plan_changed": 10,
        "mixed_move_switch": 5,
        "tera_available": 2,
    }
    return sum(weights.get(tag, 0) for tag in scenario.get("tags", []))


def _scenario_tags(
    record: dict[str, Any],
    *,
    legal_actions: list[dict[str, Any]],
    outcome: str,
) -> list[str]:
    tags = []
    if record.get("errors"):
        tags.append("protocol_regression")
    if outcome == "loss":
        tags.append("source_loss")
    kinds = {str(action.get("kind") or "") for action in legal_actions}
    if kinds == {"switch"}:
        tags.append("forced_switch")
    if "move" in kinds and "switch" in kinds:
        tags.append("mixed_move_switch")
    if any(":tera" in action["action_id"] for action in legal_actions):
        tags.append("tera_available")
    tactical = record.get("tactical_analysis")
    if isinstance(tactical, dict) and (
        tactical.get("best_ko_action_ids")
        or float(tactical.get("best_ko_probability") or 0) > 0
    ):
        tags.append("ko_opportunity")
    if record.get("plan_trigger") or record.get("plan_update"):
        tags.append("plan_changed")
    return tags


def _calculator_recommendations(record: dict[str, Any]) -> list[str]:
    tactical = record.get("tactical_analysis")
    if not isinstance(tactical, dict):
        return []
    ko = [str(value) for value in tactical.get("best_ko_action_ids") or []]
    if ko:
        return list(dict.fromkeys(ko))
    damage = [str(value) for value in tactical.get("best_damage_action_ids") or []]
    safest = {str(value) for value in tactical.get("safest_action_ids") or []}
    safe_damage = [action_id for action_id in damage if action_id in safest]
    if safe_damage:
        return list(dict.fromkeys(safe_damage))
    if damage:
        return list(dict.fromkeys(damage))
    return [
        str(value) for value in tactical.get("best_switch_action_ids") or []
    ]


def _legal_actions(policy_input: dict[str, Any]) -> list[dict[str, Any]]:
    battle = policy_input.get("battle")
    actions = battle.get("legal_actions") if isinstance(battle, dict) else None
    if not isinstance(actions, list) or not actions:
        raise ValueError("controlled scenario has no legal actions")
    normalized = []
    for action in actions:
        if not isinstance(action, dict) or not str(action.get("action_id") or ""):
            raise ValueError("controlled scenario has an invalid legal action")
        normalized.append(
            {
                **action,
                "action_id": str(action["action_id"]),
                "kind": str(action.get("kind") or ""),
            }
        )
    return normalized


def _context_hash(policy_input: dict[str, Any]) -> str:
    canonical = json.dumps(
        policy_input,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_outcome(log_path: Path) -> str:
    summary_path = log_path.with_suffix(".summary.json")
    if not summary_path.is_file():
        return "unknown"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "unknown"
    if not isinstance(summary, dict):
        return "unknown"
    if int(summary.get("agent_wins", 0)) == 1:
        return "win"
    if int(summary.get("opponent_wins", 0)) == 1:
        return "loss"
    if int(summary.get("draws", 0)) == 1:
        return "draw"
    return "unknown"


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)
