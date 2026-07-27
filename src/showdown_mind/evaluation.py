from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from showdown_mind.agent_runner import AgentSmokeResult, run_agent_battles
from showdown_mind.baselines import BASELINE_TYPES, BATTLE_FORMAT
from showdown_mind.experiment_artifacts import git_state, redact_secrets
from showdown_mind.model_runner import ModelCheckResult, run_model_check
from showdown_mind.models import ACTION_TOOL_NAME, TACTICAL_TOOL_NAME, ModelClient
from showdown_mind.policy import POLICY_MODES
from showdown_mind.policy_input import POLICY_INPUT_FORMATS

EVALUATION_SCHEMA_VERSION = "1.0"
COMPARISON_SCHEMA_VERSION = "1.0"
EVALUATION_STATE_SCHEMA_VERSION = "resumable-evaluation-v1"
ATTEMPT_SCHEMA_VERSION = "evaluation-attempt-v1"
DEFAULT_OPPONENTS = ("max-base-power", "simple-heuristics")
MIN_COMPARISON_BATTLES = 20
BOOTSTRAP_ITERATIONS = 5_000
FINAL_QUALITY_THRESHOLDS = {
    "max_fallback_rate": 0.05,
    "max_decision_error_rate": 0.10,
    "min_tool_call_coverage": 0.95,
    "min_tactical_tool_coverage": 0.95,
    "min_rationale_coverage": 0.95,
    "min_battle_plan_coverage": 0.95,
    "min_prediction_coverage": 0.95,
    "max_planner_failure_rate": 0.10,
    "max_enrichment_error_rate": 0.05,
}
HARD_STOP_THRESHOLDS = {
    "max_fallback_rate": 0.20,
    "max_decision_error_rate": 0.30,
    "min_tool_call_coverage": 0.70,
    "min_tactical_tool_coverage": 0.70,
    "min_rationale_coverage": 0.70,
    "min_battle_plan_coverage": 0.70,
    "min_prediction_coverage": 0.70,
    "max_planner_failure_rate": 0.30,
    "max_enrichment_error_rate": 0.20,
}

BattleRunner = Callable[..., Awaitable[AgentSmokeResult]]
ConnectivityChecker = Callable[..., Awaitable[ModelCheckResult]]


class EvaluationError(RuntimeError):
    """Raised when an evaluation cannot produce a valid complete report."""


class EvaluationQualityError(EvaluationError):
    """Raised when a run crosses the live cost-protection quality gate."""


@dataclass(frozen=True)
class EvaluationPlan:
    name: str
    output_dir: Path
    opponents: tuple[str, ...] = DEFAULT_OPPONENTS
    battles_per_opponent: int = 10
    repeats: int = 1
    prompt_format: str = "pruned"
    policy_mode: str = "direct"
    run_timeout_seconds: float | None = None
    stop_on_quality_failure: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("evaluation name must not be empty")
        if not self.opponents:
            raise ValueError("evaluation must include at least one opponent")
        if len(set(self.opponents)) != len(self.opponents):
            raise ValueError("evaluation opponents must be unique")
        unknown = sorted(set(self.opponents).difference(BASELINE_TYPES))
        if unknown:
            raise ValueError(f"unknown evaluation opponents: {', '.join(unknown)}")
        if self.battles_per_opponent < 1:
            raise ValueError("battles_per_opponent must be at least 1")
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        if self.prompt_format not in POLICY_INPUT_FORMATS:
            raise ValueError(f"unknown prompt format: {self.prompt_format}")
        if self.policy_mode not in POLICY_MODES:
            raise ValueError(f"unknown policy mode: {self.policy_mode}")
        if self.run_timeout_seconds is not None and self.run_timeout_seconds <= 0:
            raise ValueError("run_timeout_seconds must be positive")

    @property
    def total_runs(self) -> int:
        return self.total_battles

    @property
    def total_battles(self) -> int:
        return (
            len(self.opponents)
            * self.repeats
            * self.battles_per_opponent
        )

    @property
    def effective_run_timeout_seconds(self) -> float:
        return self.run_timeout_seconds or 300.0

    def matrix(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "opponent": opponent,
                "repeat": repeat,
                "battle_index": battle_index,
                "checkpoint_id": _checkpoint_id(
                    opponent,
                    repeat,
                    battle_index,
                ),
            }
            for opponent in self.opponents
            for repeat in range(1, self.repeats + 1)
            for battle_index in range(1, self.battles_per_opponent + 1)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "battle_format": BATTLE_FORMAT,
            "output_dir": str(self.output_dir),
            "opponents": list(self.opponents),
            "battles_per_opponent": self.battles_per_opponent,
            "repeats": self.repeats,
            "total_runs": self.total_runs,
            "total_battles": self.total_battles,
            "prompt_format": f"{self.prompt_format}-v1",
            "policy_mode": self.policy_mode,
            "run_timeout_seconds": self.effective_run_timeout_seconds,
            "stop_on_quality_failure": self.stop_on_quality_failure,
            "quality_thresholds": FINAL_QUALITY_THRESHOLDS,
            "hard_stop_thresholds": HARD_STOP_THRESHOLDS,
            "checkpointing": EVALUATION_STATE_SCHEMA_VERSION,
            "resume_supported": True,
        }

    def preview(self) -> dict[str, Any]:
        return {
            "mode": "dry-run",
            "will_call_model": False,
            "plan": self.to_dict(),
            "matrix": list(self.matrix()),
            "warning": (
                "No files or API calls were made. Add --run to execute; "
                "exact token cost is unknown."
            ),
        }


async def run_evaluation(
    model_client: ModelClient,
    plan: EvaluationPlan,
    *,
    battle_runner: BattleRunner = run_agent_battles,
    connectivity_checker: ConnectivityChecker = run_model_check,
    resume: bool = False,
) -> dict[str, Any]:
    state_path = plan.output_dir / "evaluation-state.json"
    if resume:
        state = _load_resume_state(
            state_path,
            plan=plan,
            model_client=model_client,
        )
    else:
        if plan.output_dir.exists():
            raise ValueError(
                f"evaluation output already exists: {plan.output_dir}; "
                "use --resume to continue it"
            )
        plan.output_dir.mkdir(parents=True)
        git_commit, git_dirty = git_state()
        state = {
            "schema_version": EVALUATION_STATE_SCHEMA_VERSION,
            "status": "created",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "plan": plan.to_dict(),
            "provenance": {
                "model_id": str(
                    getattr(
                        model_client,
                        "model_id",
                        type(model_client).__name__,
                    )
                ),
                "git_commit": git_commit,
                "git_dirty": git_dirty,
            },
            "preflight": None,
        }
        _write_json(state_path, state)
        _write_json(plan.output_dir / "plan.json", plan.to_dict())

    runs_dir = plan.output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    created_at = str(state["created_at"])
    provenance = dict(state["provenance"])
    attempts = _recover_attempts(
        runs_dir,
        plan=plan,
    )
    runs = _accepted_runs(plan, attempts)
    preflight = state.get("preflight")
    try:
        if preflight is None:
            check = await connectivity_checker(
                model_client,
                prompt_format=plan.prompt_format,
                policy_mode=plan.policy_mode,
            )
            preflight = check.to_dict()
            state["preflight"] = preflight
            _update_evaluation_state(state_path, state, status="in_progress")
        _write_progress_report(
            plan=plan,
            status="in_progress",
            created_at=created_at,
            provenance=provenance,
            preflight=preflight,
            runs=runs,
            attempts=attempts,
            failure=None,
        )

        accepted_ids = {
            str(run["checkpoint_id"])
            for run in runs
        }
        for cell in plan.matrix():
            checkpoint_id = str(cell["checkpoint_id"])
            if checkpoint_id in accepted_ids:
                continue
            attempt_number = _next_attempt_number(attempts, checkpoint_id)
            attempt_id = f"{checkpoint_id}-a{attempt_number:02d}"
            decision_log = runs_dir / f"{attempt_id}.jsonl"
            attempt_path = runs_dir / f"{attempt_id}.attempt.json"
            attempt_record = {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "checkpoint_id": checkpoint_id,
                "opponent": cell["opponent"],
                "repeat": cell["repeat"],
                "battle_index": cell["battle_index"],
                "attempt": attempt_number,
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "decision_log": str(decision_log),
            }
            _write_json(attempt_path, attempt_record)
            attempt_finalized = False
            try:
                result = await battle_runner(
                    model_client,
                    opponent_name=str(cell["opponent"]),
                    battles=1,
                    decision_log=decision_log,
                    timeout_seconds=plan.effective_run_timeout_seconds,
                    prompt_format=plan.prompt_format,
                    policy_mode=plan.policy_mode,
                )
                completed_run = _completed_run(
                    opponent=str(cell["opponent"]),
                    repeat=int(cell["repeat"]),
                    battle_index=int(cell["battle_index"]),
                    attempt=attempt_number,
                    result=result,
                    decision_log=decision_log,
                )
                hard_quality = assess_quality(
                    aggregate_runs([completed_run]),
                    thresholds=HARD_STOP_THRESHOLDS,
                    policy_mode=plan.policy_mode,
                )
                completed_run["quality"] = hard_quality
                accepted = not (
                    plan.stop_on_quality_failure
                    and hard_quality["status"] == "invalid"
                )
                attempt_record.update(
                    {
                        "status": "accepted" if accepted else "rejected",
                        "updated_at": datetime.now(UTC).isoformat(),
                        "finished_at": datetime.now(UTC).isoformat(),
                        "run": completed_run,
                    }
                )
                _write_json(attempt_path, attempt_record)
                attempt_finalized = True
                attempts.append(attempt_record)
                if not accepted:
                    details = "; ".join(hard_quality["violations"])
                    raise EvaluationQualityError(
                        f"checkpoint {checkpoint_id} crossed the quality "
                        f"stop gate: {details}"
                    )
                runs.append(completed_run)
                accepted_ids.add(checkpoint_id)
                _write_progress_report(
                    plan=plan,
                    status="in_progress",
                    created_at=created_at,
                    provenance=provenance,
                    preflight=preflight,
                    runs=runs,
                    attempts=attempts,
                    failure=None,
                )
            except Exception as exc:
                if not attempt_finalized:
                    attempt_record.update(
                        {
                            "status": "interrupted",
                            "updated_at": datetime.now(UTC).isoformat(),
                            "finished_at": datetime.now(UTC).isoformat(),
                            "failure": {
                                "error_type": type(exc).__name__,
                                "message": _failure_detail(exc),
                            },
                            "decision_metrics": _partial_decision_metrics(
                                decision_log
                            ),
                        }
                    )
                    _write_json(attempt_path, attempt_record)
                    attempts.append(attempt_record)
                raise
    except Exception as exc:
        safe_message = _failure_detail(exc)
        _update_evaluation_state(state_path, state, status="incomplete")
        report = _write_progress_report(
            plan=plan,
            status="incomplete",
            created_at=created_at,
            provenance=provenance,
            preflight=preflight,
            runs=runs,
            attempts=attempts,
            failure={
                "error_type": type(exc).__name__,
                "message": safe_message,
            },
        )
        raise EvaluationError(
            f"evaluation stopped after {len(runs)} accepted battles; "
            f"resume with --resume: {safe_message}"
        ) from exc

    _update_evaluation_state(state_path, state, status="complete")
    report = _write_progress_report(
        plan=plan,
        status="complete",
        created_at=created_at,
        provenance=provenance,
        preflight=preflight,
        runs=runs,
        attempts=attempts,
        failure=None,
    )
    return report


def _completed_run(
    *,
    opponent: str,
    repeat: int,
    battle_index: int,
    attempt: int,
    result: AgentSmokeResult,
    decision_log: Path,
) -> dict[str, Any]:
    if result.requested_battles != 1 or result.finished_battles != 1:
        raise ValueError(
            "resumable evaluation checkpoints must contain exactly one battle"
        )
    outcomes = (
        ["win"] * result.agent_wins
        + ["loss"] * result.opponent_wins
        + ["draw"] * result.draws
    )
    if len(outcomes) != 1:
        raise ValueError("checkpoint result must contain exactly one outcome")
    return {
        "status": "complete",
        "opponent": opponent,
        "repeat": repeat,
        "battle_index": battle_index,
        "attempt": attempt,
        "checkpoint_id": _checkpoint_id(opponent, repeat, battle_index),
        "outcomes": outcomes,
        "result": result.to_dict(),
        "decision_metrics": summarize_decision_log(decision_log),
    }


def _checkpoint_id(opponent: str, repeat: int, battle_index: int) -> str:
    return f"{opponent}-r{repeat:02d}-b{battle_index:03d}"


def _load_resume_state(
    path: Path,
    *,
    plan: EvaluationPlan,
    model_client: ModelClient,
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            f"evaluation cannot be resumed because state is missing: {path}"
        )
    state = _read_json_object(path)
    if state.get("schema_version") != EVALUATION_STATE_SCHEMA_VERSION:
        raise ValueError("evaluation uses an incompatible checkpoint schema")
    if state.get("plan") != plan.to_dict():
        raise ValueError(
            "resume plan does not exactly match the saved evaluation plan"
        )
    git_commit, git_dirty = git_state()
    current = {
        "model_id": str(
            getattr(model_client, "model_id", type(model_client).__name__)
        ),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    saved = state.get("provenance")
    if saved != current:
        raise ValueError(
            "resume provenance does not match the saved model, Git commit, "
            "or dirty state"
        )
    return state


def _recover_attempts(
    runs_dir: Path,
    *,
    plan: EvaluationPlan,
) -> list[dict[str, Any]]:
    allowed = {
        str(cell["checkpoint_id"])
        for cell in plan.matrix()
    }
    attempts: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.attempt.json")):
        attempt = _read_json_object(path)
        if attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
            raise ValueError(f"incompatible attempt record: {path}")
        checkpoint_id = str(attempt.get("checkpoint_id") or "")
        if checkpoint_id not in allowed:
            raise ValueError(
                f"attempt does not belong to the saved matrix: {path}"
            )
        if attempt.get("status") == "running":
            attempt = _recover_running_attempt(
                path,
                attempt,
                plan=plan,
            )
        attempts.append(attempt)
    return attempts


def _recover_running_attempt(
    path: Path,
    attempt: dict[str, Any],
    *,
    plan: EvaluationPlan,
) -> dict[str, Any]:
    attempt_id = str(attempt["attempt_id"])
    summary_path = path.with_name(f"{attempt_id}.summary.json")
    decision_log = path.with_name(f"{attempt_id}.jsonl")
    if summary_path.is_file() and decision_log.is_file():
        result = _agent_result_from_summary(summary_path)
        completed_run = _completed_run(
            opponent=str(attempt["opponent"]),
            repeat=int(attempt["repeat"]),
            battle_index=int(attempt["battle_index"]),
            attempt=int(attempt["attempt"]),
            result=result,
            decision_log=decision_log,
        )
        hard_quality = assess_quality(
            aggregate_runs([completed_run]),
            thresholds=HARD_STOP_THRESHOLDS,
            policy_mode=plan.policy_mode,
        )
        completed_run["quality"] = hard_quality
        accepted = not (
            plan.stop_on_quality_failure
            and hard_quality["status"] == "invalid"
        )
        attempt.update(
            {
                "status": "accepted" if accepted else "rejected",
                "updated_at": datetime.now(UTC).isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "run": completed_run,
                "recovered_after_process_exit": True,
            }
        )
    else:
        attempt.update(
            {
                "status": "abandoned",
                "updated_at": datetime.now(UTC).isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "failure": {
                    "error_type": "ProcessInterrupted",
                    "message": (
                        "The process ended before this battle produced a "
                        "complete summary."
                    ),
                },
                "decision_metrics": _partial_decision_metrics(decision_log),
            }
        )
    _write_json(path, attempt)
    return attempt


def _agent_result_from_summary(path: Path) -> AgentSmokeResult:
    value = _read_json_object(path)
    allowed = AgentSmokeResult.__dataclass_fields__
    arguments = {
        key: value[key]
        for key in allowed
        if key in value
    }
    return AgentSmokeResult(**arguments)


def _accepted_runs(
    plan: EvaluationPlan,
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        if attempt.get("status") != "accepted":
            continue
        checkpoint_id = str(attempt["checkpoint_id"])
        if checkpoint_id in accepted:
            raise ValueError(
                f"multiple accepted attempts exist for {checkpoint_id}"
            )
        run = attempt.get("run")
        if not isinstance(run, dict):
            raise ValueError(
                f"accepted attempt is missing its run: {checkpoint_id}"
            )
        accepted[checkpoint_id] = run
    return [
        accepted[str(cell["checkpoint_id"])]
        for cell in plan.matrix()
        if str(cell["checkpoint_id"]) in accepted
    ]


def _next_attempt_number(
    attempts: list[dict[str, Any]],
    checkpoint_id: str,
) -> int:
    prior = [
        int(attempt.get("attempt", 0))
        for attempt in attempts
        if attempt.get("checkpoint_id") == checkpoint_id
    ]
    return max(prior, default=0) + 1


def _partial_decision_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return summarize_decision_log(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _update_evaluation_state(
    path: Path,
    state: dict[str, Any],
    *,
    status: str,
) -> None:
    state["status"] = status
    state["updated_at"] = datetime.now(UTC).isoformat()
    _write_json(path, state)


def _write_progress_report(
    *,
    plan: EvaluationPlan,
    status: str,
    created_at: str,
    provenance: dict[str, Any],
    preflight: dict[str, Any] | None,
    runs: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    failure: dict[str, str] | None,
) -> dict[str, Any]:
    report = _evaluation_report(
        plan=plan,
        status=status,
        created_at=created_at,
        provenance=provenance,
        preflight=preflight,
        runs=runs,
        attempts=attempts,
        failure=failure,
    )
    _write_evaluation_report(plan.output_dir, report)
    return report


def summarize_decision_log(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"decision log line {line_number} is not an object")
        records.append(value)

    decisions = len(records)
    retries = 0
    decisions_with_retry = 0
    model_calls = 0
    expected_model_calls = 0
    fallbacks = sum(bool(record.get("fallback_used")) for record in records)
    decisions_with_errors = sum(bool(record.get("errors")) for record in records)
    transport_errors = 0
    validation_errors = 0
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    latency = 0.0
    confidences: list[float] = []
    tool_calls = 0
    tactical_tool_calls = 0
    rationales = 0
    battle_plan_decisions = 0
    replan_decisions = 0
    planner_model_calls = 0
    planner_error_decisions = 0
    planner_failure_decisions = 0
    planner_recovered_error_decisions = 0
    planner_input_tokens = 0
    planner_output_tokens = 0
    planner_total_tokens = 0
    planner_latency = 0.0
    enrichment_error_decisions = 0
    prediction_decisions = 0
    prediction_resolutions: dict[tuple[Any, ...], bool] = {}
    reason_counts: Counter[str] = Counter()
    normalized_decisions = 0
    decision_normalizations = 0
    policy_input_characters = 0
    max_policy_input_characters = 0

    for record in records:
        expected_calls = int(record.get("expected_model_calls", 1))
        actual_calls = int(
            record.get("model_calls")
            or record.get("attempts", 0)
        )
        model_calls += actual_calls
        expected_model_calls += expected_calls
        retry_calls = max(0, int(record.get("attempts", 0)) - 1)
        retries += retry_calls
        decisions_with_retry += retry_calls > 0
        errors = [str(error) for error in record.get("errors") or []]
        for error in errors:
            if "ModelCallError" in error or "TimeoutError" in error:
                transport_errors += 1
            else:
                validation_errors += 1
        for usage in record.get("usages") or []:
            if not isinstance(usage, dict):
                continue
            input_tokens += int(usage.get("input_tokens", 0))
            output_tokens += int(usage.get("output_tokens", 0))
            total_tokens += int(usage.get("total_tokens", 0))
        latency += float(record.get("elapsed_seconds", 0.0))
        confidence = record.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            confidences.append(float(confidence))
        tool_names = [str(name) for name in record.get("tool_names") or []]
        if not tool_names and record.get("tool_call_ids"):
            tool_names = [str(record.get("tool_name") or ACTION_TOOL_NAME)]
        tool_calls += ACTION_TOOL_NAME in tool_names
        executions = [
            execution
            for execution in record.get("tool_executions") or []
            if isinstance(execution, dict)
        ]
        tactical_tool_calls += any(
            execution.get("tool_name") == TACTICAL_TOOL_NAME
            for execution in executions
        )
        rationales += bool(str(record.get("short_rationale") or "").strip())
        battle_plan_decisions += bool(record.get("battle_plan"))
        replan_decisions += bool(str(record.get("plan_trigger") or ""))
        planner_model_calls += int(record.get("planner_model_calls", 0))
        has_planner_errors = bool(record.get("planner_errors"))
        planner_failed = bool(record.get("planner_failed", False))
        planner_error_decisions += has_planner_errors
        planner_failure_decisions += planner_failed
        planner_recovered_error_decisions += (
            has_planner_errors and not planner_failed
        )
        planner_latency += float(record.get("planner_elapsed_seconds", 0.0))
        enrichment_error_decisions += bool(record.get("enrichment_errors"))
        for usage in record.get("planner_usages") or []:
            if not isinstance(usage, dict):
                continue
            planner_input_tokens += int(usage.get("input_tokens", 0))
            planner_output_tokens += int(usage.get("output_tokens", 0))
            planner_total_tokens += int(usage.get("total_tokens", 0))
        prediction_decisions += bool(record.get("opponent_prediction"))
        memory = record.get("memory")
        resolution = (
            memory.get("previous_prediction_resolution")
            if isinstance(memory, dict)
            else None
        )
        if isinstance(resolution, dict):
            predicted = resolution.get("predicted")
            predicted_turn = (
                predicted.get("decision_turn")
                if isinstance(predicted, dict)
                else None
            )
            resolution_key = (
                predicted_turn,
                resolution.get("observed_turn"),
                resolution.get("evidence_event_id"),
            )
            prediction_resolutions[resolution_key] = bool(
                resolution.get("matched")
            )
        reason_counts.update(str(code) for code in record.get("reason_codes") or [])
        normalizations = [
            str(value)
            for value in record.get("decision_normalizations") or []
        ]
        normalized_decisions += bool(normalizations)
        decision_normalizations += len(normalizations)
        characters = int(record.get("policy_input_characters", 0))
        policy_input_characters += characters
        max_policy_input_characters = max(
            max_policy_input_characters,
            characters,
        )

    return {
        "decisions": decisions,
        "retry_calls": retries,
        "decisions_with_retry": decisions_with_retry,
        "model_calls": model_calls,
        "expected_model_calls": expected_model_calls,
        "fallbacks": fallbacks,
        "decisions_with_errors": decisions_with_errors,
        "transport_errors": transport_errors,
        "validation_errors": validation_errors,
        "tool_call_decisions": tool_calls,
        "tactical_tool_decisions": tactical_tool_calls,
        "rationale_decisions": rationales,
        "battle_plan_decisions": battle_plan_decisions,
        "replan_decisions": replan_decisions,
        "planner_model_calls": planner_model_calls,
        "planner_error_decisions": planner_error_decisions,
        "planner_failure_decisions": planner_failure_decisions,
        "planner_recovered_error_decisions": (
            planner_recovered_error_decisions
        ),
        "planner_input_tokens": planner_input_tokens,
        "planner_output_tokens": planner_output_tokens,
        "planner_total_tokens": planner_total_tokens,
        "planner_latency_seconds": round(planner_latency, 6),
        "enrichment_error_decisions": enrichment_error_decisions,
        "normalized_decisions": normalized_decisions,
        "decision_normalizations": decision_normalizations,
        "policy_input_characters": policy_input_characters,
        "max_policy_input_characters": max_policy_input_characters,
        "prediction_decisions": prediction_decisions,
        "prediction_resolutions": len(prediction_resolutions),
        "prediction_matches": sum(prediction_resolutions.values()),
        "confidence_sum": round(sum(confidences), 6),
        "confidence_count": len(confidences),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "decision_latency_seconds": round(latency, 6),
        "reason_code_counts": dict(sorted(reason_counts.items())),
    }


def _evaluation_report(
    *,
    plan: EvaluationPlan,
    status: str,
    created_at: str,
    provenance: dict[str, Any],
    preflight: dict[str, Any] | None,
    runs: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    failure: dict[str, str] | None,
) -> dict[str, Any]:
    overall = aggregate_runs(runs)
    preflight_tokens = int((preflight or {}).get("total_tokens", 0))
    excluded_attempts = [
        attempt
        for attempt in attempts
        if attempt.get("status") != "accepted"
    ]
    excluded_tokens = sum(
        int(_attempt_metrics(attempt).get("total_tokens", 0))
        for attempt in excluded_attempts
    )
    quality = assess_quality(overall, policy_mode=plan.policy_mode)
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": status,
        "name": plan.name,
        "created_at": created_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "finished_at": (
            datetime.now(UTC).isoformat()
            if status == "complete"
            else None
        ),
        "plan": plan.to_dict(),
        "provenance": provenance,
        "preflight": preflight,
        "runs": runs,
        "attempts": [_attempt_report_view(attempt) for attempt in attempts],
        "progress": {
            "target_battles": plan.total_battles,
            "accepted_battles": len(runs),
            "remaining_battles": max(0, plan.total_battles - len(runs)),
            "attempts": len(attempts),
            "excluded_attempts": len(excluded_attempts),
            "resume_supported": True,
        },
        "overall": overall,
        "quality": quality,
        "cost_accounting": {
            "battle_total_tokens": overall["total_tokens"],
            "preflight_total_tokens": preflight_tokens,
            "excluded_attempt_total_tokens": excluded_tokens,
            "evaluation_total_tokens": (
                overall["total_tokens"]
                + preflight_tokens
                + excluded_tokens
            ),
        },
        "by_opponent": {
            opponent: aggregate_runs(
                [run for run in runs if run["opponent"] == opponent]
            )
            for opponent in plan.opponents
        },
    }
    if failure is not None:
        report["failure"] = failure
    return report


def _attempt_metrics(attempt: dict[str, Any]) -> dict[str, Any]:
    run = attempt.get("run")
    if isinstance(run, dict):
        metrics = run.get("decision_metrics")
        if isinstance(metrics, dict):
            return metrics
    metrics = attempt.get("decision_metrics")
    return metrics if isinstance(metrics, dict) else {}


def _attempt_report_view(attempt: dict[str, Any]) -> dict[str, Any]:
    metrics = _attempt_metrics(attempt)
    value = {
        key: attempt[key]
        for key in (
            "attempt_id",
            "checkpoint_id",
            "opponent",
            "repeat",
            "battle_index",
            "attempt",
            "status",
            "started_at",
            "finished_at",
            "decision_log",
            "recovered_after_process_exit",
            "failure",
        )
        if key in attempt
    }
    value["decisions"] = int(metrics.get("decisions", 0))
    value["total_tokens"] = int(metrics.get("total_tokens", 0))
    run = attempt.get("run")
    if isinstance(run, dict):
        value["outcomes"] = list(run.get("outcomes") or [])
        value["quality"] = run.get("quality", {})
    return value


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(run["outcomes"].count("win") for run in runs)
    losses = sum(run["outcomes"].count("loss") for run in runs)
    draws = sum(run["outcomes"].count("draw") for run in runs)
    battles = wins + losses + draws
    decisions = sum(run["decision_metrics"]["decisions"] for run in runs)
    totals: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for run in runs:
        metrics = run["decision_metrics"]
        for key in (
            "retry_calls",
            "decisions_with_retry",
            "model_calls",
            "expected_model_calls",
            "fallbacks",
            "decisions_with_errors",
            "transport_errors",
            "validation_errors",
            "tool_call_decisions",
            "tactical_tool_decisions",
            "rationale_decisions",
            "battle_plan_decisions",
            "replan_decisions",
            "planner_model_calls",
            "planner_error_decisions",
            "planner_failure_decisions",
            "planner_recovered_error_decisions",
            "planner_input_tokens",
            "planner_output_tokens",
            "planner_total_tokens",
            "enrichment_error_decisions",
            "normalized_decisions",
            "decision_normalizations",
            "policy_input_characters",
            "prediction_decisions",
            "prediction_resolutions",
            "prediction_matches",
            "confidence_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            totals[key] += metrics.get(key, 0)
        reason_counts.update(metrics["reason_code_counts"])

    confidence_sum = sum(run["decision_metrics"]["confidence_sum"] for run in runs)
    latency = sum(run["decision_metrics"]["decision_latency_seconds"] for run in runs)
    planner_latency = sum(
        run["decision_metrics"].get("planner_latency_seconds", 0.0)
        for run in runs
    )
    wall_seconds = sum(float(run["result"].get("elapsed_seconds", 0.0)) for run in runs)
    max_policy_input_characters = max(
        (
            int(run["decision_metrics"].get("max_policy_input_characters", 0))
            for run in runs
        ),
        default=0,
    )
    action_input_tokens = max(
        0,
        totals["input_tokens"] - totals["planner_input_tokens"],
    )
    action_output_tokens = max(
        0,
        totals["output_tokens"] - totals["planner_output_tokens"],
    )
    action_total_tokens = max(
        0,
        totals["total_tokens"] - totals["planner_total_tokens"],
    )
    score_rate = _rate(wins + 0.5 * draws, battles)
    return {
        "battles": battles,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": _rate(wins, battles),
        "score_rate": score_rate,
        "score_rate_ci95": wilson_interval(wins + 0.5 * draws, battles),
        "decisions": decisions,
        "retry_calls": totals["retry_calls"],
        "retry_rate": _rate(totals["decisions_with_retry"], decisions),
        "model_calls": totals["model_calls"],
        "expected_model_calls": totals["expected_model_calls"],
        "average_model_calls_per_decision": _rate(
            totals["model_calls"],
            decisions,
        ),
        "fallbacks": totals["fallbacks"],
        "fallback_rate": _rate(totals["fallbacks"], decisions),
        "decisions_with_errors": totals["decisions_with_errors"],
        "decision_error_rate": _rate(
            totals["decisions_with_errors"],
            decisions,
        ),
        "transport_errors": totals["transport_errors"],
        "validation_errors": totals["validation_errors"],
        "tool_call_coverage": _rate(totals["tool_call_decisions"], decisions),
        "tactical_tool_coverage": _rate(
            totals["tactical_tool_decisions"],
            decisions,
        ),
        "rationale_coverage": _rate(totals["rationale_decisions"], decisions),
        "battle_plan_coverage": _rate(
            totals["battle_plan_decisions"],
            decisions,
        ),
        "replan_decisions": totals["replan_decisions"],
        "replan_rate": _rate(totals["replan_decisions"], decisions),
        "planner_model_calls": totals["planner_model_calls"],
        "planner_error_decisions": totals["planner_error_decisions"],
        "planner_error_rate": _rate(
            totals["planner_error_decisions"],
            totals["replan_decisions"],
        ),
        "planner_failure_decisions": totals["planner_failure_decisions"],
        "planner_failure_rate": _rate(
            totals["planner_failure_decisions"],
            totals["replan_decisions"],
        ),
        "planner_recovered_error_decisions": totals[
            "planner_recovered_error_decisions"
        ],
        "planner_recovered_error_rate": _rate(
            totals["planner_recovered_error_decisions"],
            totals["replan_decisions"],
        ),
        "enrichment_error_decisions": totals["enrichment_error_decisions"],
        "enrichment_error_rate": _rate(
            totals["enrichment_error_decisions"],
            decisions,
        ),
        "normalized_decisions": totals["normalized_decisions"],
        "normalization_rate": _rate(
            totals["normalized_decisions"],
            decisions,
        ),
        "decision_normalizations": totals["decision_normalizations"],
        "policy_input_characters": totals["policy_input_characters"],
        "average_policy_input_characters": _rate(
            totals["policy_input_characters"],
            decisions,
        ),
        "max_policy_input_characters": max_policy_input_characters,
        "planner_input_tokens": totals["planner_input_tokens"],
        "planner_output_tokens": totals["planner_output_tokens"],
        "planner_total_tokens": totals["planner_total_tokens"],
        "average_planner_tokens_per_replan": _rate(
            totals["planner_total_tokens"],
            totals["replan_decisions"],
        ),
        "action_input_tokens": action_input_tokens,
        "action_output_tokens": action_output_tokens,
        "action_total_tokens": action_total_tokens,
        "average_action_tokens_per_decision": _rate(
            action_total_tokens,
            decisions,
        ),
        "prediction_coverage": _rate(
            totals["prediction_decisions"],
            decisions,
        ),
        "prediction_resolutions": totals["prediction_resolutions"],
        "prediction_matches": totals["prediction_matches"],
        "prediction_accuracy": _rate(
            totals["prediction_matches"],
            totals["prediction_resolutions"],
        ),
        "average_confidence": _rate(
            confidence_sum,
            totals["confidence_count"],
        ),
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "total_tokens": totals["total_tokens"],
        "average_tokens_per_battle": _rate(totals["total_tokens"], battles),
        "average_tokens_per_decision": _rate(totals["total_tokens"], decisions),
        "decision_latency_seconds": round(latency, 6),
        "average_decision_latency_seconds": _rate(latency, decisions),
        "planner_latency_seconds": round(planner_latency, 6),
        "average_planner_latency_seconds": _rate(
            planner_latency,
            totals["replan_decisions"],
        ),
        "agent_latency_seconds": round(latency + planner_latency, 6),
        "average_agent_latency_seconds": _rate(
            latency + planner_latency,
            decisions,
        ),
        "wall_seconds": round(wall_seconds, 6),
        "average_wall_seconds_per_battle": _rate(wall_seconds, battles),
        "reason_code_counts": dict(sorted(reason_counts.items())),
    }


def assess_quality(
    metrics: dict[str, Any],
    *,
    thresholds: dict[str, float] = FINAL_QUALITY_THRESHOLDS,
    policy_mode: str = "direct",
) -> dict[str, Any]:
    decisions = int(metrics.get("decisions", 0))
    if decisions == 0:
        return {
            "status": "pending",
            "thresholds": dict(thresholds),
            "violations": [],
        }
    checks = (
        (
            "fallback_rate",
            float(metrics["fallback_rate"]),
            "<=",
            thresholds["max_fallback_rate"],
        ),
        (
            "decision_error_rate",
            float(metrics["decision_error_rate"]),
            "<=",
            thresholds["max_decision_error_rate"],
        ),
        (
            "tool_call_coverage",
            float(metrics["tool_call_coverage"]),
            ">=",
            thresholds["min_tool_call_coverage"],
        ),
        (
            "rationale_coverage",
            float(metrics["rationale_coverage"]),
            ">=",
            thresholds["min_rationale_coverage"],
        ),
    )
    if policy_mode in {"tactical-tool", "controlled-agent"}:
        checks += (
            (
                "tactical_tool_coverage",
                float(metrics.get("tactical_tool_coverage", 0.0)),
                ">=",
                thresholds["min_tactical_tool_coverage"],
            ),
        )
    if policy_mode == "controlled-agent":
        checks += (
            (
                "battle_plan_coverage",
                float(metrics.get("battle_plan_coverage", 0.0)),
                ">=",
                thresholds["min_battle_plan_coverage"],
            ),
            (
                "prediction_coverage",
                float(metrics.get("prediction_coverage", 0.0)),
                ">=",
                thresholds["min_prediction_coverage"],
            ),
            (
                "planner_failure_rate",
                float(metrics.get("planner_failure_rate", 0.0)),
                "<=",
                thresholds["max_planner_failure_rate"],
            ),
            (
                "enrichment_error_rate",
                float(metrics.get("enrichment_error_rate", 0.0)),
                "<=",
                thresholds["max_enrichment_error_rate"],
            ),
        )
    violations = []
    for name, actual, operator, expected in checks:
        failed = actual > expected if operator == "<=" else actual < expected
        if failed:
            violations.append(f"{name}={actual:.2%} must be {operator} {expected:.2%}")
    return {
        "status": "invalid" if violations else "valid",
        "thresholds": dict(thresholds),
        "violations": violations,
    }


def wilson_interval(successes: float, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def compare_evaluation_reports(
    baseline_path: Path,
    candidate_path: Path,
    *,
    output_path: Path,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    if output_path.suffix != ".json":
        raise ValueError("comparison output must use a .json suffix")
    if output_path.exists() or output_path.with_suffix(".md").exists():
        raise ValueError(f"comparison output already exists: {output_path}")
    baseline = _read_complete_report(baseline_path)
    candidate = _read_complete_report(candidate_path)
    comparison = build_comparison(
        baseline,
        candidate,
        bootstrap_iterations=bootstrap_iterations,
    )
    _write_json(output_path, comparison)
    _write_text(output_path.with_suffix(".md"), render_comparison_markdown(comparison))
    return comparison


def build_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    if bootstrap_iterations < 100:
        raise ValueError("bootstrap_iterations must be at least 100")
    baseline_opponents = set(baseline["plan"]["opponents"])
    candidate_opponents = set(candidate["plan"]["opponents"])
    if baseline["plan"]["battle_format"] != candidate["plan"]["battle_format"]:
        raise ValueError("evaluation battle formats do not match")
    if baseline_opponents != candidate_opponents:
        raise ValueError("evaluation opponent sets do not match")
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        quality = report.get("quality", {})
        if quality.get("status") != "valid":
            raise ValueError(f"{label} evaluation quality is not valid")

    bootstrap = _stratified_bootstrap(
        baseline,
        candidate,
        sorted(baseline_opponents),
        iterations=bootstrap_iterations,
    )
    baseline_metrics = baseline["overall"]
    candidate_metrics = candidate["overall"]
    minimum_battles = min(
        int(baseline_metrics["battles"]),
        int(candidate_metrics["battles"]),
    )
    low, high = bootstrap["delta_ci95"]
    if minimum_battles < MIN_COMPARISON_BATTLES:
        conclusion = "insufficient_data"
    elif low > 0:
        conclusion = "improved"
    elif high < 0:
        conclusion = "regressed"
    else:
        conclusion = "inconclusive"

    tracked = (
        "score_rate",
        "win_rate",
        "fallback_rate",
        "retry_rate",
        "decision_error_rate",
        "tool_call_coverage",
        "rationale_coverage",
        "average_tokens_per_battle",
        "average_tokens_per_decision",
        "average_action_tokens_per_decision",
        "average_policy_input_characters",
        "average_decision_latency_seconds",
    )
    deltas = {
        key: _difference(candidate_metrics.get(key), baseline_metrics.get(key))
        for key in tracked
    }
    plan_differences = _plan_differences(baseline, candidate)
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "baseline": {
            "name": baseline["name"],
            "path": str(baseline.get("_source_path", "")),
            "provenance": baseline["provenance"],
            "overall": baseline_metrics,
        },
        "candidate": {
            "name": candidate["name"],
            "path": str(candidate.get("_source_path", "")),
            "provenance": candidate["provenance"],
            "overall": candidate_metrics,
        },
        "comparability": {
            "battle_format_matches": True,
            "opponent_set_matches": True,
            "declared_differences": plan_differences,
        },
        "primary_outcome": {
            "metric": "score_rate",
            "delta": _difference(
                candidate_metrics["score_rate"],
                baseline_metrics["score_rate"],
            ),
            **bootstrap,
        },
        "metric_deltas": deltas,
        "conclusion": conclusion,
        "minimum_battles_required": MIN_COMPARISON_BATTLES,
    }


def _stratified_bootstrap(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    opponents: list[str],
    *,
    iterations: int,
) -> dict[str, Any]:
    baseline_values = _outcomes_by_opponent(baseline, opponents)
    candidate_values = _outcomes_by_opponent(candidate, opponents)
    if any(
        not baseline_values[name] or not candidate_values[name] for name in opponents
    ):
        raise ValueError("each report needs outcomes for every opponent")
    rng = random.Random(20260724)
    deltas: list[float] = []
    for _ in range(iterations):
        baseline_sample: list[float] = []
        candidate_sample: list[float] = []
        for opponent in opponents:
            baseline_group = baseline_values[opponent]
            candidate_group = candidate_values[opponent]
            baseline_sample.extend(
                rng.choice(baseline_group) for _ in range(len(baseline_group))
            )
            candidate_sample.extend(
                rng.choice(candidate_group) for _ in range(len(candidate_group))
            )
        deltas.append(
            sum(candidate_sample) / len(candidate_sample)
            - sum(baseline_sample) / len(baseline_sample)
        )
    deltas.sort()
    low = deltas[int(iterations * 0.025)]
    high = deltas[min(iterations - 1, int(iterations * 0.975))]
    better = sum(delta > 0 for delta in deltas)
    tied = sum(delta == 0 for delta in deltas)
    return {
        "delta_ci95": [round(low, 6), round(high, 6)],
        "probability_candidate_better": round(
            (better + 0.5 * tied) / iterations,
            6,
        ),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": 20260724,
    }


def _outcomes_by_opponent(
    report: dict[str, Any],
    opponents: list[str],
) -> dict[str, list[float]]:
    values = {opponent: [] for opponent in opponents}
    scores = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    for run in report["runs"]:
        opponent = run["opponent"]
        if opponent in values:
            values[opponent].extend(scores[outcome] for outcome in run["outcomes"])
    return values


def _plan_differences(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    differences: dict[str, dict[str, Any]] = {}
    for key in (
        "battles_per_opponent",
        "repeats",
        "prompt_format",
        "policy_mode",
        "run_timeout_seconds",
    ):
        left = baseline["plan"].get(key)
        right = candidate["plan"].get(key)
        if left != right:
            differences[key] = {"baseline": left, "candidate": right}
    for key in ("model_id", "git_commit", "git_dirty"):
        left = baseline["provenance"].get(key)
        right = candidate["provenance"].get(key)
        if left != right:
            differences[key] = {"baseline": left, "candidate": right}
    return differences


def _read_complete_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"evaluation report must be an object: {path}")
    if value.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported evaluation report schema: {path}")
    if value.get("status") != "complete":
        raise ValueError(f"evaluation report is not complete: {path}")
    if value.get("quality", {}).get("status") != "valid":
        raise ValueError(f"evaluation report quality is not valid: {path}")
    value["_source_path"] = str(path)
    return value


def _write_evaluation_report(output_dir: Path, report: dict[str, Any]) -> None:
    _write_json(output_dir / "report.json", report)
    _write_text(output_dir / "report.md", render_evaluation_markdown(report))


def render_evaluation_markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        f"# Evaluation: {report['name']}",
        "",
        f"- Status: `{report['status']}`",
        f"- Model: `{report['provenance']['model_id']}`",
        f"- Policy: `{report['plan'].get('policy_mode', 'direct')}`",
        f"- Git commit: `{report['provenance']['git_commit']}`",
        f"- Battles: {overall['battles']}",
        (
            "- Progress: "
            f"{report.get('progress', {}).get('accepted_battles', overall['battles'])}"
            "/"
            f"{report.get('progress', {}).get('target_battles', overall['battles'])}"
        ),
        (
            "- Excluded attempts: "
            f"{report.get('progress', {}).get('excluded_attempts', 0)}"
        ),
        f"- Research quality: `{report['quality']['status']}`",
        "",
        "## Outcomes",
        "",
        "| Opponent | Battles | W | L | D | Score rate | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for opponent, metrics in report["by_opponent"].items():
        interval = metrics["score_rate_ci95"]
        interval_text = f"{interval[0]:.1%}–{interval[1]:.1%}" if interval else "—"
        lines.append(
            f"| {opponent} | {metrics['battles']} | {metrics['wins']} | "
            f"{metrics['losses']} | {metrics['draws']} | "
            f"{metrics['score_rate']:.1%} | {interval_text} |"
        )
    lines.extend(
        [
            "",
            "## Reliability and efficiency",
            "",
            f"- Fallback rate: {overall['fallback_rate']:.2%}",
            f"- Retry rate: {overall['retry_rate']:.2%}",
            f"- Decision error rate: {overall['decision_error_rate']:.2%}",
            f"- Tool-call coverage: {overall['tool_call_coverage']:.2%}",
            (
                "- Tactical-tool coverage: "
                f"{overall.get('tactical_tool_coverage', 0.0):.2%}"
            ),
            (
                "- Battle-plan coverage: "
                f"{overall.get('battle_plan_coverage', 0.0):.2%}"
            ),
            (
                "- Replan rate: "
                f"{overall.get('replan_rate', 0.0):.2%}"
            ),
            (
                "- Enrichment error rate: "
                f"{overall.get('enrichment_error_rate', 0.0):.2%}"
            ),
            (
                "- Planner recovered-error rate: "
                f"{overall.get('planner_recovered_error_rate', 0.0):.2%}"
            ),
            (
                "- Planner final-failure rate: "
                f"{overall.get('planner_failure_rate', 0.0):.2%}"
            ),
            (
                "- Protocol normalization rate: "
                f"{overall.get('normalization_rate', 0.0):.2%}"
            ),
            (
                "- Opponent-prediction accuracy: "
                f"{overall.get('prediction_accuracy', 0.0):.2%} "
                f"({overall.get('prediction_resolutions', 0)} resolved)"
            ),
            (
                "- Average model calls / decision: "
                f"{overall.get('average_model_calls_per_decision', 0.0):.2f}"
            ),
            f"- Rationale coverage: {overall['rationale_coverage']:.2%}",
            f"- Average tokens / battle: {overall['average_tokens_per_battle']:.1f}",
            (
                "- Average action tokens / decision: "
                f"{overall.get('average_action_tokens_per_decision', 0.0):.1f}"
            ),
            (
                "- Average action-context characters: "
                f"{overall.get('average_policy_input_characters', 0.0):.1f}"
            ),
            (
                "- Average Planner tokens / replan: "
                f"{overall.get('average_planner_tokens_per_replan', 0.0):.1f}"
            ),
            (
                "- Total tokens including preflight: "
                f"{report['cost_accounting']['evaluation_total_tokens']}"
            ),
            (
                "- Average Agent latency / decision: "
                f"{overall.get('average_agent_latency_seconds', overall['average_decision_latency_seconds']):.2f}s"
            ),
        ]
    )
    if "failure" in report:
        lines.extend(
            [
                "",
                "## Failure",
                "",
                f"`{report['failure']['error_type']}`: {report['failure']['message']}",
            ]
        )
    if report["quality"]["violations"]:
        lines.extend(["", "## Quality violations", ""])
        lines.extend(f"- {violation}" for violation in report["quality"]["violations"])
    return "\n".join(lines) + "\n"


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    primary = comparison["primary_outcome"]
    deltas = comparison["metric_deltas"]
    lines = [
        (
            f"# Comparison: {comparison['candidate']['name']} vs. "
            f"{comparison['baseline']['name']}"
        ),
        "",
        f"- Conclusion: `{comparison['conclusion']}`",
        f"- Score-rate delta: {primary['delta']:+.2%}",
        (
            "- Bootstrap 95% interval: "
            f"{primary['delta_ci95'][0]:+.2%} to {primary['delta_ci95'][1]:+.2%}"
        ),
        (
            "- Probability candidate is better: "
            f"{primary['probability_candidate_better']:.2%}"
        ),
        "",
        "## Trade-offs",
        "",
        f"- Fallback-rate delta: {deltas['fallback_rate']:+.2%}",
        f"- Retry-rate delta: {deltas['retry_rate']:+.2%}",
        (f"- Tokens / battle delta: {deltas['average_tokens_per_battle']:+.1f}"),
        (
            "- Model latency / decision delta: "
            f"{deltas['average_decision_latency_seconds']:+.2f}s"
        ),
    ]
    differences = comparison["comparability"]["declared_differences"]
    if differences:
        lines.extend(["", "## Declared configuration differences", ""])
        for key, values in differences.items():
            lines.append(f"- `{key}`: `{values['baseline']}` → `{values['candidate']}`")
    return "\n".join(lines) + "\n"


def _rate(numerator: float, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _failure_detail(error: Exception) -> str:
    recorded_errors = getattr(error, "errors", ())
    if recorded_errors:
        detail = str(recorded_errors[-1])
    else:
        detail = str(error)
    return redact_secrets(detail)[:1000]


def _difference(candidate: Any, baseline: Any) -> float:
    return round(float(candidate or 0.0) - float(baseline or 0.0), 6)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)
