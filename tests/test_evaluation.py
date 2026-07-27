import json
from pathlib import Path

import pytest

from showdown_mind.agent_runner import AgentSmokeResult
from showdown_mind.evaluation import (
    EvaluationError,
    EvaluationPlan,
    aggregate_runs,
    assess_quality,
    build_comparison,
    compare_evaluation_reports,
    run_evaluation,
    summarize_decision_log,
    wilson_interval,
)
from showdown_mind.model_runner import ModelCheckResult


def decision_record(
    *,
    attempts: int = 1,
    fallback: bool = False,
    errors: list[str] | None = None,
) -> dict:
    return {
        "attempts": attempts,
        "fallback_used": fallback,
        "errors": errors or [],
        "tool_call_ids": ["call-test"],
        "short_rationale": "Use reliable damage.",
        "confidence": 0.8,
        "reason_codes": ["DAMAGE"],
        "elapsed_seconds": 2.0,
        "policy_input_characters": 1200,
        "usages": [
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
        ],
    }


def write_decisions(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _one_battle_result(path: Path, opponent: str) -> AgentSmokeResult:
    return AgentSmokeResult(
        battle_format="gen9randombattle",
        prompt_format="pruned-v1",
        opponent=opponent,
        requested_battles=1,
        finished_battles=1,
        agent_wins=1,
        opponent_wins=0,
        draws=0,
        decisions=1,
        model_calls=1,
        fallbacks=0,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        model_input_characters=1200,
        elapsed_seconds=2.0,
        decision_log=str(path),
        manifest_path=str(path.with_suffix(".manifest.json")),
        summary_path=str(path.with_suffix(".summary.json")),
        failure_path=str(path.with_suffix(".failure.json")),
    )


def run_entry(opponent: str, outcomes: list[str]) -> dict:
    decisions = len(outcomes)
    metrics = {
        "decisions": decisions,
        "retry_calls": 0,
        "decisions_with_retry": 0,
        "fallbacks": 0,
        "decisions_with_errors": 0,
        "transport_errors": 0,
        "validation_errors": 0,
        "tool_call_decisions": decisions,
        "rationale_decisions": decisions,
        "confidence_sum": 0.8 * decisions,
        "confidence_count": decisions,
        "input_tokens": 10 * decisions,
        "output_tokens": 5 * decisions,
        "total_tokens": 15 * decisions,
        "decision_latency_seconds": 2.0 * decisions,
        "reason_code_counts": {"DAMAGE": decisions},
    }
    return {
        "status": "complete",
        "opponent": opponent,
        "repeat": 1,
        "outcomes": outcomes,
        "result": {"elapsed_seconds": 3.0 * len(outcomes)},
        "decision_metrics": metrics,
    }


def report(name: str, outcomes: list[str], opponent: str = "random") -> dict:
    run = run_entry(opponent, outcomes)
    return {
        "schema_version": "1.0",
        "status": "complete",
        "name": name,
        "plan": {
            "battle_format": "gen9randombattle",
            "opponents": [opponent],
            "battles_per_opponent": len(outcomes),
            "repeats": 1,
            "prompt_format": "pruned-v1",
            "run_timeout_seconds": 300,
        },
        "provenance": {
            "model_id": "test-model",
            "git_commit": "abc",
            "git_dirty": False,
        },
        "quality": {"status": "valid", "violations": []},
        "runs": [run],
        "overall": aggregate_runs([run]),
        "by_opponent": {opponent: aggregate_runs([run])},
    }


def test_plan_preview_is_read_only_and_counts_matrix(tmp_path) -> None:
    output = tmp_path / "evaluation"
    plan = EvaluationPlan(
        name="v0",
        output_dir=output,
        opponents=("random", "max-base-power"),
        battles_per_opponent=3,
        repeats=2,
    )

    preview = plan.preview()

    assert preview["will_call_model"] is False
    assert preview["plan"]["total_runs"] == 12
    assert preview["plan"]["total_battles"] == 12
    assert len(preview["matrix"]) == 12
    assert preview["matrix"][0]["checkpoint_id"] == "random-r01-b001"
    assert not output.exists()


def test_plan_rejects_unknown_or_duplicate_opponents(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        EvaluationPlan(
            name="bad",
            output_dir=tmp_path / "bad",
            opponents=("missing",),
        )
    with pytest.raises(ValueError, match="unique"):
        EvaluationPlan(
            name="bad",
            output_dir=tmp_path / "bad",
            opponents=("random", "random"),
        )


def test_summarizes_reliability_cost_and_coverage(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    write_decisions(
        path,
        [
            decision_record(),
            decision_record(
                attempts=2,
                fallback=True,
                errors=["ModelCallError: connection"],
            ),
        ],
    )

    metrics = summarize_decision_log(path)

    assert metrics["decisions"] == 2
    assert metrics["retry_calls"] == 1
    assert metrics["fallbacks"] == 1
    assert metrics["transport_errors"] == 1
    assert metrics["total_tokens"] == 30
    assert metrics["tool_call_decisions"] == 2
    assert metrics["rationale_decisions"] == 2


def test_summarizes_controlled_agent_planning_and_predictions(tmp_path) -> None:
    path = tmp_path / "controlled.jsonl"
    first = {
        **decision_record(),
        "battle_plan": {"schema": "battle-plan-v1"},
        "plan_trigger": "initial",
        "planner_model_calls": 1,
        "planner_errors": [],
        "planner_elapsed_seconds": 0.4,
        "planner_usages": [
            {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}
        ],
        "opponent_prediction": {
            "kind": "attack",
            "detail": "waterfall",
            "confidence": 0.7,
        },
        "memory": {},
        "tool_executions": [
            {"tool_name": "analyze_battle_options", "execution_kind": "internal"}
        ],
    }
    second = {
        **first,
        "plan_trigger": "",
        "planner_model_calls": 0,
        "planner_elapsed_seconds": 0,
        "planner_usages": [],
        "memory": {
            "previous_prediction_resolution": {
                "predicted": {"decision_turn": 1},
                "actual_kind": "attack",
                "observed_turn": 1,
                "matched": True,
                "evidence_event_id": "event-2",
            }
        },
    }
    write_decisions(path, [first, second])

    metrics = summarize_decision_log(path)

    assert metrics["battle_plan_decisions"] == 2
    assert metrics["replan_decisions"] == 1
    assert metrics["planner_model_calls"] == 1
    assert metrics["planner_total_tokens"] == 12
    assert metrics["planner_failure_decisions"] == 0
    assert metrics["prediction_resolutions"] == 1
    assert metrics["prediction_matches"] == 1
    assert metrics["tactical_tool_decisions"] == 2
    assert metrics["policy_input_characters"] == 2400


def test_aggregate_runs_includes_outcomes_and_rates() -> None:
    metrics = aggregate_runs([run_entry("random", ["win", "loss", "draw", "win"])])

    assert metrics["battles"] == 4
    assert metrics["wins"] == 2
    assert metrics["score_rate"] == 0.625
    assert metrics["tool_call_coverage"] == 1.0
    assert metrics["average_tokens_per_battle"] == 15.0
    assert wilson_interval(2.5, 4) is not None


def test_summarizes_normalization_and_separates_action_cost(tmp_path) -> None:
    path = tmp_path / "normalized.jsonl"
    record = {
        **decision_record(),
        "decision_normalizations": ["reason_codes_truncated:4->3"],
        "plan_trigger": "initial",
        "planner_usages": [
            {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}
        ],
    }
    write_decisions(path, [record])
    run = run_entry("max-base-power", ["win"])
    run["decision_metrics"] = summarize_decision_log(path)

    metrics = aggregate_runs([run])

    assert metrics["normalization_rate"] == 1.0
    assert metrics["average_policy_input_characters"] == 1200.0
    assert metrics["planner_total_tokens"] == 6
    assert metrics["action_total_tokens"] == 9
    assert metrics["average_action_tokens_per_decision"] == 9.0


@pytest.mark.asyncio
async def test_runs_matrix_and_writes_reports(tmp_path) -> None:
    class FakeClient:
        model_id = "fake-model"

    async def fake_check(*args, **kwargs):
        return ModelCheckResult(
            model_id="fake-model",
            prompt_format="pruned-v1",
            model_input_characters=10,
            action_id="move:tackle",
            confidence=0.8,
            reason_codes=("DAMAGE",),
            short_rationale="Use reliable damage.",
            tool_call_id="call-check",
            attempts=1,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
        )

    async def fake_runner(*args, **kwargs):
        path = kwargs["decision_log"]
        write_decisions(path, [decision_record(), decision_record()])
        opponent = kwargs["opponent_name"]
        return AgentSmokeResult(
            battle_format="gen9randombattle",
            prompt_format="pruned-v1",
            opponent=opponent,
            requested_battles=1,
            finished_battles=1,
            agent_wins=1,
            opponent_wins=0,
            draws=0,
            decisions=2,
            model_calls=2,
            fallbacks=0,
            input_tokens=20,
            output_tokens=10,
            total_tokens=30,
            model_input_characters=20,
            elapsed_seconds=4.0,
            decision_log=str(path),
            manifest_path=str(path.with_suffix(".manifest.json")),
            summary_path=str(path.with_suffix(".summary.json")),
            failure_path=str(path.with_suffix(".failure.json")),
        )

    output = tmp_path / "evaluation"
    plan = EvaluationPlan(
        name="v0",
        output_dir=output,
        opponents=("random", "max-base-power"),
        battles_per_opponent=2,
    )

    result = await run_evaluation(
        FakeClient(),
        plan,
        battle_runner=fake_runner,
        connectivity_checker=fake_check,
    )

    assert result["status"] == "complete"
    assert result["quality"]["status"] == "valid"
    assert result["overall"]["battles"] == 4
    assert result["cost_accounting"]["evaluation_total_tokens"] == 122
    assert len(result["runs"]) == 4
    assert result["progress"]["accepted_battles"] == 4
    assert (output / "plan.json").is_file()
    assert (output / "report.json").is_file()
    assert "Evaluation: v0" in (output / "report.md").read_text()


@pytest.mark.asyncio
async def test_resume_skips_accepted_battles_and_retries_interrupted_cell(
    tmp_path,
) -> None:
    class FakeClient:
        model_id = "fake-model"

    check_calls = 0

    async def fake_check(*args, **kwargs):
        nonlocal check_calls
        check_calls += 1
        return ModelCheckResult(
            model_id="fake-model",
            prompt_format="pruned-v1",
            model_input_characters=10,
            action_id="move:tackle",
            confidence=0.8,
            reason_codes=("DAMAGE",),
            short_rationale="Use reliable damage.",
            tool_call_id="call-check",
            attempts=1,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
        )

    first_calls = 0

    async def interrupted_runner(*args, **kwargs):
        nonlocal first_calls
        first_calls += 1
        path = kwargs["decision_log"]
        write_decisions(path, [decision_record()])
        if first_calls == 2:
            raise RuntimeError("insufficient balance")
        return _one_battle_result(path, kwargs["opponent_name"])

    output = tmp_path / "resumable"
    plan = EvaluationPlan(
        name="resumable",
        output_dir=output,
        opponents=("max-base-power",),
        battles_per_opponent=3,
    )

    with pytest.raises(EvaluationError, match="resume with --resume"):
        await run_evaluation(
            FakeClient(),
            plan,
            battle_runner=interrupted_runner,
            connectivity_checker=fake_check,
        )

    incomplete = json.loads((output / "report.json").read_text())
    assert incomplete["progress"]["accepted_battles"] == 1
    assert incomplete["progress"]["remaining_battles"] == 2
    assert incomplete["attempts"][0]["attempt_id"].endswith("b001-a01")
    assert incomplete["attempts"][1]["status"] == "interrupted"

    resumed_calls: list[str] = []

    async def resumed_runner(*args, **kwargs):
        path = kwargs["decision_log"]
        resumed_calls.append(path.stem)
        write_decisions(path, [decision_record()])
        return _one_battle_result(path, kwargs["opponent_name"])

    async def unexpected_check(*args, **kwargs):
        raise AssertionError("saved successful preflight must be reused")

    complete = await run_evaluation(
        FakeClient(),
        plan,
        battle_runner=resumed_runner,
        connectivity_checker=unexpected_check,
        resume=True,
    )

    assert complete["status"] == "complete"
    assert complete["progress"]["accepted_battles"] == 3
    assert complete["progress"]["excluded_attempts"] == 1
    assert complete["cost_accounting"]["excluded_attempt_total_tokens"] == 15
    assert resumed_calls == [
        "max-base-power-r01-b002-a02",
        "max-base-power-r01-b003-a01",
    ]
    assert check_calls == 1

    async def unexpected_runner(*args, **kwargs):
        raise AssertionError("complete checkpoints must not rerun")

    repeated = await run_evaluation(
        FakeClient(),
        plan,
        battle_runner=unexpected_runner,
        connectivity_checker=unexpected_check,
        resume=True,
    )
    assert repeated["status"] == "complete"


@pytest.mark.asyncio
async def test_resume_recovers_summary_written_before_attempt_finalization(
    tmp_path,
) -> None:
    class FakeClient:
        model_id = "fake-model"

    async def fake_check(*args, **kwargs):
        return ModelCheckResult(
            model_id="fake-model",
            prompt_format="pruned-v1",
            model_input_characters=10,
            action_id="move:tackle",
            confidence=0.8,
            reason_codes=("DAMAGE",),
            short_rationale="Use reliable damage.",
            tool_call_id="call-check",
            attempts=1,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
        )

    async def fake_runner(*args, **kwargs):
        path = kwargs["decision_log"]
        write_decisions(path, [decision_record()])
        result = _one_battle_result(path, kwargs["opponent_name"])
        path.with_suffix(".summary.json").write_text(
            json.dumps(result.to_dict()),
            encoding="utf-8",
        )
        return result

    output = tmp_path / "crash-recovery"
    plan = EvaluationPlan(
        name="crash-recovery",
        output_dir=output,
        opponents=("random",),
        battles_per_opponent=1,
    )
    await run_evaluation(
        FakeClient(),
        plan,
        battle_runner=fake_runner,
        connectivity_checker=fake_check,
    )

    attempt_path = next((output / "runs").glob("*.attempt.json"))
    attempt = json.loads(attempt_path.read_text())
    attempt["status"] = "running"
    attempt.pop("run")
    attempt_path.write_text(json.dumps(attempt), encoding="utf-8")

    async def unexpected_runner(*args, **kwargs):
        raise AssertionError("completed summary must be recovered")

    report = await run_evaluation(
        FakeClient(),
        plan,
        battle_runner=unexpected_runner,
        resume=True,
    )

    assert report["status"] == "complete"
    assert report["attempts"][0]["recovered_after_process_exit"] is True


@pytest.mark.asyncio
async def test_resume_rejects_changed_plan_or_model(tmp_path) -> None:
    class FakeClient:
        model_id = "fake-model"

    async def failed_check(*args, **kwargs):
        raise RuntimeError("offline")

    output = tmp_path / "locked"
    plan = EvaluationPlan(
        name="locked",
        output_dir=output,
        opponents=("random",),
        battles_per_opponent=1,
    )
    with pytest.raises(EvaluationError):
        await run_evaluation(
            FakeClient(),
            plan,
            connectivity_checker=failed_check,
        )

    changed_plan = EvaluationPlan(
        name="locked",
        output_dir=output,
        opponents=("random",),
        battles_per_opponent=2,
    )
    with pytest.raises(ValueError, match="plan does not exactly match"):
        await run_evaluation(
            FakeClient(),
            changed_plan,
            resume=True,
        )

    class OtherClient:
        model_id = "other-model"

    with pytest.raises(ValueError, match="provenance"):
        await run_evaluation(
            OtherClient(),
            plan,
            resume=True,
        )


@pytest.mark.asyncio
async def test_failed_preflight_writes_incomplete_report(tmp_path) -> None:
    class FakeClient:
        model_id = "fake-model"

    class DetailedProviderError(RuntimeError):
        errors = ("ModelCallError: unavailable with sk-abcdefghijk",)

    async def failed_check(*args, **kwargs):
        raise DetailedProviderError("generic policy failure")

    output = tmp_path / "failed"
    plan = EvaluationPlan(
        name="failed",
        output_dir=output,
        opponents=("random",),
        battles_per_opponent=1,
    )

    with pytest.raises(EvaluationError, match="evaluation stopped") as error:
        await run_evaluation(
            FakeClient(),
            plan,
            connectivity_checker=failed_check,
        )

    assert "abcdefghijk" not in str(error.value)
    saved = json.loads((output / "report.json").read_text())
    assert saved["status"] == "incomplete"
    assert saved["overall"]["battles"] == 0
    assert saved["failure"]["error_type"] == "DetailedProviderError"
    assert "ModelCallError: unavailable" in saved["failure"]["message"]
    assert "abcdefghijk" not in json.dumps(saved)


@pytest.mark.asyncio
async def test_hard_quality_gate_stops_before_next_opponent(tmp_path) -> None:
    class FakeClient:
        model_id = "fake-model"

    async def fake_check(*args, **kwargs):
        return ModelCheckResult(
            model_id="fake-model",
            prompt_format="pruned-v1",
            model_input_characters=10,
            action_id="move:tackle",
            confidence=0.8,
            reason_codes=("DAMAGE",),
            short_rationale="Use reliable damage.",
            tool_call_id="call-check",
            attempts=1,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
        )

    opponents_run = []

    async def fallback_heavy_runner(*args, **kwargs):
        opponents_run.append(kwargs["opponent_name"])
        path = kwargs["decision_log"]
        records = [
            {
                **decision_record(
                    attempts=2,
                    fallback=True,
                    errors=["ModelCallError: provider unavailable"],
                ),
                "tool_call_ids": [],
            }
            for _ in range(5)
        ]
        write_decisions(path, records)
        return AgentSmokeResult(
            battle_format="gen9randombattle",
            prompt_format="pruned-v1",
            opponent=kwargs["opponent_name"],
            requested_battles=1,
            finished_battles=1,
            agent_wins=1,
            opponent_wins=0,
            draws=0,
            decisions=5,
            model_calls=10,
            fallbacks=5,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model_input_characters=10,
            elapsed_seconds=1.0,
            decision_log=str(path),
            manifest_path=str(path.with_suffix(".manifest.json")),
            summary_path=str(path.with_suffix(".summary.json")),
            failure_path=str(path.with_suffix(".failure.json")),
        )

    output = tmp_path / "quality-stop"
    plan = EvaluationPlan(
        name="quality-stop",
        output_dir=output,
        opponents=("random", "max-base-power"),
        battles_per_opponent=1,
    )

    with pytest.raises(EvaluationError, match="quality stop gate"):
        await run_evaluation(
            FakeClient(),
            plan,
            battle_runner=fallback_heavy_runner,
            connectivity_checker=fake_check,
        )

    assert opponents_run == ["random"]
    saved = json.loads((output / "report.json").read_text())
    assert saved["status"] == "incomplete"
    assert saved["quality"]["status"] == "pending"
    assert len(saved["runs"]) == 0
    assert saved["progress"]["excluded_attempts"] == 1
    assert saved["attempts"][0]["status"] == "rejected"


def test_quality_gate_rejects_fallback_heavy_results() -> None:
    metrics = aggregate_runs([run_entry("random", ["win"] * 3)])
    metrics["fallback_rate"] = 0.25
    metrics["decision_error_rate"] = 0.4
    metrics["tool_call_coverage"] = 0.6

    quality = assess_quality(metrics)

    assert quality["status"] == "invalid"
    assert len(quality["violations"]) == 3


def test_tactical_quality_requires_executed_calculator_tool() -> None:
    metrics = aggregate_runs([run_entry("random", ["win"] * 3)])
    metrics["tactical_tool_coverage"] = 0.5

    quality = assess_quality(metrics, policy_mode="tactical-tool")

    assert quality["status"] == "invalid"
    assert any(
        "tactical_tool_coverage" in violation
        for violation in quality["violations"]
    )


def test_controlled_agent_quality_requires_automatic_calculator() -> None:
    metrics = aggregate_runs([run_entry("random", ["win"] * 3)])
    metrics["tactical_tool_coverage"] = 0.5

    quality = assess_quality(metrics, policy_mode="controlled-agent")

    assert quality["status"] == "invalid"


def test_controlled_agent_quality_requires_plan_and_prediction_coverage() -> None:
    metrics = aggregate_runs([run_entry("random", ["win"] * 3)])
    metrics["tactical_tool_coverage"] = 1.0

    quality = assess_quality(metrics, policy_mode="controlled-agent")

    assert quality["status"] == "invalid"
    assert any(
        "battle_plan_coverage" in violation
        for violation in quality["violations"]
    )
    assert any(
        "prediction_coverage" in violation
        for violation in quality["violations"]
    )


def test_controlled_quality_distinguishes_recovered_planner_error() -> None:
    metrics = aggregate_runs([run_entry("max-base-power", ["win"])])
    metrics.update(
        {
            "tactical_tool_coverage": 1.0,
            "battle_plan_coverage": 1.0,
            "prediction_coverage": 1.0,
            "planner_error_rate": 1.0,
            "planner_recovered_error_rate": 1.0,
            "planner_failure_rate": 0.0,
            "enrichment_error_rate": 0.0,
        }
    )

    quality = assess_quality(metrics, policy_mode="controlled-agent")

    assert quality["status"] == "valid"

    metrics["planner_failure_rate"] = 1.0
    quality = assess_quality(metrics, policy_mode="controlled-agent")
    assert quality["status"] == "invalid"
    assert any(
        "planner_failure_rate" in violation
        for violation in quality["violations"]
    )


def test_comparison_detects_clear_improvement() -> None:
    baseline = report("v0", ["loss"] * 30)
    candidate = report("v1", ["win"] * 30)

    comparison = build_comparison(
        baseline,
        candidate,
        bootstrap_iterations=200,
    )

    assert comparison["conclusion"] == "improved"
    assert comparison["primary_outcome"]["delta"] == 1.0
    assert comparison["primary_outcome"]["delta_ci95"] == [1.0, 1.0]


def test_comparison_marks_small_samples_insufficient() -> None:
    comparison = build_comparison(
        report("v0", ["loss"] * 3),
        report("v1", ["win"] * 3),
        bootstrap_iterations=200,
    )

    assert comparison["conclusion"] == "insufficient_data"


def test_comparison_requires_matching_opponent_sets() -> None:
    with pytest.raises(ValueError, match="opponent sets"):
        build_comparison(
            report("v0", ["loss"] * 20, opponent="random"),
            report("v1", ["win"] * 20, opponent="max-base-power"),
            bootstrap_iterations=200,
        )


def test_comparison_rejects_invalid_quality_report() -> None:
    baseline = report("v0", ["loss"] * 20)
    candidate = report("v1", ["win"] * 20)
    candidate["quality"] = {"status": "invalid", "violations": ["fallbacks"]}

    with pytest.raises(ValueError, match="quality is not valid"):
        build_comparison(
            baseline,
            candidate,
            bootstrap_iterations=200,
        )


def test_comparison_writes_json_and_markdown(tmp_path) -> None:
    baseline_path = tmp_path / "v0.json"
    candidate_path = tmp_path / "v1.json"
    output = tmp_path / "comparison.json"
    baseline_path.write_text(json.dumps(report("v0", ["loss"] * 20)))
    candidate_path.write_text(json.dumps(report("v1", ["win"] * 20)))

    result = compare_evaluation_reports(
        baseline_path,
        candidate_path,
        output_path=output,
        bootstrap_iterations=200,
    )

    assert result["conclusion"] == "improved"
    assert output.is_file()
    assert "Comparison: v1 vs. v0" in output.with_suffix(".md").read_text()
