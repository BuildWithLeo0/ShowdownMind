import json
import os

import pytest

from showdown_mind.agent_runner import run_agent_battles
from showdown_mind.baselines import run_baseline_battles
from showdown_mind.evaluation import EvaluationPlan, run_evaluation
from showdown_mind.models import DeterministicModelClient
from showdown_mind.showdown import managed_showdown_server

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("SHOWDOWN_MIND_RUN_INTEGRATION") != "1",
        reason="set SHOWDOWN_MIND_RUN_INTEGRATION=1 to run local battles",
    ),
]


@pytest.mark.asyncio
async def test_random_players_finish_a_local_battle() -> None:
    with managed_showdown_server():
        result = await run_baseline_battles(battles=1)

    assert result.finished_battles == 1


@pytest.mark.asyncio
async def test_policy_agent_finishes_and_logs_a_local_battle(tmp_path) -> None:
    decision_log = tmp_path / "agent.jsonl"
    with managed_showdown_server():
        result = await run_agent_battles(
            DeterministicModelClient(),
            battles=1,
            decision_log=decision_log,
            timeout_seconds=60,
        )

    assert result.finished_battles == 1
    assert result.decisions > 0
    assert result.fallbacks == 0
    assert decision_log.read_text(encoding="utf-8").strip()
    assert result.manifest_path.endswith(".manifest.json")
    assert result.summary_path.endswith(".summary.json")
    assert result.failure_path.endswith(".failure.json")
    assert (tmp_path / "agent.manifest.json").is_file()
    assert (tmp_path / "agent.summary.json").is_file()
    assert not (tmp_path / "agent.failure.json").exists()


@pytest.mark.asyncio
async def test_controlled_agent_finishes_with_memory_plan_and_tactics(tmp_path) -> None:
    decision_log = tmp_path / "controlled.jsonl"
    with managed_showdown_server():
        result = await run_agent_battles(
            DeterministicModelClient(),
            opponent_name="max-base-power",
            battles=1,
            decision_log=decision_log,
            timeout_seconds=90,
            policy_mode="controlled-agent",
        )

    records = [
        json.loads(line)
        for line in decision_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(
        (tmp_path / "controlled.manifest.json").read_text(encoding="utf-8")
    )

    assert result.finished_battles == 1
    assert result.fallbacks == 0
    assert records
    assert records[0]["plan_trigger"] == "initial"
    assert all(record["battle_plan"] for record in records)
    assert all(record["opponent_prediction"] for record in records)
    assert all(
        any(
            execution["tool_name"] == "analyze_battle_options"
            and execution["execution_kind"] == "internal"
            for execution in record["tool_executions"]
        )
        for record in records
    )
    assert manifest["policy"]["architecture"]["kind"] == (
        "hierarchical-controlled-agent-v2"
    )


@pytest.mark.asyncio
async def test_evaluation_finishes_and_writes_aggregate_report(tmp_path) -> None:
    output = tmp_path / "evaluation"
    plan = EvaluationPlan(
        name="deterministic-integration",
        output_dir=output,
        opponents=("max-base-power",),
        battles_per_opponent=1,
        run_timeout_seconds=60,
        policy_mode="controlled-agent",
    )
    with managed_showdown_server():
        report = await run_evaluation(DeterministicModelClient(), plan)

    assert report["status"] == "complete"
    assert report["overall"]["battles"] == 1
    assert report["overall"]["fallbacks"] == 0
    assert report["overall"]["battle_plan_coverage"] == 1.0
    assert report["overall"]["prediction_coverage"] == 1.0
    assert report["overall"]["tactical_tool_coverage"] == 1.0
    assert (output / "report.json").is_file()
    assert (output / "report.md").is_file()
