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
async def test_evaluation_finishes_and_writes_aggregate_report(tmp_path) -> None:
    output = tmp_path / "evaluation"
    plan = EvaluationPlan(
        name="deterministic-integration",
        output_dir=output,
        opponents=("random",),
        battles_per_opponent=1,
        run_timeout_seconds=60,
    )
    with managed_showdown_server():
        report = await run_evaluation(DeterministicModelClient(), plan)

    assert report["status"] == "complete"
    assert report["overall"]["battles"] == 1
    assert report["overall"]["fallbacks"] == 0
    assert (output / "report.json").is_file()
    assert (output / "report.md").is_file()
