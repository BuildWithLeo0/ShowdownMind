import pytest

from showdown_mind.agent_runner import run_agent_battles
from showdown_mind.models import DeterministicModelClient


@pytest.mark.asyncio
async def test_agent_runner_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        await run_agent_battles(
            DeterministicModelClient(),
            timeout_seconds=0,
        )


@pytest.mark.asyncio
async def test_agent_runner_rejects_existing_decision_log(tmp_path) -> None:
    decision_log = tmp_path / "run.jsonl"
    decision_log.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        await run_agent_battles(
            DeterministicModelClient(),
            decision_log=decision_log,
        )


@pytest.mark.asyncio
async def test_agent_runner_writes_sanitized_failure_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    def fail_to_construct(*args, **kwargs):
        raise RuntimeError("provider failed with sk-abcdefghijk")

    monkeypatch.setattr(
        "showdown_mind.agent_runner.ResearchPlayer",
        fail_to_construct,
    )
    decision_log = tmp_path / "failed.jsonl"

    with pytest.raises(RuntimeError, match="provider failed"):
        await run_agent_battles(
            DeterministicModelClient(),
            decision_log=decision_log,
        )

    failure = decision_log.with_suffix(".failure.json")
    assert failure.is_file()
    assert "abcdefghijk" not in failure.read_text(encoding="utf-8")
