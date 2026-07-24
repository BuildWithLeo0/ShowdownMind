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
