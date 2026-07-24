import os

import pytest

from showdown_mind.baselines import run_baseline_battles
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
