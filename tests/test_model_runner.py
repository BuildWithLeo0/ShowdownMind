import pytest

from showdown_mind.model_runner import run_model_check
from showdown_mind.models import ScriptedModelClient


@pytest.mark.asyncio
async def test_model_check_uses_the_policy_contract() -> None:
    result = await run_model_check(
        ScriptedModelClient(
            ['{"action_id":"move:thunderbolt","confidence":0.9}'],
            model_id="check-model",
        )
    )

    assert result.model_id == "check-model"
    assert result.action_id == "move:thunderbolt"
    assert result.attempts == 1
    assert result.total_tokens == 0
