from argparse import Namespace

from showdown_mind import cli
from showdown_mind.agent_runner import AgentSmokeResult


def smoke_result() -> AgentSmokeResult:
    return AgentSmokeResult(
        battle_format="gen9randombattle",
        opponent="random",
        requested_battles=1,
        finished_battles=1,
        agent_wins=1,
        opponent_wins=0,
        draws=0,
        decisions=1,
        model_calls=1,
        fallbacks=0,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        model_input_characters=100,
        prompt_format="compact",
        elapsed_seconds=0.1,
        decision_log="test.jsonl",
        manifest_path="test.manifest.json",
        summary_path="test.summary.json",
        failure_path="test.failure.json",
    )


def test_agent_smoke_does_not_require_live_model_timeout(monkeypatch) -> None:
    async def fake_run(*args, **kwargs):
        assert "timeout_seconds" not in kwargs
        return smoke_result()

    monkeypatch.setattr(cli, "run_agent_battles", fake_run)
    args = Namespace(
        opponent="random",
        battles=1,
        decision_log=None,
        prompt_format="compact",
        no_manage_server=True,
    )

    assert cli._run_agent_smoke(args) == 0


def test_llm_smoke_passes_configured_timeout(monkeypatch) -> None:
    calls = {}

    class FakeClient:
        async def aclose(self) -> None:
            return None

    async def fake_model_check(client, **kwargs):
        calls["check_prompt_format"] = kwargs["prompt_format"]
        return Namespace(to_dict=dict)

    async def fake_run(*args, **kwargs):
        calls.update(kwargs)
        return smoke_result()

    monkeypatch.setattr(cli, "live_model_client_from_env", FakeClient)
    monkeypatch.setattr(cli, "run_model_check", fake_model_check)
    monkeypatch.setattr(cli, "run_agent_battles", fake_run)
    args = Namespace(
        opponent="random",
        battles=1,
        battle_timeout=321.0,
        decision_log=None,
        prompt_format="full",
        no_manage_server=True,
    )

    assert cli._run_llm_smoke(args) == 0
    assert calls["timeout_seconds"] == 321.0
    assert calls["prompt_format"] == "full"
    assert calls["check_prompt_format"] == "full"
