from argparse import Namespace
import json

from showdown_mind import cli
from showdown_mind.agent_runner import AgentSmokeResult
from showdown_mind.viewer import ViewerBuildResult


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


def test_visualize_opens_generated_viewer(monkeypatch, tmp_path) -> None:
    output = tmp_path / "viewer.html"
    opened = []

    def fake_build(*args, **kwargs):
        assert kwargs["battle_id"] == "battle-1"
        return ViewerBuildResult(
            battle_id="battle-1",
            decisions=2,
            replay_path="replay.html",
            output_path=str(output),
        )

    monkeypatch.setattr(cli, "build_replay_viewer", fake_build)
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    args = Namespace(
        decision_log=tmp_path / "decisions.jsonl",
        replay=None,
        battle_id="battle-1",
        output=output,
        no_open=False,
        force=False,
    )

    assert cli._run_visualize(args) == 0
    assert opened == [output.resolve().as_uri()]


def test_evaluate_is_dry_run_by_default(monkeypatch, tmp_path, capsys) -> None:
    def fail_if_live_client_is_created():
        raise AssertionError("dry-run must not load credentials")

    monkeypatch.setattr(
        cli,
        "live_model_client_from_env",
        fail_if_live_client_is_created,
    )
    output = tmp_path / "evaluation"
    args = Namespace(
        name="v0",
        output_dir=output,
        opponents=["random", "max-base-power"],
        battles_per_opponent=2,
        repeats=1,
        prompt_format="pruned",
        run_timeout=None,
        run=False,
        no_manage_server=False,
    )

    assert cli._run_evaluate(args) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "dry-run"
    assert printed["plan"]["total_battles"] == 4
    assert not output.exists()
