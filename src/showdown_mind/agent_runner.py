from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from showdown_mind.agent import ResearchPlayer
from showdown_mind.baselines import (
    BATTLE_FORMAT,
    LOCAL_SERVER_CONFIGURATION,
    BaselineError,
    ensure_localhost_bypasses_proxy,
    make_baseline,
)
from showdown_mind.experiment_artifacts import (
    ExperimentArtifactWriter,
    ExperimentSpec,
)
from showdown_mind.models import ModelClient
from showdown_mind.paths import REPLAY_DIR, RUNTIME_DIR
from showdown_mind.policy import SingleCallPolicy
from showdown_mind.storage import JsonlDecisionWriter


@dataclass(frozen=True)
class AgentSmokeResult:
    battle_format: str
    prompt_format: str
    opponent: str
    requested_battles: int
    finished_battles: int
    agent_wins: int
    opponent_wins: int
    draws: int
    decisions: int
    model_calls: int
    fallbacks: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_input_characters: int
    elapsed_seconds: float
    decision_log: str
    manifest_path: str
    summary_path: str
    failure_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_agent_battles(
    model_client: ModelClient,
    *,
    opponent_name: str = "max-base-power",
    battles: int = 1,
    decision_log: Path | None = None,
    timeout_seconds: float | None = None,
    prompt_format: str = "pruned",
) -> AgentSmokeResult:
    if battles < 1:
        raise ValueError("battles must be at least 1")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    ensure_localhost_bypasses_proxy()
    effective_timeout = timeout_seconds or max(60.0, battles * 30.0)
    log_path = decision_log or (
        RUNTIME_DIR / "decisions" / f"agent-smoke-{time.time_ns()}.jsonl"
    )
    artifacts = ExperimentArtifactWriter(log_path)
    artifacts.assert_new_run()
    artifacts.write_manifest(
        model_client,
        ExperimentSpec(
            battle_format=BATTLE_FORMAT,
            opponent=opponent_name,
            requested_battles=battles,
            prompt_format=f"{prompt_format}-v1",
            timeout_seconds=effective_timeout,
        ),
    )
    writer = JsonlDecisionWriter(log_path)
    agent = None
    opponent = None
    try:
        agent = ResearchPlayer(
            SingleCallPolicy(model_client, input_format=prompt_format),
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=1,
            save_replays=str(REPLAY_DIR),
            server_configuration=LOCAL_SERVER_CONFIGURATION,
            decision_sink=writer,
        )
        opponent = make_baseline(opponent_name)
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                agent.battle_against(opponent, n_battles=battles),
                timeout=effective_timeout,
            )
        except TimeoutError as exc:
            raise BaselineError(
                "Agent battle timed out after requesting "
                f"{battles} battles; increase --battle-timeout"
            ) from exc

        finished = agent.n_finished_battles
        if finished != battles:
            raise BaselineError(
                f"Requested {battles} battles but only {finished} finished"
            )

        agent_wins = agent.n_won_battles
        opponent_wins = opponent.n_won_battles
        result = AgentSmokeResult(
            battle_format=BATTLE_FORMAT,
            prompt_format=f"{prompt_format}-v1",
            opponent=opponent_name,
            requested_battles=battles,
            finished_battles=finished,
            agent_wins=agent_wins,
            opponent_wins=opponent_wins,
            draws=finished - agent_wins - opponent_wins,
            decisions=len(agent.decision_records),
            model_calls=sum(record.attempts for record in agent.decision_records),
            fallbacks=sum(record.fallback_used for record in agent.decision_records),
            input_tokens=sum(
                usage.input_tokens
                for record in agent.decision_records
                for usage in record.usages
            ),
            output_tokens=sum(
                usage.output_tokens
                for record in agent.decision_records
                for usage in record.usages
            ),
            total_tokens=sum(
                usage.total_tokens
                for record in agent.decision_records
                for usage in record.usages
            ),
            model_input_characters=sum(
                record.policy_input_characters for record in agent.decision_records
            ),
            elapsed_seconds=round(time.monotonic() - started, 3),
            decision_log=str(log_path),
            manifest_path=str(artifacts.paths.manifest),
            summary_path=str(artifacts.paths.summary),
            failure_path=str(artifacts.paths.failure),
        )
        artifacts.write_summary(result.to_dict())
        return result
    except Exception as exc:
        artifacts.write_failure(exc)
        raise
    finally:
        players = [player for player in (agent, opponent) if player is not None]
        await asyncio.gather(
            *(player.ps_client.stop_listening() for player in players),
            return_exceptions=True,
        )
