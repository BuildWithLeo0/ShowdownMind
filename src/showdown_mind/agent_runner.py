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
from showdown_mind.models import ModelClient
from showdown_mind.paths import REPLAY_DIR, RUNTIME_DIR
from showdown_mind.policy import SingleCallPolicy
from showdown_mind.storage import JsonlDecisionWriter


@dataclass(frozen=True)
class AgentSmokeResult:
    battle_format: str
    opponent: str
    requested_battles: int
    finished_battles: int
    agent_wins: int
    opponent_wins: int
    draws: int
    decisions: int
    fallbacks: int
    elapsed_seconds: float
    decision_log: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_agent_battles(
    model_client: ModelClient,
    *,
    opponent_name: str = "max-base-power",
    battles: int = 1,
    decision_log: Path | None = None,
) -> AgentSmokeResult:
    if battles < 1:
        raise ValueError("battles must be at least 1")

    ensure_localhost_bypasses_proxy()
    log_path = decision_log or (
        RUNTIME_DIR / "decisions" / f"agent-smoke-{time.time_ns()}.jsonl"
    )
    writer = JsonlDecisionWriter(log_path)
    agent = ResearchPlayer(
        SingleCallPolicy(model_client),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
        save_replays=str(REPLAY_DIR),
        server_configuration=LOCAL_SERVER_CONFIGURATION,
        decision_sink=writer,
    )
    opponent = make_baseline(opponent_name)
    started = time.monotonic()
    try:
        try:
            await asyncio.wait_for(
                agent.battle_against(opponent, n_battles=battles),
                timeout=max(60.0, battles * 30.0),
            )
        except TimeoutError as exc:
            raise BaselineError(
                f"Agent battle timed out after requesting {battles} battles"
            ) from exc
    finally:
        await asyncio.gather(
            agent.ps_client.stop_listening(),
            opponent.ps_client.stop_listening(),
            return_exceptions=True,
        )

    finished = agent.n_finished_battles
    if finished != battles:
        raise BaselineError(f"Requested {battles} battles but only {finished} finished")

    agent_wins = agent.n_won_battles
    opponent_wins = opponent.n_won_battles
    return AgentSmokeResult(
        battle_format=BATTLE_FORMAT,
        opponent=opponent_name,
        requested_battles=battles,
        finished_battles=finished,
        agent_wins=agent_wins,
        opponent_wins=opponent_wins,
        draws=finished - agent_wins - opponent_wins,
        decisions=len(agent.decision_records),
        fallbacks=sum(record.fallback_used for record in agent.decision_records),
        elapsed_seconds=round(time.monotonic() - started, 3),
        decision_log=str(log_path),
    )
