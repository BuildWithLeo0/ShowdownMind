from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from poke_env.player import (
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.ps_client import ServerConfiguration

from showdown_mind.paths import REPLAY_DIR
from showdown_mind.showdown import SHOWDOWN_HOST, SHOWDOWN_PORT


BATTLE_FORMAT = "gen9randombattle"
LOCAL_SERVER_CONFIGURATION = ServerConfiguration(
    f"ws://{SHOWDOWN_HOST}:{SHOWDOWN_PORT}/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)
BASELINE_TYPES = {
    "random": RandomPlayer,
    "max-base-power": MaxBasePowerPlayer,
    "simple-heuristics": SimpleHeuristicsPlayer,
}


class BaselineError(RuntimeError):
    """Raised when a baseline experiment does not complete as requested."""


@dataclass(frozen=True)
class SmokeResult:
    battle_format: str
    player: str
    opponent: str
    requested_battles: int
    finished_battles: int
    player_wins: int
    opponent_wins: int
    draws: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_baseline(name: str, replay_dir: Path = REPLAY_DIR):
    try:
        player_type = BASELINE_TYPES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(BASELINE_TYPES))
        raise ValueError(f"Unknown baseline {name!r}; choose one of: {choices}") from exc

    replay_dir.mkdir(parents=True, exist_ok=True)
    return player_type(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
        save_replays=str(replay_dir),
        server_configuration=LOCAL_SERVER_CONFIGURATION,
    )


def ensure_localhost_bypasses_proxy() -> None:
    """Keep local WebSocket traffic out of system HTTP/SOCKS proxies."""
    required = {"localhost", "127.0.0.1", "::1"}
    for variable in ("NO_PROXY", "no_proxy"):
        existing = {
            entry.strip()
            for entry in os.environ.get(variable, "").split(",")
            if entry.strip()
        }
        os.environ[variable] = ",".join(sorted(existing | required))


async def run_baseline_battles(
    *,
    player_name: str = "random",
    opponent_name: str = "random",
    battles: int = 1,
) -> SmokeResult:
    if battles < 1:
        raise ValueError("battles must be at least 1")

    ensure_localhost_bypasses_proxy()
    player = make_baseline(player_name)
    opponent = make_baseline(opponent_name)
    started = time.monotonic()
    try:
        try:
            await asyncio.wait_for(
                player.battle_against(opponent, n_battles=battles),
                timeout=max(60.0, battles * 30.0),
            )
        except TimeoutError as exc:
            raise BaselineError(
                f"Baseline battle timed out after requesting {battles} battles"
            ) from exc
    finally:
        await asyncio.gather(
            player.ps_client.stop_listening(),
            opponent.ps_client.stop_listening(),
            return_exceptions=True,
        )
    elapsed = time.monotonic() - started

    finished = player.n_finished_battles
    if finished != battles:
        raise BaselineError(
            f"Requested {battles} battles but only {finished} finished"
        )
    player_wins = player.n_won_battles
    opponent_wins = opponent.n_won_battles
    draws = finished - player_wins - opponent_wins
    return SmokeResult(
        battle_format=BATTLE_FORMAT,
        player=player_name,
        opponent=opponent_name,
        requested_battles=battles,
        finished_battles=finished,
        player_wins=player_wins,
        opponent_wins=opponent_wins,
        draws=draws,
        elapsed_seconds=round(elapsed, 3),
    )
