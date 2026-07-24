from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from showdown_mind.agent_runner import run_agent_battles
from showdown_mind.baselines import (
    BASELINE_TYPES,
    BaselineError,
    run_baseline_battles,
)
from showdown_mind.doctor import collect_checks, doctor_succeeded
from showdown_mind.models import DeterministicModelClient
from showdown_mind.showdown import (
    ShowdownError,
    managed_showdown_server,
    setup_showdown,
    start_showdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="showdown-mind",
        description="Research harness for Pokémon Showdown agents",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="check local dependencies and runtime")

    showdown = subparsers.add_parser("showdown", help="manage the local server")
    showdown_subparsers = showdown.add_subparsers(
        dest="showdown_command",
        required=True,
    )
    showdown_subparsers.add_parser("setup", help="install the pinned server")
    showdown_subparsers.add_parser("start", help="run the server in foreground")

    smoke = subparsers.add_parser("smoke", help="run baseline battles")
    choices = sorted(BASELINE_TYPES)
    smoke.add_argument("--player", choices=choices, default="random")
    smoke.add_argument("--opponent", choices=choices, default="random")
    smoke.add_argument("--battles", type=int, default=1)
    smoke.add_argument(
        "--no-manage-server",
        action="store_true",
        help="require an already-running server instead of starting one",
    )

    agent_smoke = subparsers.add_parser(
        "agent-smoke",
        help="exercise the Policy-first agent with a deterministic model double",
    )
    agent_smoke.add_argument(
        "--opponent",
        choices=choices,
        default="max-base-power",
    )
    agent_smoke.add_argument("--battles", type=int, default=1)
    agent_smoke.add_argument(
        "--decision-log",
        type=Path,
        help="write per-turn JSONL records to this path",
    )
    agent_smoke.add_argument(
        "--no-manage-server",
        action="store_true",
        help="require an already-running server instead of starting one",
    )
    return parser


def _run_doctor() -> int:
    checks = collect_checks()
    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"{check.name:<{width}}  {check.status:<7}  {check.detail}")
    return 0 if doctor_succeeded(checks) else 1


def _run_smoke(args: argparse.Namespace) -> int:
    async def execute() -> None:
        result = await run_baseline_battles(
            player_name=args.player,
            opponent_name=args.opponent,
            battles=args.battles,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    if args.no_manage_server:
        asyncio.run(execute())
    else:
        with managed_showdown_server():
            asyncio.run(execute())
    return 0


def _run_agent_smoke(args: argparse.Namespace) -> int:
    async def execute() -> None:
        result = await run_agent_battles(
            DeterministicModelClient(),
            opponent_name=args.opponent,
            battles=args.battles,
            decision_log=args.decision_log,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    if args.no_manage_server:
        asyncio.run(execute())
    else:
        with managed_showdown_server():
            asyncio.run(execute())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            return _run_doctor()
        if args.command == "showdown":
            if args.showdown_command == "setup":
                commit = setup_showdown()
                print(f"Pokémon Showdown is ready at commit {commit}")
                return 0
            if args.showdown_command == "start":
                return start_showdown()
        if args.command == "smoke":
            return _run_smoke(args)
        if args.command == "agent-smoke":
            return _run_agent_smoke(args)
    except (BaselineError, ShowdownError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error("unsupported command")
    return 2
