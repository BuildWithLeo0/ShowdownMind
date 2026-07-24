from __future__ import annotations

import argparse
import asyncio
import json
import sys
import webbrowser
from pathlib import Path

from showdown_mind.agent_runner import run_agent_battles
from showdown_mind.baselines import (
    BASELINE_TYPES,
    BaselineError,
    run_baseline_battles,
)
from showdown_mind.doctor import collect_checks, doctor_succeeded
from showdown_mind.model_runner import run_model_check
from showdown_mind.models import (
    DeterministicModelClient,
    OpenAICompatibleModelClient,
    live_model_client_from_env,
)
from showdown_mind.policy import PolicyFailure
from showdown_mind.policy_input import POLICY_INPUT_FORMATS
from showdown_mind.prompt_benchmark import benchmark_decision_log
from showdown_mind.showdown import (
    ShowdownError,
    managed_showdown_server,
    setup_showdown,
    start_showdown,
)
from showdown_mind.viewer import build_replay_viewer


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
        "--prompt-format",
        choices=POLICY_INPUT_FORMATS,
        default="pruned",
    )
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

    model_check = subparsers.add_parser(
        "model-check",
        help="make one live model call and validate its decision",
    )
    model_check.add_argument(
        "--prompt-format",
        choices=POLICY_INPUT_FORMATS,
        default="pruned",
    )

    llm_smoke = subparsers.add_parser(
        "llm-smoke",
        help="run Policy-first battles with the configured live model",
    )
    llm_smoke.add_argument(
        "--opponent",
        choices=choices,
        default="max-base-power",
    )
    llm_smoke.add_argument("--battles", type=int, default=1)
    llm_smoke.add_argument(
        "--prompt-format",
        choices=POLICY_INPUT_FORMATS,
        default="pruned",
    )
    llm_smoke.add_argument(
        "--battle-timeout",
        type=float,
        default=300.0,
        help="maximum seconds for the complete batch (default: 300)",
    )
    llm_smoke.add_argument(
        "--decision-log",
        type=Path,
        help="write per-turn JSONL records to this path",
    )
    llm_smoke.add_argument(
        "--no-manage-server",
        action="store_true",
        help="require an already-running server instead of starting one",
    )

    prompt_benchmark = subparsers.add_parser(
        "prompt-benchmark",
        help="compare full, pruned, and compact inputs from a decision log",
    )
    prompt_benchmark.add_argument("decision_log", type=Path)

    visualize = subparsers.add_parser(
        "visualize",
        help="build a local native replay and Agent decision viewer",
    )
    visualize.add_argument("decision_log", type=Path)
    visualize.add_argument(
        "--replay",
        type=Path,
        help="use this replay HTML instead of discovering it by battle ID",
    )
    visualize.add_argument(
        "--battle-id",
        help="select one battle when the decision log contains several",
    )
    visualize.add_argument(
        "--output",
        type=Path,
        help="write the generated viewer to this HTML path",
    )
    visualize.add_argument(
        "--no-open",
        action="store_true",
        help="generate the viewer without opening the browser",
    )
    visualize.add_argument(
        "--force",
        action="store_true",
        help="replace an existing viewer output",
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
            prompt_format=args.prompt_format,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    if args.no_manage_server:
        asyncio.run(execute())
    else:
        with managed_showdown_server():
            asyncio.run(execute())
    return 0


def _run_model_check(args: argparse.Namespace) -> int:
    async def execute() -> None:
        client = live_model_client_from_env()
        try:
            result = await run_model_check(
                client,
                prompt_format=args.prompt_format,
            )
        finally:
            await client.aclose()
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    asyncio.run(execute())
    return 0


def _run_llm_smoke(args: argparse.Namespace) -> int:
    async def execute(client: OpenAICompatibleModelClient) -> None:
        check = await run_model_check(
            client,
            prompt_format=args.prompt_format,
        )
        print(
            json.dumps(
                {"model_check": check.to_dict()},
                indent=2,
                sort_keys=True,
            )
        )
        result = await run_agent_battles(
            client,
            opponent_name=args.opponent,
            battles=args.battles,
            decision_log=args.decision_log,
            timeout_seconds=args.battle_timeout,
            prompt_format=args.prompt_format,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    async def run_and_close() -> None:
        client = live_model_client_from_env()
        try:
            await execute(client)
        finally:
            await client.aclose()

    if args.no_manage_server:
        asyncio.run(run_and_close())
    else:
        with managed_showdown_server():
            asyncio.run(run_and_close())
    return 0


def _run_prompt_benchmark(args: argparse.Namespace) -> int:
    result = benchmark_decision_log(args.decision_log)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _run_visualize(args: argparse.Namespace) -> int:
    result = build_replay_viewer(
        args.decision_log,
        replay_path=args.replay,
        output_path=args.output,
        battle_id=args.battle_id,
        force=args.force,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if not args.no_open:
        webbrowser.open(Path(result.output_path).resolve().as_uri())
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
        if args.command == "model-check":
            return _run_model_check(args)
        if args.command == "llm-smoke":
            return _run_llm_smoke(args)
        if args.command == "prompt-benchmark":
            return _run_prompt_benchmark(args)
        if args.command == "visualize":
            return _run_visualize(args)
    except PolicyFailure as exc:
        detail = exc.errors[-1] if exc.errors else str(exc)
        print(f"error: model policy failed: {detail}", file=sys.stderr)
        return 1
    except (BaselineError, ShowdownError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error("unsupported command")
    return 2
