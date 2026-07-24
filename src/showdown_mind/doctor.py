from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
from dataclasses import dataclass

from showdown_mind.paths import SHOWDOWN_DIR
from showdown_mind.showdown import SHOWDOWN_HOST, SHOWDOWN_PORT, server_is_healthy


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _command_version(command: str, argument: str = "--version") -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, argument],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip() or f"exit {completed.returncode}"


def collect_checks() -> list[Check]:
    checks: list[Check] = []

    python_ok = (3, 12) <= sys.version_info[:2] < (3, 15)
    checks.append(
        Check(
            "python",
            "ok" if python_ok else "error",
            sys.version.split()[0],
        )
    )

    for command in ("node", "npm", "git"):
        version = _command_version(command)
        checks.append(
            Check(
                command,
                "ok" if version else "error",
                version or "not found",
            )
        )

    try:
        poke_env_version = importlib.metadata.version("poke-env")
    except importlib.metadata.PackageNotFoundError:
        checks.append(Check("poke-env", "error", "not installed"))
    else:
        checks.append(Check("poke-env", "ok", poke_env_version))

    runtime_ready = (SHOWDOWN_DIR / "pokemon-showdown").is_file()
    checks.append(
        Check(
            "showdown-runtime",
            "ok" if runtime_ready else "warning",
            str(SHOWDOWN_DIR) if runtime_ready else "run `showdown setup`",
        )
    )
    checks.append(
        Check(
            "showdown-server",
            "ok" if server_is_healthy() else "warning",
            (
                f"{SHOWDOWN_HOST}:{SHOWDOWN_PORT}"
                if server_is_healthy()
                else "not running"
            ),
        )
    )
    return checks


def doctor_succeeded(checks: list[Check]) -> bool:
    return all(check.status != "error" for check in checks)
