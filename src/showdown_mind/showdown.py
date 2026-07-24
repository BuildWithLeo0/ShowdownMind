from __future__ import annotations

import base64
import contextlib
import os
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from showdown_mind.paths import (
    SHOWDOWN_DIR,
    SHOWDOWN_LOCAL_CONFIG,
    SHOWDOWN_LOCK_FILE,
)


SHOWDOWN_REPOSITORY = "https://github.com/smogon/pokemon-showdown.git"
SHOWDOWN_HOST = "127.0.0.1"
SHOWDOWN_PORT = 8765


class ShowdownError(RuntimeError):
    """Raised when the local Pokémon Showdown runtime cannot be managed."""


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    stdout: str


def read_showdown_commit(lock_file: Path = SHOWDOWN_LOCK_FILE) -> str:
    """Read the pinned commit from a comment-friendly lock file."""
    for raw_line in lock_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            return line
    raise ShowdownError(f"No commit found in {lock_file}")


def run_command(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
) -> CommandResult:
    """Run an external command and surface useful output on failure."""
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        command = " ".join(args)
        raise ShowdownError(
            f"Command failed with exit code {completed.returncode}: {command}\n"
            f"{completed.stdout.rstrip()}"
        )
    return CommandResult(tuple(args), completed.stdout)


def setup_showdown(
    runtime_dir: Path = SHOWDOWN_DIR,
    lock_file: Path = SHOWDOWN_LOCK_FILE,
    local_config: Path = SHOWDOWN_LOCAL_CONFIG,
) -> str:
    """Clone, pin, install, and configure the local Showdown server."""
    commit = read_showdown_commit(lock_file)
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)

    if runtime_dir.exists() and not (runtime_dir / ".git").is_dir():
        raise ShowdownError(
            f"{runtime_dir} exists but is not a Git checkout; move it aside first"
        )

    if not runtime_dir.exists():
        run_command(
            [
                "git",
                "-c",
                "http.version=HTTP/1.1",
                "clone",
                "--filter=blob:none",
                SHOWDOWN_REPOSITORY,
                str(runtime_dir),
            ]
        )

    run_command(
        [
            "git",
            "-c",
            "http.version=HTTP/1.1",
            "fetch",
            "--depth",
            "1",
            "origin",
            commit,
        ],
        cwd=runtime_dir,
    )
    run_command(["git", "checkout", "--detach", commit], cwd=runtime_dir)
    run_command(["npm", "ci"], cwd=runtime_dir)

    destination = runtime_dir / "config" / "config.js"
    shutil.copy2(local_config, destination)
    return commit


def server_command(runtime_dir: Path = SHOWDOWN_DIR) -> tuple[str, ...]:
    return (
        "node",
        str(runtime_dir / "pokemon-showdown"),
        "start",
        "--no-security",
    )


def server_is_healthy(
    host: str = SHOWDOWN_HOST,
    port: int = SHOWDOWN_PORT,
    *,
    timeout: float = 0.25,
) -> bool:
    websocket_key = base64.b64encode(b"showdown-mind-ok").decode("ascii")
    request = (
        "GET /showdown/websocket HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Connection: Upgrade\r\n"
        "Upgrade: websocket\r\n"
        f"Sec-WebSocket-Key: {websocket_key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).encode("ascii")
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.sendall(request)
            response = connection.recv(256)
            return response.startswith(b"HTTP/1.1 101")
    except OSError:
        return False


def wait_for_server(
    *,
    timeout: float = 60.0,
    host: str = SHOWDOWN_HOST,
    port: int = SHOWDOWN_PORT,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server_is_healthy(host, port):
            return
        time.sleep(0.25)
    raise ShowdownError(
        f"Pokémon Showdown did not become healthy at {host}:{port} "
        f"within {timeout:.0f} seconds"
    )


def start_showdown(runtime_dir: Path = SHOWDOWN_DIR) -> int:
    """Run the local server in the foreground."""
    if not (runtime_dir / "pokemon-showdown").is_file():
        raise ShowdownError("Showdown is not installed; run `showdown setup` first")
    return subprocess.call(server_command(runtime_dir), cwd=runtime_dir)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


@contextlib.contextmanager
def managed_showdown_server(
    runtime_dir: Path = SHOWDOWN_DIR,
) -> Iterator[bool]:
    """Start a temporary local server unless one is already running."""
    if server_is_healthy():
        yield False
        return

    if not (runtime_dir / "pokemon-showdown").is_file():
        raise ShowdownError("Showdown is not installed; run `showdown setup` first")

    log_dir = runtime_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "showdown.log"

    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            server_command(runtime_dir),
            cwd=runtime_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_server()
            yield True
        except Exception as exc:
            tail = _read_log_tail(log_path)
            if isinstance(exc, ShowdownError) and tail:
                raise ShowdownError(f"{exc}\nServer log:\n{tail}") from exc
            raise
        finally:
            _terminate_process_group(process)


def _read_log_tail(path: Path, max_bytes: int = 4_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        return stream.read().decode("utf-8", errors="replace").strip()
