from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from showdown_mind.paths import PROJECT_ROOT
from showdown_mind.showdown import read_showdown_commit

ARTIFACT_SCHEMA_VERSION = "1.0"
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)\bBearer\s+\S+"),
    re.compile(r"(?i)(api[_-]?key=)[^&\s]+"),
    re.compile(r"(https?://)[^/@\s]+:[^/@\s]+@"),
)


@dataclass(frozen=True)
class ArtifactPaths:
    decisions: Path
    manifest: Path
    summary: Path
    failure: Path

    @classmethod
    def from_decision_log(cls, decision_log: Path) -> ArtifactPaths:
        return cls(
            decisions=decision_log,
            manifest=decision_log.with_suffix(".manifest.json"),
            summary=decision_log.with_suffix(".summary.json"),
            failure=decision_log.with_suffix(".failure.json"),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    battle_format: str
    opponent: str
    requested_battles: int
    prompt_format: str
    timeout_seconds: float


class ExperimentArtifactWriter:
    def __init__(self, decision_log: Path):
        self.paths = ArtifactPaths.from_decision_log(decision_log)

    def assert_new_run(self) -> None:
        occupied = [path for path in asdict(self.paths).values() if Path(path).exists()]
        if occupied:
            paths = ", ".join(str(path) for path in occupied)
            raise ValueError(
                f"Experiment artifact path already exists; choose a new log: {paths}"
            )

    def write_manifest(self, model_client: Any, spec: ExperimentSpec) -> None:
        git_commit, git_dirty = git_state()
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": self.paths.decisions.stem,
            "started_at": datetime.now(UTC).isoformat(),
            "battle": {
                "format": spec.battle_format,
                "opponent": spec.opponent,
                "requested_battles": spec.requested_battles,
                "timeout_seconds": spec.timeout_seconds,
            },
            "policy": {"input_format": spec.prompt_format},
            "model": {
                "model_id": str(
                    getattr(model_client, "model_id", type(model_client).__name__)
                ),
                "client_type": (
                    f"{type(model_client).__module__}.{type(model_client).__qualname__}"
                ),
                "base_url": _sanitize_provider_url(
                    getattr(model_client, "base_url", None)
                ),
            },
            "software": {
                "showdown_mind_git_commit": git_commit,
                "showdown_mind_git_dirty": git_dirty,
                "pokemon_showdown_commit": read_showdown_commit(),
                "python": platform.python_version(),
                "packages": {
                    "openai": _package_version("openai"),
                    "poke-env": _package_version("poke-env"),
                },
            },
            "artifacts": {
                "decisions": str(self.paths.decisions),
                "summary": str(self.paths.summary),
                "failure": str(self.paths.failure),
            },
        }
        _write_json_atomically(self.paths.manifest, manifest)

    def write_summary(self, summary: dict[str, Any]) -> None:
        _write_json_atomically(self.paths.summary, summary)

    def write_failure(self, error: Exception) -> None:
        failure = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "failed_at": datetime.now(UTC).isoformat(),
            "error_type": type(error).__name__,
            "message": redact_secrets(str(error))[:1000],
        }
        _write_json_atomically(self.paths.failure, failure)


def _sanitize_provider_url(value: Any) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(str(value))
    if not parsed.scheme or not parsed.hostname:
        return None
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def redact_secrets(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        if pattern.pattern.startswith("(https?://)"):
            redacted = pattern.sub(r"\1[REDACTED]@", redacted)
        elif pattern.pattern.startswith("(?i)(api"):
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def _write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
