from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from showdown_mind.experiment_artifacts import redact_secrets
from showdown_mind.paths import REPLAY_DIR

VIEWER_SCHEMA_VERSION = "1.0"
TITLE_PATTERN = re.compile(r"<title>\s*([^<]+?)\s*</title>", re.IGNORECASE)


class ViewerError(ValueError):
    """Raised when a decision log cannot become an unambiguous replay viewer."""


@dataclass(frozen=True)
class ViewerBuildResult:
    battle_id: str
    decisions: int
    replay_path: str
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_replay_viewer(
    decision_log: Path,
    *,
    replay_path: Path | None = None,
    output_path: Path | None = None,
    battle_id: str | None = None,
    replay_dir: Path = REPLAY_DIR,
    force: bool = False,
) -> ViewerBuildResult:
    records = _read_decision_log(decision_log)
    selected_battle = _select_battle(records, battle_id)
    selected_records = [
        record for record in records if record["battle_id"] == selected_battle
    ]
    selected_replay = (
        _validate_replay(replay_path, selected_battle)
        if replay_path is not None
        else _discover_replay(replay_dir, selected_battle)
    )
    destination = output_path or decision_log.with_suffix(".viewer.html")
    if destination.exists() and not force:
        raise ViewerError(
            f"Viewer output already exists: {destination}; pass --force to replace it"
        )

    payload = {
        "schema_version": VIEWER_SCHEMA_VERSION,
        "battle_id": selected_battle,
        "decision_log": decision_log.name,
        "replay": selected_replay.name,
        "decisions": [
            _viewer_decision(record, index)
            for index, record in enumerate(selected_records, start=1)
        ],
    }
    html = _render_html(
        payload,
        selected_replay.read_text(encoding="utf-8"),
    )
    _write_text_atomically(destination, html)
    return ViewerBuildResult(
        battle_id=selected_battle,
        decisions=len(selected_records),
        replay_path=str(selected_replay),
        output_path=str(destination),
    )


def _read_decision_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ViewerError(f"Decision log not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ViewerError(
                f"Invalid JSON on line {line_number} of {path}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ViewerError(
                f"Decision log line {line_number} must contain a JSON object"
            )
        if not isinstance(value.get("battle_id"), str) or not value["battle_id"]:
            raise ViewerError(f"Decision log line {line_number} has no battle_id")
        if not isinstance(value.get("snapshot"), dict):
            raise ViewerError(f"Decision log line {line_number} has no snapshot")
        records.append(value)
    if not records:
        raise ViewerError(f"Decision log contains no decisions: {path}")
    return records


def _select_battle(
    records: list[dict[str, Any]],
    requested: str | None,
) -> str:
    battle_ids = list(dict.fromkeys(str(record["battle_id"]) for record in records))
    if requested is not None:
        if requested not in battle_ids:
            available = ", ".join(battle_ids)
            raise ViewerError(
                f"Battle {requested!r} is not in the decision log; "
                f"available: {available}"
            )
        return requested
    if len(battle_ids) > 1:
        available = ", ".join(battle_ids)
        raise ViewerError(
            "Decision log contains multiple battles; pass --battle-id with one of: "
            f"{available}"
        )
    return battle_ids[0]


def _discover_replay(replay_dir: Path, battle_id: str) -> Path:
    if not replay_dir.is_dir():
        raise ViewerError(f"Replay directory not found: {replay_dir}")
    suffix = f" - {battle_id}.html"
    matches = sorted(
        path
        for path in replay_dir.iterdir()
        if path.is_file() and path.name.endswith(suffix)
    )
    if not matches:
        raise ViewerError(
            f"No replay found for {battle_id} in {replay_dir}; pass --replay"
        )
    research_player = [
        path for path in matches if path.name.startswith("ResearchPlayer ")
    ]
    return _validate_replay((research_player or matches)[0], battle_id)


def _validate_replay(path: Path, battle_id: str) -> Path:
    if not path.is_file():
        raise ViewerError(f"Replay not found: {path}")
    match = TITLE_PATTERN.search(path.read_text(encoding="utf-8"))
    replay_battle = match.group(1).strip() if match else None
    if replay_battle != battle_id:
        raise ViewerError(
            f"Replay title {replay_battle!r} does not match battle {battle_id!r}"
        )
    return path


def _viewer_decision(record: dict[str, Any], index: int) -> dict[str, Any]:
    snapshot = record["snapshot"]
    legal_actions = snapshot.get("legal_actions", [])
    if not isinstance(legal_actions, list):
        legal_actions = []
    action_id = str(record.get("action_id", ""))
    chosen_action = next(
        (
            action
            for action in legal_actions
            if isinstance(action, dict) and action.get("action_id") == action_id
        ),
        None,
    )
    usages = record.get("usages", [])
    if not isinstance(usages, list):
        usages = []
    valid_usages = [usage for usage in usages if isinstance(usage, dict)]
    errors = record.get("errors", [])
    if not isinstance(errors, list):
        errors = []
    return {
        "sequence": index,
        "turn": int(record.get("turn", snapshot.get("turn", 0))),
        "request_id": int(record.get("request_id", snapshot.get("request_id", 0))),
        "action_id": action_id,
        "chosen_action": chosen_action,
        "confidence": record.get("confidence"),
        "reason_codes": list(record.get("reason_codes") or []),
        "short_rationale": str(record.get("short_rationale") or ""),
        "attempts": int(record.get("attempts", 0)),
        "fallback_used": bool(record.get("fallback_used", False)),
        "errors": [redact_secrets(str(error)) for error in errors],
        "model_ids": [str(model) for model in record.get("model_ids") or []],
        "usage": {
            "input_tokens": sum(
                int(usage.get("input_tokens", 0)) for usage in valid_usages
            ),
            "output_tokens": sum(
                int(usage.get("output_tokens", 0)) for usage in valid_usages
            ),
            "total_tokens": sum(
                int(usage.get("total_tokens", 0)) for usage in valid_usages
            ),
        },
        "elapsed_seconds": float(record.get("elapsed_seconds", 0.0)),
        "policy_input": {
            "format": str(record.get("policy_input_format") or ""),
            "characters": int(record.get("policy_input_characters", 0)),
            "hash": str(record.get("policy_input_hash") or ""),
        },
        "snapshot": {
            "own_side": snapshot.get("own_side", {}),
            "opponent_side": snapshot.get("opponent_side", {}),
            "field": snapshot.get("field", {}),
            "resources": snapshot.get("resources", {}),
            "legal_actions": legal_actions,
        },
    }


def _render_html(payload: dict[str, Any], replay_html: str) -> str:
    package = files("showdown_mind.viewer_assets")
    template = package.joinpath("viewer.html").read_text(encoding="utf-8")
    stylesheet = package.joinpath("viewer.css").read_text(encoding="utf-8")
    script = package.joinpath("viewer.js").read_text(encoding="utf-8")
    encoded_payload = _encode_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    encoded_replay = _encode_text(replay_html)
    return (
        template.replace("/*__VIEWER_CSS__*/", stylesheet)
        .replace("/*__VIEWER_JS__*/", script)
        .replace("__VIEWER_DATA_BASE64__", encoded_payload)
        .replace("__REPLAY_HTML_BASE64__", encoded_replay)
    )


def _encode_text(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _write_text_atomically(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)
