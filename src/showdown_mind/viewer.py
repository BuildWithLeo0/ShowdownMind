from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from showdown_mind.experiment_artifacts import redact_secrets
from showdown_mind.models import ACTION_TOOL_NAME
from showdown_mind.paths import REPLAY_DIR

VIEWER_SCHEMA_VERSION = "1.2"
TITLE_PATTERN = re.compile(r"<title>\s*([^<]+?)\s*</title>", re.IGNORECASE)
LOG_PATTERN = re.compile(
    r'<script[^>]*class=["\'][^"\']*\bbattle-log-data\b[^"\']*["\'][^>]*>'
    r"(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)
RESEARCH_PLAYER_PATTERN = re.compile(
    r"^\|player\|(p[12])\|ResearchPlayer\b",
    re.IGNORECASE,
)


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

    replay_html = selected_replay.read_text(encoding="utf-8")
    decisions = [
        _viewer_decision(record, index)
        for index, record in enumerate(selected_records, start=1)
    ]
    protocol_lines = _extract_protocol_lines(replay_html)
    agent_side = _find_agent_side(protocol_lines)
    anchors = _build_replay_anchors(decisions, protocol_lines, agent_side)
    for decision, anchor in zip(decisions, anchors, strict=True):
        decision["replay_step"] = anchor

    payload = {
        "schema_version": VIEWER_SCHEMA_VERSION,
        "battle_id": selected_battle,
        "decision_log": decision_log.name,
        "replay": selected_replay.name,
        "replay_sync": {
            "strategy": "protocol-step-v1",
            "agent_side": agent_side,
            "protocol_steps": len(protocol_lines),
            "anchored_decisions": sum(anchor is not None for anchor in anchors),
        },
        "decisions": decisions,
    }
    html = _render_html(payload, replay_html)
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
    tool_executions = record.get("tool_executions", [])
    if not isinstance(tool_executions, list):
        tool_executions = []
    tactical_analysis = record.get("tactical_analysis", {})
    if not isinstance(tactical_analysis, dict):
        tactical_analysis = {}
    planner_usages = record.get("planner_usages", [])
    if not isinstance(planner_usages, list):
        planner_usages = []
    valid_planner_usages = [
        usage for usage in planner_usages if isinstance(usage, dict)
    ]
    controlled_fields = {}
    for key, default in (
        ("memory", {}),
        ("belief_state", {}),
        ("battle_plan", {}),
        ("plan_update", {}),
        ("opponent_prediction", {}),
    ):
        value = record.get(key, default)
        controlled_fields[key] = value if isinstance(value, dict) else default
    for key in ("new_events", "belief_changes"):
        value = record.get(key, [])
        controlled_fields[key] = value if isinstance(value, list) else []
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
        "model_calls": int(
            record.get("model_calls") or record.get("attempts", 0)
        ),
        "expected_model_calls": int(record.get("expected_model_calls", 1)),
        "fallback_used": bool(record.get("fallback_used", False)),
        "errors": [redact_secrets(str(error)) for error in errors],
        "model_ids": [str(model) for model in record.get("model_ids") or []],
        "tool": {
            "name": str(record.get("tool_name") or ACTION_TOOL_NAME),
            "names": [
                str(tool_name) for tool_name in record.get("tool_names") or []
            ],
            "call_ids": [
                str(tool_call_id) for tool_call_id in record.get("tool_call_ids") or []
            ],
            "executions": [
                {
                    "tool_name": str(execution.get("tool_name") or ""),
                    "tool_call_id": str(execution.get("tool_call_id") or ""),
                    "execution_kind": str(
                        execution.get("execution_kind") or "model_tool_call"
                    ),
                    "arguments": execution.get("arguments", {}),
                }
                for execution in tool_executions
                if isinstance(execution, dict)
            ],
        },
        "tactical_analysis": tactical_analysis,
        **controlled_fields,
        "plan_trigger": str(record.get("plan_trigger") or ""),
        "request_replan": bool(record.get("request_replan", False)),
        "enrichment_errors": [
            redact_secrets(str(error))
            for error in record.get("enrichment_errors") or []
        ],
        "decision_normalizations": [
            str(value)
            for value in record.get("decision_normalizations") or []
        ],
        "planner": {
            "model_calls": int(record.get("planner_model_calls", 0)),
            "failed": bool(record.get("planner_failed", False)),
            "elapsed_seconds": float(
                record.get("planner_elapsed_seconds", 0.0)
            ),
            "errors": [
                redact_secrets(str(error))
                for error in record.get("planner_errors") or []
            ],
            "usage": {
                "input_tokens": sum(
                    int(usage.get("input_tokens", 0))
                    for usage in valid_planner_usages
                ),
                "output_tokens": sum(
                    int(usage.get("output_tokens", 0))
                    for usage in valid_planner_usages
                ),
                "total_tokens": sum(
                    int(usage.get("total_tokens", 0))
                    for usage in valid_planner_usages
                ),
            },
        },
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


def _extract_protocol_lines(replay_html: str) -> list[str]:
    match = LOG_PATTERN.search(replay_html)
    if not match:
        raise ViewerError("Replay contains no battle-log-data protocol")
    return match.group(1).replace("\\/", "/").split("\n")


def _find_agent_side(protocol_lines: list[str]) -> str:
    for line in protocol_lines:
        match = RESEARCH_PLAYER_PATTERN.match(line)
        if match:
            return match.group(1).lower()
    raise ViewerError("Replay does not identify a ResearchPlayer side")


def _build_replay_anchors(
    decisions: list[dict[str, Any]],
    protocol_lines: list[str],
    agent_side: str,
) -> list[int | None]:
    turn_steps = {
        int(line.removeprefix("|turn|")): index + 1
        for index, line in enumerate(protocol_lines)
        if line.startswith("|turn|") and line.removeprefix("|turn|").isdigit()
    }
    anchors: list[int | None] = []
    seen_turns: dict[int, int] = {}
    for decision in decisions:
        turn = int(decision["turn"])
        occurrence = seen_turns.get(turn, 0)
        if occurrence == 0:
            anchor = turn_steps.get(turn)
        else:
            previous = next(
                (
                    prior
                    for prior in reversed(anchors)
                    if prior is not None
                ),
                turn_steps.get(turn),
            )
            next_turn_step = min(
                (
                    step
                    for later_turn, step in turn_steps.items()
                    if later_turn > turn
                ),
                default=len(protocol_lines) + 1,
            )
            anchor = _find_action_step(
                protocol_lines,
                action_id=str(decision["action_id"]),
                agent_side=agent_side,
                start_step=previous or 0,
                end_step=next_turn_step - 1,
            )
        anchors.append(anchor)
        seen_turns[turn] = occurrence + 1
    return anchors


def _find_action_step(
    protocol_lines: list[str],
    *,
    action_id: str,
    agent_side: str,
    start_step: int,
    end_step: int,
) -> int | None:
    for index in range(max(start_step, 0), min(end_step, len(protocol_lines))):
        if _action_matches_line(action_id, protocol_lines[index], agent_side):
            return index + 1
    return None


def _action_matches_line(action_id: str, line: str, agent_side: str) -> bool:
    parts = line.split("|")
    if len(parts) < 4:
        return False
    actor = parts[2].lower()
    if not actor.startswith(agent_side):
        return False
    if action_id.startswith("move:") and parts[1] == "move":
        move_id = action_id.split(":", 2)[1]
        return _to_id(parts[3]) == move_id
    if action_id.startswith("switch:") and parts[1] in {"switch", "drag", "replace"}:
        species_id = action_id.split(":", 1)[1]
        replay_species = parts[3].split(",", 1)[0]
        return _to_id(replay_species) == species_id
    return False


def _to_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


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
