from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from showdown_mind.domain import BattleSnapshot
from showdown_mind.policy_input import compile_policy_input


@dataclass(frozen=True)
class PromptBenchmarkResult:
    decisions: int
    full_characters: int
    pruned_characters: int
    compact_characters: int
    pruned_saved_characters: int
    compact_saved_characters: int
    pruned_reduction_percent: float
    compact_reduction_percent: float
    average_full_characters: float
    average_pruned_characters: float
    average_compact_characters: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_decision_log(path: Path) -> PromptBenchmarkResult:
    if not path.is_file():
        raise ValueError(f"Decision log does not exist: {path}")

    snapshots: list[BattleSnapshot] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            snapshots.append(BattleSnapshot.from_dict(row["snapshot"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid decision log row at line {line_number}") from exc

    if not snapshots:
        raise ValueError(f"Decision log contains no snapshots: {path}")

    full_characters = sum(
        compile_policy_input(snapshot, "full").characters for snapshot in snapshots
    )
    compact_characters = sum(
        compile_policy_input(snapshot, "compact").characters for snapshot in snapshots
    )
    pruned_characters = sum(
        compile_policy_input(snapshot, "pruned").characters for snapshot in snapshots
    )
    decisions = len(snapshots)
    pruned_saved = full_characters - pruned_characters
    compact_saved = full_characters - compact_characters
    return PromptBenchmarkResult(
        decisions=decisions,
        full_characters=full_characters,
        pruned_characters=pruned_characters,
        compact_characters=compact_characters,
        pruned_saved_characters=pruned_saved,
        compact_saved_characters=compact_saved,
        pruned_reduction_percent=round(
            pruned_saved / full_characters * 100,
            2,
        ),
        compact_reduction_percent=round(
            compact_saved / full_characters * 100,
            2,
        ),
        average_full_characters=round(full_characters / decisions, 2),
        average_pruned_characters=round(pruned_characters / decisions, 2),
        average_compact_characters=round(compact_characters / decisions, 2),
    )
