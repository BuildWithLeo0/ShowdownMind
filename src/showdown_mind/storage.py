from __future__ import annotations

import json
from pathlib import Path

from showdown_mind.domain import DecisionRecord


class JsonlDecisionWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record: DecisionRecord) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    record.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            stream.write("\n")
