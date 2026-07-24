import os

import pytest

from showdown_mind.baselines import (
    BASELINE_TYPES,
    ensure_localhost_bypasses_proxy,
    make_baseline,
)


def test_expected_baselines_are_available() -> None:
    assert set(BASELINE_TYPES) == {
        "random",
        "max-base-power",
        "simple-heuristics",
    }


def test_unknown_baseline_has_helpful_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown baseline"):
        make_baseline("missing", replay_dir=tmp_path)


def test_localhost_bypasses_proxy(monkeypatch) -> None:
    monkeypatch.setenv("NO_PROXY", "example.com")
    monkeypatch.delenv("no_proxy", raising=False)

    ensure_localhost_bypasses_proxy()

    for variable in ("NO_PROXY", "no_proxy"):
        entries = set(os.environ[variable].split(","))
        assert {"localhost", "127.0.0.1", "::1"} <= entries
