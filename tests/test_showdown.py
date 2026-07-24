from pathlib import Path

import pytest

from showdown_mind.showdown import ShowdownError, read_showdown_commit, server_command


def test_read_showdown_commit_skips_comments(tmp_path: Path) -> None:
    lock_file = tmp_path / "showdown.lock"
    lock_file.write_text("# pinned\n\nabc123\n", encoding="utf-8")

    assert read_showdown_commit(lock_file) == "abc123"


def test_read_showdown_commit_rejects_empty_lock(tmp_path: Path) -> None:
    lock_file = tmp_path / "showdown.lock"
    lock_file.write_text("# no commit\n", encoding="utf-8")

    with pytest.raises(ShowdownError, match="No commit"):
        read_showdown_commit(lock_file)


def test_server_command_is_local_runtime_entrypoint(tmp_path: Path) -> None:
    assert server_command(tmp_path) == (
        "node",
        str(tmp_path / "pokemon-showdown"),
        "start",
        "--no-security",
    )
