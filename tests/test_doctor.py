from showdown_mind.doctor import Check, doctor_succeeded


def test_doctor_allows_warnings() -> None:
    assert doctor_succeeded([Check("runtime", "warning", "not installed")])


def test_doctor_rejects_errors() -> None:
    assert not doctor_succeeded([Check("python", "error", "unsupported")])
