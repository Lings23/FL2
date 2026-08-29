from experiments.rtc_v3_semantic_test_evidence import build_test_evidence


def test_junit_evidence_counts_failures_and_errors(tmp_path, monkeypatch) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuites><testsuite tests="3" failures="1" errors="0" skipped="1"/></testsuites>',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "experiments.rtc_v3_semantic_test_evidence._git_output",
        lambda workspace, *args: "head\n" if args[:2] == ("rev-parse", "HEAD") else "",
    )
    evidence = build_test_evidence(workspace=tmp_path, junit_path=junit)
    assert evidence["tests"] == 3
    assert evidence["failures"] == 1
    assert evidence["passed"] is False

