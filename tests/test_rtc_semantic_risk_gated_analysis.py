import json

import pytest

from analysis.rtc_semantic_risk_gated.analyze_results import _validate_completion


def _write_status(tmp_path, *, state="completed", exit_code=0, last_round=60):
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    path = status_dir / "lie__continuous__rtc_semantic_restricted_only__iid.json"
    path.write_text(
        json.dumps(
            {
                "state": state,
                "exit_code": exit_code,
                "last_round": last_round,
            }
        ),
        encoding="utf-8",
    )


def test_validate_completion_accepts_terminal_success(tmp_path):
    _write_status(tmp_path)

    _validate_completion(tmp_path, "lie", "rtc_semantic_restricted_only")


@pytest.mark.parametrize(
    ("state", "exit_code", "last_round"),
    [
        ("running", None, 21),
        ("completed", 1, 60),
        ("completed", 0, 59),
    ],
)
def test_validate_completion_rejects_nonterminal_or_incomplete_run(
    tmp_path, state, exit_code, last_round
):
    _write_status(
        tmp_path,
        state=state,
        exit_code=exit_code,
        last_round=last_round,
    )

    with pytest.raises(RuntimeError, match="is not complete"):
        _validate_completion(tmp_path, "lie", "rtc_semantic_restricted_only")
