import json

import pandas as pd
import pytest

from analysis.rtc_anchor_recycle_b3.analyze_results import (
    _metrics,
    _validate_completion,
)
from analysis.rtc_residual_rank_cap_b4.analyze_results import _rank_cap_audit


def _write_status(
    tmp_path, *, state="completed", exit_code=0, last_round=60
):
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    path = status_dir / "dba__continuous__rtc_cumulative_q_cap_anchor__iid.json"
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


@pytest.mark.parametrize("state", ["completed", "completed_cached"])
def test_validate_completion_accepts_terminal_success(tmp_path, state):
    _write_status(tmp_path, state=state)

    _validate_completion(tmp_path, "dba", "rtc_cumulative_q_cap_anchor")


@pytest.mark.parametrize(
    ("state", "exit_code", "last_round"),
    [
        ("running", None, 21),
        ("failed", 1, 60),
        ("completed_cached", 1, 60),
        ("completed_cached", 0, 59),
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

    with pytest.raises(RuntimeError, match="is incomplete"):
        _validate_completion(tmp_path, "dba", "rtc_cumulative_q_cap_anchor")


def test_metrics_treats_missing_asr_as_zero_for_untargeted_attack(tmp_path):
    path = tmp_path / "rounds.csv"
    active = [0] * 11 + [1] * 50
    pd.DataFrame(
        {
            "round": list(range(61)),
            "planned_attack_active": active,
            "server_accuracy": [0.8] * 61,
            "fit_malicious_aggregation_weight_share": [0.1] * 61,
            "fit_rtc_v3_weight_sum": [0.9] * 61,
            "fit_malicious_impact_share": [0.2] * 61,
        }
    ).to_csv(path, index=False)

    metrics = _metrics(path, "lie", "rtc_b4_residual_rank_cap")

    assert metrics["active_asr"] == 0.0
    assert metrics["peak_asr"] == 0.0


def test_b4_rank_cap_audit_reconstructs_identity_and_benign_hit_rate():
    rounds = pd.DataFrame(
        {
            "round": [11],
            "planned_attack_active": [1],
            "fit_rtc_v3_residual_rank_cap_client_ids": ["z|a"],
        }
    )
    clients = pd.DataFrame(
        {
            "round": [11, 11, 11],
            "cid": ["z", "a", "m"],
            "principal_id": ["z", "a", "m"],
            "is_malicious": [True, False, False],
            "rtc_v3_residual_norm": [3.0, 2.0, 1.0],
            "rtc_v3_client_q_cap": [0.5, 0.4, 1.0],
        }
    )

    audit, hits = _rank_cap_audit(rounds, clients)

    assert audit["rank_identity_matches_all_rounds"] is True
    assert audit["rank_cap_guard_passes_all_rounds"] is True
    assert audit["rank_cap_hits"] == 2
    assert audit["rank_cap_malicious_precision"] == pytest.approx(0.5)
    assert audit["benign_rank_cap_rate"] == pytest.approx(0.5)
    assert set(hits["cid"]) == {"z", "a"}


def test_b4_rank_cap_audit_rejects_wrong_runtime_ids_and_cap():
    rounds = pd.DataFrame(
        {
            "round": [11],
            "planned_attack_active": [1],
            "fit_rtc_v3_residual_rank_cap_client_ids": ["a|m"],
        }
    )
    clients = pd.DataFrame(
        {
            "round": [11, 11, 11],
            "cid": ["z", "a", "m"],
            "principal_id": ["z", "a", "m"],
            "is_malicious": [True, False, False],
            "rtc_v3_residual_norm": [3.0, 2.0, 1.0],
            "rtc_v3_client_q_cap": [0.5, 0.6, 0.5],
        }
    )

    audit, _ = _rank_cap_audit(rounds, clients)

    assert audit["rank_identity_matches_all_rounds"] is False
    assert audit["rank_cap_guard_passes_all_rounds"] is False
