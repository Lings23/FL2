"""Phase 4 gates for principal-first selection and participation ledgers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from config.config_loader import DefenseConfig
from attacks.attack_client import MPAFClient, get_attack_client_class
from defenses.rtc.calibration import build_manifest
from defenses.rtc.v3 import RTCv3Defense
from strategies.fed_strategy import FedSecStrategy
from experiments.periodic_attack import build_matrix


def _params(value=0.0):
    return [np.asarray([value], dtype=np.float32)]


def _manifest(*, windows=(1, 3), betas=None, mass_policy="unit_principal"):
    return build_manifest(
        params=_params(),
        clip_lower=10.0,
        clip_upper=10.0,
        validated_mass_policy=mass_policy,
        resolutions=("full",),
        residual_scales={"full": 1.0},
        server_windows=(1,),
        server_budgets={
            "full": {
                "1": {
                    "anchor": 100.0,
                    "residual": 100.0,
                    "total": 100.0,
                }
            }
        },
        principal_windows=windows,
        principal_betas={
            "full": (
                betas
                or {str(window): 100.0 for window in windows}
            )
        },
    )


def _defense(manifest, principal_map, **custom):
    return RTCv3Defense(
        DefenseConfig(
            enabled=True,
            type="rtc_v3_candidate",
            custom_params={
                "calibration_manifest": manifest,
                "implementation_phase": 4,
                "principal_map": principal_map,
                "principal_first_sampling_verified": True,
                **custom,
            },
        )
    )


def _round(defense, round_value, global_value, clients):
    ids = [client_id for client_id, _ in clients]
    defense.set_context(
        round_value,
        ids,
        _params(global_value),
        server_optimizer="fedavg",
    )
    result = defense.aggregate(
        [(_params(global_value + delta), 1) for _, delta in clients]
    )
    return float(result[0][0])


def test_principal_long_window_limits_late_residual_spike():
    mapping = {"attacker": "p", "good-a": "a", "good-b": "b"}
    defense = _defense(
        _manifest(windows=(1, 3), betas={"1": 1.0, "3": 0.5}),
        mapping,
    )
    value = _round(
        defense,
        1,
        0.0,
        [("attacker", 0.4), ("good-a", 0.0), ("good-b", 0.0)],
    )
    value = _round(
        defense,
        2,
        value,
        [("attacker", 0.4), ("good-a", 0.0), ("good-b", 0.0)],
    )
    value = _round(
        defense,
        3,
        value,
        [("attacker", 1.0), ("good-a", 0.0), ("good-b", 0.0)],
    )

    assert defense.last_client_aggregation_weights["attacker"] < 1.0 / 3.0
    events = defense._principal_ledger.snapshot()["p"]
    assert len(events) == 3
    assert sum(event.exposures["full"] for event in events) <= 0.5 + 1e-8


def test_absence_does_not_advance_or_clear_principal_history():
    mapping = {"attacker": "p", "good-a": "a", "good-b": "b"}
    defense = _defense(
        _manifest(windows=(1, 2), betas={"1": 1.0, "2": 0.5}),
        mapping,
    )
    value = _round(
        defense,
        1,
        0.0,
        [("attacker", 0.4), ("good-a", 0.0), ("good-b", 0.0)],
    )
    value = _round(defense, 2, value, [("good-a", 0.0), ("good-b", 0.0)])
    value = _round(defense, 3, value, [("good-a", 0.0), ("good-b", 0.0)])
    value = _round(defense, 4, value, [("good-a", 0.0), ("good-b", 0.0)])

    assert defense._principal_ledger.participation_count("p") == 1
    _round(
        defense,
        5,
        value,
        [("attacker", 1.0), ("good-a", 0.0), ("good-b", 0.0)],
    )
    assert defense._principal_ledger.participation_count("p") == 2
    assert defense.last_client_aggregation_weights["attacker"] < 1.0 / 3.0


def test_new_client_id_inherits_old_principal_ledger():
    mapping = {
        "attacker-old": "p",
        "attacker-new": "p",
        "good-a": "a",
        "good-b": "b",
    }
    defense = _defense(
        _manifest(windows=(1, 2), betas={"1": 1.0, "2": 0.5}),
        mapping,
    )
    value = _round(
        defense,
        1,
        0.0,
        [("attacker-old", 0.4), ("good-a", 0.0), ("good-b", 0.0)],
    )
    _round(
        defense,
        2,
        value,
        [("attacker-new", 1.0), ("good-a", 0.0), ("good-b", 0.0)],
    )

    assert defense._principal_ledger.participation_count("p") == 2
    assert defense.last_client_aggregation_weights["attacker-new"] < 1.0 / 3.0


def test_same_principal_split_consumes_one_shared_budget():
    manifest = _manifest(windows=(1,), betas={"1": 0.5})
    unsplit = _defense(
        manifest,
        {"p0": "p", "q": "q", "r": "r"},
    )
    _round(unsplit, 1, 0.0, [("p0", 1.0), ("q", 0.0), ("r", 0.0)])

    split = _defense(
        manifest,
        {"p0": "p", "p1": "p", "q": "q", "r": "r"},
    )
    _round(
        split,
        1,
        0.0,
        [("p0", 1.0), ("p1", 1.0), ("q", 0.0), ("r", 0.0)],
    )

    combined = (
        split.last_client_aggregation_weights["p0"]
        + split.last_client_aggregation_weights["p1"]
    )
    assert combined == pytest.approx(
        unsplit.last_client_aggregation_weights["p0"], abs=1e-7
    )
    assert split._principal_ledger.participation_count("p") == 1
    assert split._principal_ledger.snapshot()["p"][0].exposures[
        "full"
    ] == pytest.approx(
        unsplit._principal_ledger.snapshot()["p"][0].exposures["full"],
        abs=1e-7,
    )


def test_mpaf_self_reported_mass_is_clipped_to_registered_caps():
    mapping = {"fake": "p", "good": "g"}
    defense = _defense(
        _manifest(
            windows=(1,),
            betas={"1": 100.0},
            mass_policy="capped_num_examples",
        ),
        mapping,
        client_mass_caps={"fake": 10.0, "good": 10.0},
        principal_mass_caps={"p": 10.0, "g": 10.0},
    )
    defense.set_context(
        1,
        ["fake", "good"],
        _params(),
        server_optimizer="fedavg",
    )
    defense.aggregate([(_params(1.0), 10**9), (_params(0.0), 10)])

    records = {record.client_id: record for record in defense._last_records}
    assert records["fake"].validated_mass == pytest.approx(10.0)
    assert records["fake"].nominal_mass == pytest.approx(0.5)


def test_mpaf_uses_fixed_base_global_only_and_respects_norm_ceiling():
    client = object.__new__(MPAFClient)
    client.client_id = 9
    client._attack_active = True
    client.attack_cfg = SimpleNamespace(
        mpaf_base_scale=0.0,
        mpaf_lambda=2.0,
        mpaf_max_update_norm=0.5,
    )
    initial = [
        np.asarray([2.0], dtype=np.float32),
        np.asarray([7], dtype=np.int64),
    ]
    client.on_before_fit(initial, {})

    later = [
        np.asarray([1.0], dtype=np.float32),
        np.asarray([11], dtype=np.int64),
    ]
    client.on_before_fit(later, {})
    metrics = {}
    attacked = client.on_after_fit(
        [np.asarray([999.0], dtype=np.float32), np.asarray([99], dtype=np.int64)],
        metrics,
    )

    assert get_attack_client_class("mpaf") is MPAFClient
    assert attacked[0] == pytest.approx([0.5])
    assert attacked[1].tolist() == [11]
    assert metrics["mpaf_raw_update_norm"] == pytest.approx(2.0)
    assert metrics["mpaf_applied_scale"] == pytest.approx(0.25)
    assert {cell["attack"] for cell in build_matrix("mpaf")} == {"mpaf"}


def test_principal_first_sampling_never_allocates_two_slots_to_one_principal():
    defense = SimpleNamespace(
        principal_id_for=lambda client_id: {
            "p0": "p",
            "p1": "p",
            "q0": "q",
            "r0": "r",
        }[client_id]
    )
    strategy = object.__new__(FedSecStrategy)
    strategy.defense = defense
    clients = {
        name: SimpleNamespace(cid=name, partition_id=name)
        for name in ("p0", "p1", "q0", "r0")
    }
    manager = SimpleNamespace(all=lambda: clients)

    selected = strategy._principal_first_sample(manager, 3, 3, 42)
    principals = [
        defense.principal_id_for(client.partition_id) for client in selected
    ]
    assert len(selected) == 3
    assert len(set(principals)) == 3
