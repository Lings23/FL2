from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.common import Code, EvaluateRes, FitRes, Status, ndarrays_to_parameters
from torch.utils.data import DataLoader, Dataset, TensorDataset

from attacks.attack_client import LabelFlipDataset
from client.fl_client import FedSecClient, _configure_cpu_worker_threads
from config.config_loader import ClientConfig, DefenseConfig, StrategyConfig
from experiments.trial_plan import (
    TrialPlanError,
    TrialPlanV1,
    attach_trial_plans,
    generate_trial_plan,
)
from defenses.rtc.calibration import build_manifest
from strategies.fed_strategy import FedSecStrategy
from models.model_factory import get_parameters


def _args(**overrides):
    values = {
        "num_clients": 12,
        "rounds": 4,
        "smoke": False,
        "pairing_mode": "strict",
        "sampling_protocol": "principal_uniform",
        "max_client_samples": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _spec(attack="label_flip_targeted", defense="fedavg"):
    return {
        "attack": attack,
        "period": "continuous_1_0",
        "on_rounds": 1,
        "off_rounds": 0,
        "malicious_fraction": 0.2,
        "seed": 42,
        "defense": defense,
        "defense_type": "none" if defense == "fedavg" else defense,
        "custom_params": {},
        "partition": "iid",
        "dirichlet_alpha": 0.5,
        "participation_rate": 0.5,
        "attack_start_round": 1,
        "attack_end_round": -1,
    }


def test_same_condition_reuses_plan_across_defenses(tmp_path):
    rows = attach_trial_plans(
        [_spec(defense=name) for name in ("fedavg", "foolsgold", "krum")],
        _args(),
        tmp_path,
    )
    assert len({row["trial_plan_hash"] for row in rows}) == 1
    assert len({row["trial_plan_path"] for row in rows}) == 1
    plan = TrialPlanV1.load(rows[0]["trial_plan_path"])
    assert plan.trial_plan_hash == rows[0]["trial_plan_hash"]


def test_attack_conditions_do_not_reuse_plans_and_repeat_exactly(tmp_path):
    first = attach_trial_plans([_spec("label_flip_targeted")], _args(), tmp_path / "a")[0]
    repeated = attach_trial_plans([_spec("label_flip_targeted")], _args(), tmp_path / "b")[0]
    reverse = attach_trial_plans([_spec("label_flip_all_reverse")], _args(), tmp_path / "c")[0]
    assert first["trial_plan_hash"] == repeated["trial_plan_hash"]
    assert first["trial_plan_hash"] != reverse["trial_plan_hash"]


def test_numeric_ids_and_multi_endpoint_principals_are_unambiguous():
    condition = {
        "attack": "label_flip_targeted",
        "period": "continuous_1_0",
        "malicious_fraction": 0.2,
        "seed": 7,
        "num_clients": 12,
        "rounds": 8,
        "clients_per_round": 5,
        "sampling_protocol": "principal_uniform",
    }
    principal_map = {str(index): str(index // 2) for index in range(12)}
    payload = generate_trial_plan(condition, principal_map)
    for round_plan in payload["rounds"]:
        assert len(round_plan["principal_ids"]) == len(set(round_plan["principal_ids"]))
        assert all(value in {str(index) for index in range(12)} for value in round_plan["partition_ids"])
    # Numeric logical order is 0,1,2,...,10,11; lexical 0,1,10,11,... is forbidden.
    identity = generate_trial_plan(
        {**condition, "clients_per_round": 12},
        {str(index): str(index) for index in range(12)},
    )
    assert identity["rounds"][0]["partition_ids"] == [str(index) for index in range(12)]


def test_tampered_plan_is_rejected(tmp_path):
    row = attach_trial_plans([_spec()], _args(), tmp_path)[0]
    path = row["trial_plan_path"]
    payload = json.loads(open(path, encoding="utf-8").read())
    payload["rounds"][0]["partition_ids"][0] = "999"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    with pytest.raises(TrialPlanError, match="hash mismatch"):
        TrialPlanV1.load(path)


class _Proxy:
    def __init__(self, partition_id):
        self.partition_id = str(partition_id)
        self.cid = str(partition_id)


class _Manager:
    def __init__(self, count):
        self.clients = {str(index): _Proxy(index) for index in range(count)}

    def all(self):
        return self.clients


def test_strict_strategy_consumes_ordered_plan_and_rejects_missing(tmp_path):
    row = attach_trial_plans([_spec()], _args(), tmp_path)[0]
    plan = TrialPlanV1.load(row["trial_plan_path"])
    strategy = FedSecStrategy(
        strategy_cfg=StrategyConfig(),
        defense_cfg=DefenseConfig(),
        initial_params=[np.zeros(2, dtype=np.float32)],
        num_clients=12,
        clients_per_round=6,
        pairing_mode="strict",
        sampling_protocol="principal_uniform",
        trial_plan=plan,
    )
    manager = _Manager(12)
    configured = strategy.configure_fit(
        1, ndarrays_to_parameters([np.zeros(2, dtype=np.float32)]), manager,
    )
    assert [client.partition_id for client, _ in configured] == plan.round(1)["partition_ids"]
    missing = plan.round(2)["partition_ids"][0]
    manager.clients.pop(missing)
    with pytest.raises(TrialPlanError, match="unavailable"):
        strategy.configure_fit(
            2, ndarrays_to_parameters([np.zeros(2, dtype=np.float32)]), manager,
        )
    with pytest.raises(TrialPlanError, match="whole run is invalid"):
        strategy.aggregate_fit(1, [], [RuntimeError("client failed")])


def test_strict_evaluation_uses_and_validates_plan_order(tmp_path):
    row = attach_trial_plans([_spec()], _args(), tmp_path)[0]
    plan = TrialPlanV1.load(row["trial_plan_path"])
    strategy = FedSecStrategy(
        strategy_cfg=StrategyConfig(),
        defense_cfg=DefenseConfig(),
        initial_params=[np.zeros(2, dtype=np.float32)],
        num_clients=12,
        clients_per_round=6,
        min_evaluate_clients=3,
        pairing_mode="strict",
        sampling_protocol="principal_uniform",
        trial_plan=plan,
    )
    manager = _Manager(12)
    parameters = ndarrays_to_parameters([np.zeros(2, dtype=np.float32)])
    configured = strategy.configure_evaluate(1, parameters, manager)
    planned = plan.round(1)["partition_ids"][:3]
    assert [client.partition_id for client, _ in configured] == planned

    results = [
        (
            client,
            EvaluateRes(
                status=Status(code=Code.OK, message=""),
                loss=float(client.partition_id),
                num_examples=10,
                metrics={
                    "val_accuracy": float(client.partition_id) / 10.0,
                    "client_id": int(client.partition_id),
                    "eval_seed": int(evaluate_ins.config["eval_seed"]),
                },
            ),
        )
        for client, evaluate_ins in reversed(configured)
    ]
    _, metrics = strategy.aggregate_evaluate(1, results, [])
    assert metrics["planned_evaluate_partition_ids"] == ",".join(planned)
    assert metrics["completed_evaluate_partition_ids"] == ",".join(planned)

    with pytest.raises(TrialPlanError, match="evaluation result set differs"):
        strategy.aggregate_evaluate(1, results[:-1], [])


@pytest.mark.parametrize("defense_name", ["fedavg", "foolsgold", "rtc_v3"])
def test_three_round_strict_cpu_aggregation_smoke(tmp_path, defense_name):
    spec = _spec(defense=defense_name)
    spec["participation_rate"] = 0.75
    args = _args(num_clients=4, rounds=3)
    if defense_name == "rtc_v3":
        params = [
            np.zeros((2, 3), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.asarray([7], dtype=np.int64),
        ]
        spec["defense_type"] = "rtc_v3_candidate"
        spec["custom_params"] = {
            "calibration_manifest": build_manifest(params=params),
            "implementation_phase": 6,
            "principal_map": {str(index): str(index) for index in range(4)},
            "principal_first_sampling_verified": False,
        }
    else:
        params = [np.zeros(6, dtype=np.float32)]
    planned = attach_trial_plans([spec], args, tmp_path)[0]
    plan = TrialPlanV1.load(planned["trial_plan_path"])
    strategy = FedSecStrategy(
        strategy_cfg=StrategyConfig(),
        defense_cfg=DefenseConfig(
            enabled=defense_name != "fedavg",
            type=str(planned["defense_type"]),
            custom_params=dict(planned["custom_params"]),
        ),
        initial_params=[value.copy() for value in params],
        num_clients=4,
        clients_per_round=3,
        pairing_mode="strict",
        sampling_protocol="principal_uniform",
        trial_plan=plan,
    )
    manager = _Manager(4)
    parameters = ndarrays_to_parameters(params)
    observed = []
    for server_round in range(1, 4):
        configured = strategy.configure_fit(server_round, parameters, manager)
        results = []
        for client, fit_ins in configured:
            client_id = client.partition_id
            update = [
                value.copy() if not np.issubdtype(value.dtype, np.floating)
                else value + (int(client_id) + 1) * 0.01
                for value in params
            ]
            results.append((client, FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(update),
                num_examples=10,
                metrics={
                    "client_id": int(client_id),
                    "is_malicious": False,
                    "attack_active": False,
                    "fit_seed": int(fit_ins.config["fit_seed"]),
                    "attack_seed": int(fit_ins.config["attack_seed"]),
                },
            )))
        parameters, metrics = strategy.aggregate_fit(server_round, results, [])
        observed.append(metrics["completed_partition_ids_json"])
    assert observed == [
        json.dumps(plan.round(server_round)["partition_ids"])
        for server_round in range(1, 4)
    ]


class _RandomAugmentationDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.tensor(index), torch.rand(())


def test_same_client_round_seed_replays_batch_and_augmentation_and_poisoning():
    source = DataLoader(_RandomAugmentationDataset(), batch_size=3, shuffle=True)
    torch.manual_seed(1234)
    first = list(FedSecClient._rebuild_train_loader(source, 1234))
    torch.manual_seed(1234)
    second = list(FedSecClient._rebuild_train_loader(source, 1234))
    assert all(torch.equal(a, b) for left, right in zip(first, second) for a, b in zip(left, right))

    base = TensorDataset(torch.arange(20).float().unsqueeze(1), torch.tensor([5] * 20))
    poison_a = LabelFlipDataset(
        base, attack_type="label_flip_targeted", num_classes=10,
        poison_fraction=0.5, source=5, target=3, seed=88,
    )
    poison_b = LabelFlipDataset(
        base, attack_type="label_flip_targeted", num_classes=10,
        poison_fraction=0.5, source=5, target=3, seed=88,
    )
    assert poison_a.poison_indices == poison_b.poison_indices


def test_three_round_full_client_cpu_smoke_is_paired_across_defenses(tmp_path):
    torch.manual_seed(17)
    base_model = torch.nn.Linear(4, 2)
    initial = get_parameters(base_model)
    rtc_custom = {
        "calibration_manifest": build_manifest(params=initial),
        "implementation_phase": 6,
        "principal_map": {str(index): str(index) for index in range(4)},
        "principal_first_sampling_verified": False,
    }
    specs = []
    for defense_name in ("fedavg", "foolsgold", "rtc_v3"):
        spec = _spec(defense=defense_name)
        spec["participation_rate"] = 0.75
        if defense_name == "rtc_v3":
            spec["defense_type"] = "rtc_v3_candidate"
            spec["custom_params"] = rtc_custom
        specs.append(spec)
    planned = attach_trial_plans(specs, _args(num_clients=4, rounds=3), tmp_path)

    schedules = {}
    for spec in planned:
        plan = TrialPlanV1.load(spec["trial_plan_path"])
        strategy = FedSecStrategy(
            strategy_cfg=StrategyConfig(),
            defense_cfg=DefenseConfig(
                enabled=spec["defense"] != "fedavg",
                type=spec["defense_type"],
                custom_params=dict(spec["custom_params"]),
            ),
            initial_params=[value.copy() for value in initial],
            num_clients=4,
            clients_per_round=3,
            pairing_mode="strict",
            sampling_protocol="principal_uniform",
            trial_plan=plan,
        )
        manager = _Manager(4)
        clients = {}
        for client_id in range(4):
            generator = torch.Generator().manual_seed(100 + client_id)
            x = torch.randn(12, 4, generator=generator)
            y = torch.randint(0, 2, (12,), generator=generator)
            dataset = TensorDataset(x, y)
            clients[str(client_id)] = FedSecClient(
                client_id=client_id,
                model=torch.nn.Linear(4, 2),
                train_loader=DataLoader(dataset, batch_size=4, shuffle=True),
                val_loader=DataLoader(dataset, batch_size=4, shuffle=False),
                client_cfg=ClientConfig(
                    local_epochs=1, batch_size=4, learning_rate=0.01,
                    momentum=0.0, weight_decay=0.0, lr_scheduler="none",
                ),
                device=torch.device("cpu"),
                experiment_seed=42,
                num_classes=2,
            )
        parameters = ndarrays_to_parameters(initial)
        defense_schedule = []
        for server_round in range(1, 4):
            configured = strategy.configure_fit(server_round, parameters, manager)
            results = [
                (proxy, clients[proxy.partition_id].fit(fit_ins))
                for proxy, fit_ins in configured
            ]
            parameters, metrics = strategy.aggregate_fit(server_round, results, [])
            defense_schedule.append(metrics["completed_partition_ids_json"])
        schedules[spec["defense"]] = defense_schedule
    assert len({tuple(value) for value in schedules.values()}) == 1


def test_cpu_worker_thread_ceiling_is_opt_in(monkeypatch):
    calls = []
    monkeypatch.delenv("FEDSEC_CPU_THREADS", raising=False)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: calls.append(("intra", value)))
    monkeypatch.setattr(
        torch, "set_num_interop_threads", lambda value: calls.append(("interop", value))
    )
    _configure_cpu_worker_threads(torch.device("cpu"))
    _configure_cpu_worker_threads(torch.device("cuda"))
    assert calls == []

    monkeypatch.setenv("FEDSEC_CPU_THREADS", "1")
    _configure_cpu_worker_threads(torch.device("cpu"))
    assert calls == [("intra", 1), ("interop", 1)]


def test_cpu_worker_thread_ceiling_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("FEDSEC_CPU_THREADS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        _configure_cpu_worker_threads(torch.device("cpu"))
