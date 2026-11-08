"""Paired clean/DBA pilot for the layer-conditioned relation-graph defense.

Four independently trained branches share one initial MNIST-CNN model, one IID
client split and one client-seed rule:

    clean_fedavg, clean_relation_graph, dba_fedavg, dba_relation_graph

DBA is trained on clients 0-3 with one local MNIST trigger fragment each, from
round 1 through the complete budget, with the frozen poison rate and no upload
amplification.  Malicious identities are used only for attack assignment and
post-hoc evaluation; the defense receives only stable identities, the server
parameter layout, classifier roles and upload arrays.

The evaluator is the existing 2000-example training holdout with the frozen
mixed-class order (seed 2042) and an 1800-example non-target ASR denominator.
The final test set is never opened.

Run:
    python -m experiments.relation_graph_mnist_paired --config \
        config/relation_graph_mnist_paired_120.json --output analysis/<run>

Wiring check (short, separate output directory):
    python -m experiments.relation_graph_mnist_paired --config \
        config/relation_graph_mnist_paired_120.json --output analysis/<smoke> \
        --smoke-rounds 2

Resume from the last complete round checkpoint:
    ... --output analysis/<run> --resume
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays

from attacks.attack_client import validate_attack_config
from attacks.reference_backdoors import PGD_REFERENCE_CONFIG_KEY
from attacks.spec import attack_contract_payload, attack_source_hash
from client.fl_client import make_client_fn
from config.config_loader import AttackConfig, ClientConfig, DefenseConfig
from defenses.defense_base import get_defense
from defenses.relation_graph.settings import Settings
from experiments.direction_a_validation import (
    ROOT, append, check_stop, clean_metrics, dump, load_checkpoint, load_lines,
    save_checkpoint, split_indices, status, tensor_batches,
)
from experiments.guardfl_mnist import (
    evaluation_order, model_factory as stateless_model_factory,
)
from experiments.reference_attack_qualification import (
    attack_cfg, load_training_data, profile_model,
)
from models.model_factory import (
    get_parameter_names, get_parameter_roles, get_parameters,
    get_trainable_parameter_indices,
)
from server.fl_server import evaluate_targeted_asr

GROUPS = (
    dict(name="clean_fedavg", attack="none", defense="fedavg"),
    dict(name="clean_relation_graph", attack="none", defense="relation_graph"),
    dict(name="dba_fedavg", attack="dba", defense="fedavg"),
    dict(name="dba_relation_graph", attack="dba", defense="relation_graph"),
)
NAMES = tuple(group["name"] for group in GROUPS)
SELF = "experiments/relation_graph_mnist_paired.py"
CRITICAL_SOURCES = (
    SELF,
    "experiments/guardfl_mnist.py",
    "experiments/direction_a_validation.py",
    "experiments/reference_attack_qualification.py",
    "client/fl_client.py",
    "attacks/attack_client.py",
    "attacks/reference_backdoors.py",
    "attacks/spec.py",
    "attacks/coordinator.py",
    "config/config_loader.py",
    "models/model_factory.py",
    "models/mnist_cnn.py",
    "data/dataset.py",
    "server/fl_server.py",
    "strategies/fed_strategy.py",
    "defenses/defense_base.py",
    "utils/numerical_failure.py",
    "defenses/relation_graph/__init__.py",
    "defenses/relation_graph/defense.py",
    "defenses/relation_graph/exclusion.py",
    "defenses/relation_graph/graph.py",
    "defenses/relation_graph/settings.py",
)
EVALUATION_ATTACK = "dba"
MULTI_ATTACK_PURPOSE = "relation_graph_mnist_multi_attack_paired"
ATTACK_TYPES = ("badnet", "neurotoxin", "pgd")
MULTI_ATTACK_GROUPS = (
    dict(name="clean_fedavg", attack="none", defense="fedavg"),
    dict(name="clean_relation_graph", attack="none", defense="relation_graph"),
    *(dict(name=f"{attack}_{defense}", attack=attack, defense=defense)
      for attack in ATTACK_TYPES
      for defense in ("fedavg", "relation_graph")),
)


def group_names(cfg: dict) -> tuple[str, ...]:
    return tuple(group["name"] for group in cfg["groups"])


def group_name(cfg: dict, attack: str, defense: str) -> str:
    return next(group["name"] for group in cfg["groups"]
                if group["attack"] == attack and group["defense"] == defense)


def pilot_attack_cfg(cfg: dict, condition: str) -> AttackConfig:
    """Build an attack config, applying multi-attack profile overrides."""
    if cfg.get("purpose") != MULTI_ATTACK_PURPOSE:
        return attack_cfg(cfg, condition)
    if condition == "none":
        return AttackConfig(**cfg["attack"], type="none", enabled=False)
    values = dict(cfg["attack"])
    values.update(cfg.get("attack_profiles", {}).get(condition, {}))
    return AttackConfig(**values, type=condition, enabled=True)


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def source_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in CRITICAL_SOURCES}


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        array = value.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def paired_model_factory(cfg: dict) -> torch.nn.Module:
    """Build the model with the protocol's normalization contract.

    ``"batch"`` keeps the standard ``BatchNorm2d(track_running_stats=True)``
    used by ``experiments/reference_attack_qualification.py``.  Its
    ``running_mean``/``running_var``/``num_batches_tracked`` buffers are
    aggregated by ``RelationGraphDefense`` and make evaluation use clean
    training moments instead of the moments of a 100%-triggered batch.
    ``"batch_no_running_stats"`` retains the legacy stateless BatchNorm.
    """
    normalization = cfg.get("normalization")
    if normalization == "batch":
        model = profile_model(cfg)
        if not any(True for _ in model.named_buffers()):
            raise ValueError("Tracked MNIST/CNN must have BatchNorm running buffers")
        return model
    if normalization == "batch_no_running_stats":
        return stateless_model_factory(cfg)
    raise ValueError(f"Unsupported normalization mode: {normalization}")


def validate(cfg: dict) -> None:
    if cfg.get("purpose") == MULTI_ATTACK_PURPOSE:
        _validate_multi_attack(cfg)
        return
    _validate_dba_pilot(cfg)


def _validate_dba_pilot(cfg: dict) -> None:
    if cfg.get("schema_version") != 1 or cfg.get("purpose") != "relation_graph_mnist_paired_clean_dba_pilot":
        raise ValueError("Explicit relation_graph paired pilot protocol required")
    if (cfg.get("dataset"), cfg.get("architecture")) != ("mnist", "mnist_cnn") or \
            cfg.get("normalization") not in {"batch", "batch_no_running_stats"}:
        raise ValueError("MNIST/CNN with tracked ('batch') or stateless "
                         "('batch_no_running_stats') BatchNorm required")
    if cfg.get("clients") != 20 or cfg.get("malicious_clients") != [0, 1, 2, 3]:
        raise ValueError("Frozen 20-client / malicious-0-3 protocol required")
    expected_groups = [dict(group) for group in GROUPS]
    if cfg.get("groups") != expected_groups:
        raise ValueError("The four paired groups are fixed")
    if cfg.get("warmup_rounds") != 0 or int(cfg.get("total_rounds", 0)) < 1:
        raise ValueError("No warmup and a positive round budget are required")
    snapshots = cfg.get("snapshots")
    if (not isinstance(snapshots, list) or snapshots != sorted(set(snapshots))
            or any(not isinstance(value, int) or not 1 <= value <= cfg["total_rounds"] for value in snapshots)):
        raise ValueError("Invalid snapshots")
    if cfg.get("max_client_samples") != 500 or cfg.get("drop_last") is not True:
        raise ValueError("Frozen 500-example drop-last client loaders required")
    if (cfg.get("reference_probe_size"), cfg.get("reference_gate_size"),
            cfg.get("evaluator_validation_size")) != (300, 200, 2000):
        raise ValueError("Frozen reference/validation partition required")
    if cfg.get("round_lr_decay") != 0.99 or cfg.get("evaluation_order_seed") != 2042:
        raise ValueError("Frozen learning-rate decay and evaluator order required")
    if cfg.get("client") != {
        "local_epochs": 2, "batch_size": 64, "optimizer": "sgd", "learning_rate": 0.1,
        "momentum": 0.0, "weight_decay": 0.0001, "lr_scheduler": "none",
    }:
        raise ValueError("Client training recipe differs from the frozen profile")
    attack = cfg.get("attack", {})
    if (attack.get("malicious_fraction") != 0.2 or attack.get("backdoor_target_label") != 7
            or attack.get("poison_fraction") != 0.5
            or attack.get("reference_backdoor_dataset") != "mnist"
            or attack.get("dba_pattern_mode") != "paper_mnist_1x4_2x2"
            or attack.get("dba_trigger_value_mode") != "mnist_normalized_white"
            or attack.get("dba_trigger_num") != 4 or attack.get("dba_gap") != 2):
        raise ValueError("DBA mechanism differs from the frozen profile")
    if (attack.get("dba_scale_update") is not False or attack.get("dba_boost_factor") != 1.0
            or attack.get("aggregation_aware_scaling") is not False):
        raise ValueError("This pilot forbids DBA upload amplification")
    if attack.get("attack_start_round") != 1 or attack.get("attack_end_round") < cfg["total_rounds"]:
        raise ValueError("Continuous DBA must cover the complete budget")
    validate_attack_config(attack_cfg(cfg, EVALUATION_ATTACK), 10)
    settings = cfg.get("relation_graph", {}).get("settings")
    if not isinstance(settings, dict) or settings != asdict(Settings()):
        raise ValueError("relation_graph must use the frozen current default settings")
    # The defense contract also needs a classifier head; this is checked in Engine.
    if cfg.get("device") not in {"cuda", "cpu"}:
        raise ValueError("Unsupported device")
    if cfg.get("defense_variant") not in (None, "group_zero_weight"):
        raise ValueError("Unsupported defense_variant")


def _validate_multi_attack(cfg: dict) -> None:
    if cfg.get("schema_version") != 1:
        raise ValueError("Multi-attack protocol requires schema_version=1")
    if (cfg.get("dataset"), cfg.get("architecture")) != ("mnist", "mnist_cnn") or \
            cfg.get("normalization") not in {"batch", "batch_no_running_stats"}:
        raise ValueError("MNIST/CNN with tracked ('batch') or stateless "
                         "('batch_no_running_stats') BatchNorm required")
    if cfg.get("clients") != 20 or cfg.get("malicious_clients") != [0, 1, 2, 3]:
        raise ValueError("Frozen 20-client / malicious-0-3 protocol required")
    if cfg.get("groups") != [dict(group) for group in MULTI_ATTACK_GROUPS]:
        raise ValueError("The eight paired groups are fixed")
    if cfg.get("evaluation_attack") not in ATTACK_TYPES:
        raise ValueError("evaluation_attack must be badnet, neurotoxin or pgd")
    if cfg.get("warmup_rounds") != 0 or int(cfg.get("total_rounds", 0)) < 1:
        raise ValueError("No warmup and a positive round budget are required")
    snapshots = cfg.get("snapshots")
    if (not isinstance(snapshots, list) or snapshots != sorted(set(snapshots))
            or any(not isinstance(value, int) or not 1 <= value <= cfg["total_rounds"]
                   for value in snapshots)):
        raise ValueError("Invalid snapshots")
    if cfg.get("max_client_samples") != 500 or cfg.get("drop_last") is not True:
        raise ValueError("Frozen 500-example drop-last client loaders required")
    if (cfg.get("reference_probe_size"), cfg.get("reference_gate_size"),
            cfg.get("evaluator_validation_size")) != (300, 200, 2000):
        raise ValueError("Frozen reference/validation partition required")
    if cfg.get("round_lr_decay") != 0.99 or cfg.get("evaluation_order_seed") != 2042:
        raise ValueError("Frozen learning-rate decay and evaluator order required")
    if cfg.get("client") != {
        "local_epochs": 2, "batch_size": 64, "optimizer": "sgd", "learning_rate": 0.1,
        "momentum": 0.0, "weight_decay": 0.0001, "lr_scheduler": "none",
    }:
        raise ValueError("Client training recipe differs from the frozen profile")
    attack = cfg.get("attack", {})
    if (attack.get("malicious_fraction") != 0.2
            or attack.get("backdoor_target_label") != 7
            or attack.get("poison_fraction") != 0.5
            or attack.get("reference_backdoor_dataset") != "mnist"
            or attack.get("attack_start_round") != 1
            or attack.get("attack_end_round") < cfg["total_rounds"]):
        raise ValueError("Multi-attack mechanism differs from the frozen profile")
    if attack.get("aggregation_aware_scaling") is not False:
        raise ValueError("This pilot forbids upload amplification")
    profiles = cfg.get("attack_profiles", {})
    if not isinstance(profiles, dict) or set(profiles) - set(ATTACK_TYPES) \
            or any(not isinstance(values, dict) for values in profiles.values()):
        raise ValueError("attack_profiles must map known attacks to mappings")
    for condition in ATTACK_TYPES:
        validate_attack_config(pilot_attack_cfg(cfg, condition), 10)
    neurotoxin = {**attack, **profiles.get("neurotoxin", {})}
    if (neurotoxin.get("neurotoxin_mask_source") != "clean_gradient"
            or neurotoxin.get("neurotoxin_aggregate_all_layer") is not False):
        raise ValueError("Neurotoxin must use the local clean-gradient GradMask")
    pgd = {**attack, **profiles.get("pgd", {})}
    if (pgd.get("pgd_state_scope") != "parameters"
            or pgd.get("pgd_radius_source") != "benign_update_norm"):
        raise ValueError("PGD requires pgd_state_scope=parameters and "
                         "pgd_radius_source=benign_update_norm")
    settings = cfg.get("relation_graph", {}).get("settings")
    if not isinstance(settings, dict) or settings != asdict(Settings()):
        raise ValueError("relation_graph must use the frozen current default settings")
    if cfg.get("device") not in {"cuda", "cpu"}:
        raise ValueError("Unsupported device")
    if cfg.get("defense_variant") not in (None, "group_zero_weight"):
        raise ValueError("Unsupported defense_variant")


def smoke_config(cfg: dict, rounds: int) -> dict:
    if not 1 <= rounds < cfg["total_rounds"]:
        raise ValueError("Smoke rounds must be positive and shorter than the formal budget")
    result = copy.deepcopy(cfg)
    result["total_rounds"] = rounds
    result["snapshots"] = sorted(set(value for value in cfg["snapshots"] if value < rounds) | {rounds})
    result["smoke_wiring_check"] = dict(source_total_rounds=cfg["total_rounds"], smoke_rounds=rounds)
    return result


def client_train_summary(metrics: dict) -> dict:
    keys = ("train_loss", "train_accuracy", "local_epochs", "is_malicious", "attack_active",
            "dba_fragment_index", "dba_poisoned_examples", "dba_total_examples",
            "dba_update_scaled", "backdoor_poisoned_examples", "backdoor_eligible_examples",
            "pgd_radius", "pgd_projection_count", "pgd_max_pre_projection_norm",
            "pgd_final_update_norm", "pgd_clipped_epochs", "pgd_reference_norm",
            "pgd_reference_available", "pgd_norm_scope", "pgd_radius_source")
    return {key: metrics[key] for key in keys if key in metrics}


class Engine:
    def __init__(self, cfg, indices, train, view, device):
        self.cfg, self.device = cfg, device
        self.model = paired_model_factory(cfg).to(device)
        self.names = get_parameter_names(self.model)
        self.roles = get_parameter_roles(self.model)
        self.param_indices = get_trainable_parameter_indices(self.model)
        if not self.param_indices:
            raise ValueError("No trainable parameters")
        if not self.roles:
            raise ValueError("No classifier roles were identified")
        self.validation = tensor_batches(
            view, evaluation_order(indices["validation"], cfg["evaluation_order_seed"]),
            cfg["evaluation_batch_size"])
        loader_map = {
            index: (DataLoader(Subset(train, indices[f"client_{index}"].tolist()),
                               batch_size=cfg["client"]["batch_size"], drop_last=cfg["drop_last"]),
                    DataLoader(Subset(view, indices["gate"].tolist()),
                               batch_size=cfg["evaluation_batch_size"]))
            for index in range(cfg["clients"])
        }
        if any(len(loaders[0]) == 0 for loaders in loader_map.values()):
            raise ValueError("An empty client training loader")
        self.malicious = set(cfg["malicious_clients"])
        self.evaluation_attack = cfg.get("evaluation_attack", EVALUATION_ATTACK)
        self.pgd_reference = {group["name"]: None for group in cfg["groups"]}
        self.clients, self.defenses = {}, {}
        for group in cfg["groups"]:
            name, attack = group["name"], group["attack"]
            factory = make_client_fn(
                lambda: self.model, loader_map, ClientConfig(**cfg["client"]),
                attack_cfg=pilot_attack_cfg(cfg, attack),
                malicious_ids=self.malicious if attack != "none" else set(),
                device=device, experiment_seed=cfg["seed"])
            self.clients[name] = [factory(str(index)) for index in range(cfg["clients"])]
            if group["defense"] == "relation_graph":
                defense_config = DefenseConfig(
                    enabled=True, type="relation_graph",
                    custom_params={"settings": cfg["relation_graph"]["settings"]})
            else:
                defense_config = DefenseConfig(enabled=True, type="fedavg")
            if (group["defense"] == "relation_graph"
                    and cfg.get("defense_variant") == "group_zero_weight"):
                from defenses.relation_graph.exclusion import (
                    GroupExclusionRelationGraphDefense,
                )
                self.defenses[name] = GroupExclusionRelationGraphDefense(
                    defense_config, num_clients=cfg["clients"])
            else:
                self.defenses[name] = get_defense(
                    defense_config, num_clients=cfg["clients"])

    def fit_round(self, group, state, number, output):
        cfg = self.cfg
        name, attack_kind = group["name"], group["attack"]
        self.model.load_state_dict(state)
        arrays = [value.copy() for value in get_parameters(self.model)]
        lr = cfg["client"]["learning_rate"] * cfg["round_lr_decay"] ** number
        fit_config = {"server_round": number, "deterministic_client_training": True}
        if attack_kind == "pgd" and self.pgd_reference[name] is not None:
            fit_config[PGD_REFERENCE_CONFIG_KEY] = self.pgd_reference[name]
        ins = FitIns(ndarrays_to_parameters(arrays), fit_config)
        updates, metrics, norms = [], [], []
        started = time.monotonic()
        for identity, client in enumerate(self.clients[name]):
            check_stop(output)
            status(output, "client_training", group=name, round=number, client=identity,
                   learning_rate=lr)
            client.client_cfg.learning_rate = lr
            result = client.fit(ins)
            values = parameters_to_ndarrays(result.parameters)
            if result.num_examples <= 0 or any(not np.isfinite(value).all() for value in values):
                raise FloatingPointError(f"{name} round {number} client {identity}: invalid upload")
            updates.append((values, result.num_examples))
            metrics.append(result.metrics)
            norms.append(float(np.sqrt(sum(
                np.sum((values[index].astype(np.float64) - arrays[index].astype(np.float64)) ** 2)
                for index in self.param_indices))))
        if attack_kind == "pgd":
            # Public benign-scale reference for the next round: the median
            # client parameter-update norm of this completed round.
            self.pgd_reference[name] = float(np.median(norms))
        client_seconds = time.monotonic() - started

        identities = [str(index) for index in range(cfg["clients"])]
        defense = self.defenses[name]
        defense.set_context(
            number, identities, arrays, principal_ids=identities, server_optimizer="fedavg",
            parameter_roles=self.roles, parameter_names=self.names,
            trainable_parameter_indices=self.param_indices)
        aggregation_started = time.monotonic()
        aggregated = defense.aggregate(updates)
        # FedAvg's dtype-safe helper rounds integer buffers and can return a
        # NumPy scalar for a 0-D counter such as ``num_batches_tracked``.
        # Materialize every entry as an ndarray before rebuilding the state.
        aggregated = [np.asarray(value) for value in aggregated]
        aggregation_seconds = time.monotonic() - aggregation_started
        candidate = {key: torch.from_numpy(value.copy())
                     for key, value in zip(self.model.state_dict(), aggregated)}
        self.model.load_state_dict(candidate)

        evaluation_started = time.monotonic()
        clean = clean_metrics(self.model, self.validation, self.device)
        evaluation_attack = (attack_kind if attack_kind != "none"
                             else self.evaluation_attack)
        asr = evaluate_targeted_asr(
            self.model, self.validation, self.device,
            pilot_attack_cfg(cfg, evaluation_attack))
        if not asr or not asr["asr_valid"] or asr["asr_total"] != 1800:
            raise ValueError("Invalid DBA ASR evaluation")
        evaluation_seconds = time.monotonic() - evaluation_started

        diagnostics = defense.diagnostics() if hasattr(defense, "diagnostics") else {}
        graph_metrics = diagnostics.get("metrics", {}) if diagnostics else {}
        counts = np.asarray([count for _, count in updates], dtype=np.float64)
        nominal = counts / float(counts.sum())
        actual = nominal.copy()
        if group["defense"] == "relation_graph":
            records = {str(row["client_id"]): row["aggregation_weight"]
                       for row in diagnostics["clients"]}
            actual = np.asarray([records[str(index)] for index in range(cfg["clients"])],
                                dtype=np.float64)
        anchor = float(graph_metrics.get("relation_graph_anchor_weight", 0.0))
        if not math.isclose(float(actual.sum()) + anchor, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{name} round {number}: aggregation weights do not sum to one")
        malicious = self.malicious if attack_kind != "none" else set()

        def weight_summary(ids):
            if not ids:
                return None
            ids = sorted(ids)
            nominal_sum = float(sum(nominal[index] for index in ids))
            actual_sum = float(sum(actual[index] for index in ids))
            drops = [float(nominal[index] - actual[index]) for index in ids]
            down = [index for index, drop in zip(ids, drops) if drop > 1e-12]
            ratios = [float(actual[index] / nominal[index]) for index in ids if nominal[index] > 0]
            return dict(count=len(ids), nominal_sum=nominal_sum, aggregation_sum=actual_sum,
                        downweighted_count=len(down), downweighted_ids=down,
                        mean_weight_ratio=float(np.mean(ratios)) if ratios else None,
                        min_weight_ratio=float(np.min(ratios)) if ratios else None,
                        total_drop=float(sum(drops)), max_drop=float(max(drops)))

        clients = []
        for index in range(cfg["clients"]):
            ratio = float(actual[index] / nominal[index]) if nominal[index] > 0 else None
            clients.append(dict(
                client_id=index, nominal_weight=float(nominal[index]),
                aggregation_weight=float(actual[index]), weight_ratio=ratio,
                weight_drop=float(nominal[index] - actual[index]),
                update_norm=norms[index],
                is_malicious_evaluation_only=index in malicious,
                train=client_train_summary(metrics[index]),
            ))

        aggregate_norm = float(np.sqrt(sum(
            np.sum((aggregated[index].astype(np.float64) - arrays[index].astype(np.float64)) ** 2)
            for index in self.param_indices)))
        graph_row = None
        if group["defense"] == "relation_graph":
            graph_row = dict(
                round=number,
                state=graph_metrics.get("relation_graph_state"),
                calibrated_fraction=graph_metrics.get("relation_graph_calibrated_fraction"),
                observed_edges=graph_metrics.get("relation_graph_observed_edges"),
                confirmed_groups=graph_metrics.get("relation_graph_confirmed_groups"),
                downweighted_clients=graph_metrics.get("relation_graph_downweighted_clients"),
                anchor_weight=anchor,
                groups=diagnostics.get("groups", []),
                edges=diagnostics.get("edges", []),
            )
        row = dict(
            round=number, group=name, attack=attack_kind, defense=group["defense"],
            learning_rate=lr, clean=clean, asr=asr,
            graph=None if graph_row is None else {
                key: graph_row[key] for key in
                ("state", "calibrated_fraction", "observed_edges",
                 "confirmed_groups", "downweighted_clients", "anchor_weight")},
            clients=clients,
            benign_weight_summary=weight_summary(set(range(cfg["clients"])) - malicious),
            malicious_weight_summary=weight_summary(malicious),
            broadcast_model_weight=anchor,
            aggregate_update_norm=aggregate_norm,
            state_sha256=state_hash(candidate),
            client_seconds=client_seconds,
            aggregation_seconds=aggregation_seconds,
            evaluation_seconds=evaluation_seconds,
            round_seconds=time.monotonic() - started,
        )
        return candidate, row, graph_row


def qualification(cfg, baseline, attacked, label):
    q = cfg["qualification"]
    base = {row["round"]: row for row in baseline}
    checks = []
    for row in attacked[-q["sustained_rounds"]:]:
        reference = base.get(row["round"])
        if reference is None:
            continue
        checks.append(dict(
            round=row["round"],
            clean_model_ready=reference["clean"]["accuracy"] >= q["minimum_clean_accuracy"],
            task_preserved=(row["clean"]["accuracy"] >=
                            reference["clean"]["accuracy"] - q["maximum_accuracy_drop"]),
            attack_strong=row["asr"]["asr"] >= q["minimum_asr"],
            exceeds_clean_trigger_response=(row["asr"]["asr"] - reference["asr"]["asr"]
                                            >= q["minimum_asr_excess"]),
        ))
    passed = (len(checks) == q["sustained_rounds"] and
              all(all(value for key, value in check.items() if key != "round") for check in checks))
    return dict(
        group=label, qualified=passed, checks=checks, criteria=q,
        interpretation=("eligible_for_defense_evaluation" if passed
                        else "not_qualified_no_defense_success_claim"))


def _weight_aggregates(records, malicious):
    def collect(want_malicious):
        ratios, drops = [], []
        for row in records:
            for client in row["clients"]:
                if client["is_malicious_evaluation_only"] == want_malicious:
                    ratios.append(client["weight_ratio"])
                    drops.append(client["weight_drop"])
        return dict(
            client_rounds=len(ratios),
            downweighted_client_rounds=sum(drop > 1e-12 for drop in drops),
            mean_weight_ratio=float(np.mean(ratios)) if ratios else None,
            min_weight_ratio=float(np.min(ratios)) if ratios else None,
            total_drop=float(np.sum(drops)) if drops else 0.0,
            max_drop=float(np.max(drops)) if drops else 0.0,
        )
    return dict(benign=collect(False),
                malicious=collect(True) if malicious else None)


def _longest_true_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def group_stats(records, malicious):
    last = records[-1]
    recent = records[-10:]
    graph_rows = [row["graph"] for row in records if row["graph"] is not None]
    benign_down = [sum(1 for client in row["clients"]
                       if not client["is_malicious_evaluation_only"]
                       and client["weight_drop"] > 1e-12) for row in records]
    malicious_down = [sum(1 for client in row["clients"]
                          if client["is_malicious_evaluation_only"]
                          and client["weight_drop"] > 1e-12) for row in records]
    zero_weight_clients = sorted({
        client["client_id"] for row in records for client in row["clients"]
        if client["aggregation_weight"] <= 0.0})
    zero_update_clients = sorted({
        client["client_id"] for row in records for client in row["clients"]
        if client["update_norm"] <= 1e-12})
    consecutive_zero_weight = {}
    consecutive_zero_update = {}
    for index in range(len(records[0]["clients"])):
        consecutive_zero_weight[index] = _longest_true_run(
            [row["clients"][index]["aggregation_weight"] <= 0.0 for row in records])
        consecutive_zero_update[index] = _longest_true_run(
            [row["clients"][index]["update_norm"] <= 1e-12 for row in records])
    result = dict(
        completed_rounds=len(records),
        final=dict(round=last["round"], accuracy=last["clean"]["accuracy"],
                   loss=last["clean"]["loss"], asr=last["asr"]["asr"],
                   aggregate_update_norm=last["aggregate_update_norm"]),
        last10=dict(accuracy=float(np.mean([row["clean"]["accuracy"] for row in recent])),
                    loss=float(np.mean([row["clean"]["loss"] for row in recent])),
                    asr=float(np.mean([row["asr"]["asr"] for row in recent])),
                    aggregate_update_norm=float(np.mean([row["aggregate_update_norm"] for row in recent]))),
        peak=dict(asr=max(record["asr"]["asr"] for record in records),
                  asr_round=max(records, key=lambda row: row["asr"]["asr"])["round"],
                  accuracy=max(record["clean"]["accuracy"] for record in records)),
        benign_downweighted_client_rounds=int(sum(benign_down)),
        benign_downweighted_rounds=int(sum(value > 0 for value in benign_down)),
        malicious_downweighted_client_rounds=int(sum(malicious_down)),
        malicious_downweighted_rounds=int(sum(value > 0 for value in malicious_down)),
        weight_aggregates=_weight_aggregates(records, bool(malicious)),
        zero_weight_clients=zero_weight_clients,
        zero_update_clients=zero_update_clients,
        max_consecutive_zero_weight=max(consecutive_zero_weight.values(), default=0),
        max_consecutive_zero_update=max(consecutive_zero_update.values(), default=0),
        median_client_seconds=float(np.median([row["client_seconds"] for row in records])),
        median_aggregation_seconds=float(np.median([row["aggregation_seconds"] for row in records])),
        median_evaluation_seconds=float(np.median([row["evaluation_seconds"] for row in records])),
        median_round_seconds=float(np.median([row["round_seconds"] for row in records])),
        final_state_sha256=last["state_sha256"],
    )
    if graph_rows:
        states = [row["state"] for row in graph_rows]
        result["graph"] = dict(
            cold_start_rounds=sum(1 for row in graph_rows
                                  if row["calibrated_fraction"] == 0.0),
            insufficient_evidence_rounds=sum(1 for value in states if value == "insufficient_evidence"),
            no_anomaly_evidence_rounds=sum(1 for value in states if value == "no_anomaly_evidence"),
            anomalous_structure_rounds=sum(1 for value in states if value == "anomalous_structure"),
            confirmed_group_rounds=sum(1 for row in graph_rows if row["confirmed_groups"] > 0),
            downweighted_client_rounds=sum(row["downweighted_clients"] for row in graph_rows),
            last10_mean_calibrated_fraction=float(np.mean(
                [row["calibrated_fraction"] for row in graph_rows[-10:]])),
            last10_mean_anchor_weight=float(np.mean(
                [row["anchor_weight"] for row in graph_rows[-10:]])),
        )
    return result


def report(cfg, output, rows, completed):
    if cfg.get("purpose") == MULTI_ATTACK_PURPOSE:
        return _report_multi_attack(cfg, output, rows, completed)
    return _report_dba(cfg, output, rows, completed)


def _report_multi_attack(cfg, output, rows, completed):
    summary = dict(
        purpose=cfg["purpose"], seed=cfg["seed"], budget_rounds=cfg["total_rounds"],
        completed_rounds=completed, groups={}, comparisons={},
        attack_strength={}, defended_descriptive_checks={},
        defense_evaluated=True, final_test_set_used=False,
        normalization=cfg["normalization"],
        defense_variant=cfg.get("defense_variant"),
        interpretation=cfg.get("interpretation", ""),
    )
    attack_by_name = {group["name"]: group["attack"] for group in cfg["groups"]}
    names = group_names(cfg)
    for name in names:
        if rows.get(name):
            summary["groups"][name] = group_stats(
                rows[name], attack_by_name[name] != "none")
    clean_fed = group_name(cfg, "none", "fedavg")
    clean_rg = group_name(cfg, "none", "relation_graph")
    if rows.get(clean_fed) and rows.get(clean_rg):
        summary["comparisons"]["clean_accuracy_delta_graph_minus_fedavg"] = (
            rows[clean_rg][-1]["clean"]["accuracy"]
            - rows[clean_fed][-1]["clean"]["accuracy"])
        summary["comparisons"]["clean_last10_accuracy_delta_graph_minus_fedavg"] = (
            float(np.mean([row["clean"]["accuracy"] for row in rows[clean_rg][-10:]]))
            - float(np.mean([row["clean"]["accuracy"] for row in rows[clean_fed][-10:]])))
    for attack in ATTACK_TYPES:
        fed_name = group_name(cfg, attack, "fedavg")
        rg_name = group_name(cfg, attack, "relation_graph")
        fed, rg = rows.get(fed_name, []), rows.get(rg_name, [])
        if fed and rg:
            summary["comparisons"][f"{attack}_asr_delta_graph_minus_fedavg"] = (
                rg[-1]["asr"]["asr"] - fed[-1]["asr"]["asr"])
            summary["comparisons"][f"{attack}_last10_asr_delta_graph_minus_fedavg"] = (
                float(np.mean([row["asr"]["asr"] for row in rg[-10:]]))
                - float(np.mean([row["asr"]["asr"] for row in fed[-10:]])))
            summary["comparisons"][f"{attack}_peak_asr_fedavg"] = max(
                row["asr"]["asr"] for row in fed)
            summary["comparisons"][f"{attack}_peak_asr_relation_graph"] = max(
                row["asr"]["asr"] for row in rg)
        if fed and rows.get(clean_fed):
            summary["attack_strength"][attack] = qualification(
                cfg, rows[clean_fed], fed, fed_name)
        if rg and rows.get(clean_rg):
            summary["defended_descriptive_checks"][attack] = qualification(
                cfg, rows[clean_rg], rg, rg_name)
    if all(rows.get(name) for name in names):
        summary["completed_rounds"] = min(len(rows[name]) for name in names)
    dump(output / "summary.json", summary)
    return summary


def _report_dba(cfg, output, rows, completed):
    summary = dict(
        purpose=cfg["purpose"], seed=cfg["seed"], budget_rounds=cfg["total_rounds"],
        completed_rounds=completed, groups={}, comparisons={},
        defense_evaluated=True, final_test_set_used=False,
        normalization=cfg["normalization"],
        interpretation=cfg.get("interpretation", ""),
    )
    attack_by_name = {group["name"]: group["attack"] for group in cfg["groups"]}
    for name in NAMES:
        if rows.get(name):
            summary["groups"][name] = group_stats(
                rows[name], attack_by_name[name] != "none")
    clean_fedavg = rows.get("clean_fedavg", [])
    clean_graph = rows.get("clean_relation_graph", [])
    dba_fedavg = rows.get("dba_fedavg", [])
    dba_graph = rows.get("dba_relation_graph", [])
    if clean_fedavg and clean_graph:
        summary["comparisons"]["clean_accuracy_delta_graph_minus_fedavg"] = (
            clean_graph[-1]["clean"]["accuracy"] - clean_fedavg[-1]["clean"]["accuracy"])
        summary["comparisons"]["clean_last10_accuracy_delta_graph_minus_fedavg"] = (
            float(np.mean([row["clean"]["accuracy"] for row in clean_graph[-10:]]))
            - float(np.mean([row["clean"]["accuracy"] for row in clean_fedavg[-10:]])))
    if dba_fedavg and dba_graph:
        summary["comparisons"]["dba_asr_delta_graph_minus_fedavg"] = (
            dba_graph[-1]["asr"]["asr"] - dba_fedavg[-1]["asr"]["asr"])
        summary["comparisons"]["dba_peak_asr_fedavg"] = max(row["asr"]["asr"] for row in dba_fedavg)
        summary["comparisons"]["dba_peak_asr_relation_graph"] = max(
            row["asr"]["asr"] for row in dba_graph)
    if dba_fedavg and clean_fedavg:
        summary["attack_strength_fedavg"] = qualification(
            cfg, clean_fedavg, dba_fedavg, "dba_fedavg")
    if dba_graph and clean_graph:
        summary["defended_descriptive_checks"] = qualification(
            cfg, clean_graph, dba_graph, "dba_relation_graph")
    if dba_fedavg and clean_fedavg and dba_graph and clean_graph:
        summary["completed_rounds"] = min(
            len(rows[name]) for name in NAMES)
    dump(output / "summary.json", summary)
    return summary


def reconcile(output, completed, names):
    rows, tails = {}, {}
    for name in names:
        for stem in (f"rounds_{name}", f"graph_{name}"):
            path = output / f"{stem}.jsonl"
            if not path.exists():
                continue
            records = load_lines(path)
            retained = [row for row in records if row["round"] <= completed]
            if [row["round"] for row in retained] != list(range(1, completed + 1)):
                raise ValueError(f"Checkpoint and {stem} logs disagree")
            if len(retained) != len(records):
                tails[stem] = records[len(retained):]
                path.write_text("".join(json.dumps(row, allow_nan=False) + "\n"
                                        for row in retained), encoding="utf-8")
            if stem.startswith("rounds_"):
                rows[name] = retained
    if tails:
        dump(output / f"interrupted_tails_{time.time_ns()}.json", tails)
    for name in names:
        rows.setdefault(name, [])
    return rows


def defense_state(defense):
    return defense.state_dict() if hasattr(defense, "state_dict") else None


def load_defense_state(defense, state):
    if state is not None:
        if not hasattr(defense, "load_state_dict"):
            raise ValueError("Checkpoint contains defense state for a stateless defense")
        defense.load_state_dict(state)


def verify_smoke(output, cfg, engine, rows, bundle):
    results = {"groups": {}, "passed": True}
    rounds = cfg["total_rounds"]
    for group in cfg["groups"]:
        name, attack = group["name"], group["attack"]
        records = rows[name]
        checks = dict(rounds_complete=(len(records) == rounds and
                                       [row["round"] for row in records] == list(range(1, rounds + 1))),
                      asr_denominator=all(row["asr"]["asr_total"] == 1800 for row in records),
                      clean_count=all(row["clean"]["count"] == 2000 for row in records))
        if attack == "dba":
            active = []
            fragments = set()
            poison_counts = set()
            scaled = set()
            for row in records:
                for client in row["clients"]:
                    if client["is_malicious_evaluation_only"]:
                        if client["train"].get("attack_active") is not True:
                            active.append((row["round"], client["client_id"], False))
                        fragments.add(client["train"].get("dba_fragment_index"))
                        poison_counts.add(client["train"].get("dba_poisoned_examples"))
                        scaled.add(client["train"].get("dba_update_scaled"))
            checks["dba_four_fragments"] = fragments == {0, 1, 2, 3}
            checks["dba_poisoned_examples_250"] = poison_counts == {250}
            checks["dba_no_upload_scaling"] = scaled == {False}
            checks["dba_all_malicious_rounds_active"] = not active
            checks["dba_only_four_malicious"] = all(
                sum(client["is_malicious_evaluation_only"] for client in row["clients"]) == 4
                for row in records)
        elif attack in ATTACK_TYPES:
            active = []
            poison_counts = set()
            eligible_counts = set()
            for row in records:
                for client in row["clients"]:
                    if client["is_malicious_evaluation_only"]:
                        train = client["train"]
                        if train.get("attack_active") is not True:
                            active.append((row["round"], client["client_id"]))
                        poison_counts.add(train.get("backdoor_poisoned_examples"))
                        eligible_counts.add(train.get("backdoor_eligible_examples"))
            checks["attack_all_malicious_rounds_active"] = not active
            checks["attack_only_four_malicious"] = all(
                sum(client["is_malicious_evaluation_only"] for client in row["clients"]) == 4
                for row in records)
            checks["attack_poisoning_positive"] = all(
                isinstance(value, (int, float)) and value > 0 for value in poison_counts)
            checks["attack_eligible_positive"] = all(
                isinstance(value, (int, float)) and value > 0 for value in eligible_counts)
            if attack == "pgd":
                first = [client["train"] for client in records[0]["clients"]
                         if client["is_malicious_evaluation_only"]]
                later = [client["train"] for row in records[1:] for client in row["clients"]
                         if client["is_malicious_evaluation_only"]]
                checks["pgd_first_round_unconstrained"] = all(
                    train.get("pgd_projection_count") == 0
                    and not train.get("pgd_reference_available")
                    for train in first)
                checks["pgd_projections_per_epoch"] = all(
                    train.get("pgd_projection_count") == cfg["client"]["local_epochs"]
                    and bool(train.get("pgd_reference_available"))
                    for train in later)
        else:
            checks["no_attack_active"] = all(
                client["train"].get("attack_active") in (None, False)
                for row in records for client in row["clients"])
        if group["defense"] == "relation_graph":
            saved = defense_state(engine.defenses[name])
            restored = get_defense(
                DefenseConfig(enabled=True, type="relation_graph",
                              custom_params={"settings": cfg["relation_graph"]["settings"]}),
                num_clients=cfg["clients"])
            restored.load_state_dict(json.loads(json.dumps(saved, allow_nan=False)))
            checks["defense_state_roundtrip"] = restored.state_dict() == saved
            checks["defense_last_round"] = saved["last_round"] == rounds
        checks["checkpoint_round"] = bundle["completed_round"] == rounds
        checks["checkpoint_state_hash"] = (
            state_hash(bundle["states"][name]) == records[-1]["state_sha256"])
        results["groups"][name] = checks
        if not all(checks.values()):
            results["passed"] = False
    dump(output / "smoke_verification.json", results)
    if not results["passed"]:
        raise ValueError(f"Smoke wiring check failed: {results}")


def run(cfg, output, data_dir, resume=False, max_rounds=None):
    validate(cfg)
    names = group_names(cfg)
    if max_rounds is not None and not 1 <= max_rounds <= cfg["total_rounds"]:
        raise ValueError("--max-rounds must be inside the budget")
    if output.exists() != resume:
        raise FileExistsError("Use a new output directory, or --resume for an existing run")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    train, view, labels, dataset_inputs = load_training_data(cfg, data_dir)
    current_sources = source_hashes()

    if resume:
        saved_cfg = read(output / "protocol.json")
        manifest = read(output / "manifest.json")
        if saved_cfg != cfg:
            raise ValueError("Resume requires an identical frozen protocol")
        if manifest["source_sha256"] != current_sources:
            raise ValueError("Resume requires identical sources")
        if manifest["dataset_inputs_sha256"] != dataset_inputs:
            raise ValueError("Dataset changed at resume")
        if manifest["executed_groups"] != list(names):
            raise ValueError("Resume groups changed")
        with np.load(output / "split_indices.npz", allow_pickle=False) as archive:
            indices = {key: archive[key].copy() for key in archive.files}
        bundle = load_checkpoint(output / "latest_bundle.pt")
        completed = bundle["completed_round"]
        states = bundle["states"]
        rows = reconcile(output, completed, names)
        for name in names:
            expected = (rows[name][-1]["state_sha256"] if completed
                        else manifest["initial_state_sha256"])
            if state_hash(states[name]) != expected:
                raise ValueError(f"{name}: checkpoint state does not match committed logs")
        append(output / "resume_journal.jsonl",
               dict(pid=os.getpid(), completed_round=completed, unix=time.time()))
    else:
        output.mkdir(parents=True)
        dump(output / "protocol.json", cfg)
        initial = paired_model_factory(cfg)
        initial_state = {key: value.detach().cpu().clone()
                         for key, value in initial.state_dict().items()}
        initial_hash = state_hash(initial_state)
        if any(state_hash({key: value.clone() for key, value in initial_state.items()})
               != initial_hash for _ in range(1)):
            raise ValueError("Initial state hashing is not reproducible")
        save_checkpoint(output / "initial_state.pt",
                        dict(round=0, state=initial_state, sha256=initial_hash))
        manifest = dict(
            created_unix=time.time(), pid=os.getpid(), seed=cfg["seed"],
            groups=list(names), executed_groups=list(names),
            source_sha256=current_sources, attack_source_sha256=attack_source_hash(),
            dataset_inputs_sha256=dataset_inputs, initial_state_sha256=initial_hash,
            python=platform.python_version(), torch=str(torch.__version__),
            numpy=np.__version__, platform=platform.platform(),
            device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            final_test_set_used=False, client_training_authorized=True,
            smoke_wiring_check=cfg.get("smoke_wiring_check"),
        )
        dump(output / "manifest.json", manifest)
        for name in current_sources:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        indices = split_indices(labels, cfg)
        np.savez_compressed(output / "split_indices.npz", **indices)
        np.savez_compressed(
            output / "evaluation_order.npz",
            indices=evaluation_order(indices["validation"], cfg["evaluation_order_seed"]))
        dump(output / "splits.json", {
            key: dict(count=len(value),
                      sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                      class_counts=np.bincount(np.asarray(labels)[value], minlength=10).tolist())
            for key, value in indices.items()})
        states = {name: copy.deepcopy(initial_state) for name in names}
        if len({state_hash(states[name]) for name in names}) != 1:
            raise ValueError("Paired groups do not share the initial model")
        for attack in sorted({group["attack"] for group in cfg["groups"]} | {"none"}):
            dump(output / f"contract_{attack}.json",
                 attack_contract_payload(attack, asdict(pilot_attack_cfg(cfg, attack))))
        rows, completed = {name: [] for name in names}, 0

    dump(output / "runtime_owner.json", dict(pid=os.getpid(), started_unix=time.time()))
    engine = Engine(cfg, indices, train, view, device)
    if resume:
        for name in names:
            load_defense_state(engine.defenses[name], bundle["defenses"][name])
        saved_reference = bundle.get("pgd_reference", {})
        engine.pgd_reference = {name: saved_reference.get(name) for name in names}

    def checkpoint(number):
        return dict(
            completed_round=number,
            states={name: states[name] for name in names},
            defenses={name: defense_state(engine.defenses[name]) for name in names},
            pgd_reference=dict(engine.pgd_reference),
            initial_state_sha256=manifest["initial_state_sha256"],
        )

    if not resume:
        save_checkpoint(output / "latest_bundle.pt", checkpoint(0))
    summary = report(cfg, output, rows, completed)
    print(f"START relation_graph paired pilot seed={cfg['seed']} device={device}; "
          f"{cfg['total_rounds']} rounds; groups={list(names)}", flush=True)
    for number in range(completed + 1, cfg["total_rounds"] + 1):
        check_stop(output)
        if source_hashes() != current_sources:
            raise RuntimeError("Source changed; refusing a mixed-code experiment")
        for group in cfg["groups"]:
            check_stop(output)
            name = group["name"]
            state, row, graph_row = engine.fit_round(group, states[name], number, output)
            states[name] = state
            append(output / f"rounds_{name}.jsonl", row)
            if graph_row is not None:
                append(output / f"graph_{name}.jsonl", graph_row)
            rows[name].append(row)
            per_group = dict(round=number, group=name, attack=group["attack"],
                             defense=group["defense"], state=state,
                             defense_state=defense_state(engine.defenses[name]),
                             state_sha256=row["state_sha256"])
            save_checkpoint(output / f"latest_{name}.pt", per_group)
            if number in cfg["snapshots"]:
                save_checkpoint(output / "snapshots" / name / f"round_{number:03d}.pt",
                                per_group)
            graph_text = ("" if row["graph"] is None else
                          f" state={row['graph']['state']}"
                          f" cal={row['graph']['calibrated_fraction']:.3f}"
                          f" down={row['graph']['downweighted_clients']}")
            print(f"{name} round {number}/{cfg['total_rounds']}: "
                  f"ACC={row['clean']['accuracy']:.4f} ASR={row['asr']['asr']:.4f}"
                  f"{graph_text} clients={row['client_seconds']:.1f}s"
                  f" aggregate={row['aggregation_seconds']:.2f}s", flush=True)
        save_checkpoint(output / "latest_bundle.pt", checkpoint(number))
        summary = report(cfg, output, rows, number)
        status(output, "round_completed", round=number,
               budget=cfg["total_rounds"])
        if max_rounds is not None and number >= max_rounds:
            status(output, "paused", round=number, budget=cfg["total_rounds"])
            print(f"PAUSED after round {number}; resume with --resume", flush=True)
            return
    last_completed = max((rows[name][-1]["round"] for name in names if rows[name]),
                         default=completed)
    if last_completed < cfg["total_rounds"]:
        status(output, "paused", round=last_completed, budget=cfg["total_rounds"])
        print(f"PAUSED at round {last_completed}; resume with --resume", flush=True)
        return
    bundle = load_checkpoint(output / "latest_bundle.pt")
    if bundle["completed_round"] != cfg["total_rounds"]:
        raise ValueError("Final checkpoint round mismatch")
    for name in names:
        if state_hash(bundle["states"][name]) != rows[name][-1]["state_sha256"]:
            raise ValueError(f"{name}: final checkpoint/log state mismatch")
        if defense_state(engine.defenses[name]) != bundle["defenses"][name]:
            raise ValueError(f"{name}: final defense state mismatch")
    if cfg.get("smoke_wiring_check"):
        verify_smoke(output, cfg, engine, rows, bundle)
    status(output, "completed", round=cfg["total_rounds"])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("config/relation_graph_mnist_paired_120.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-rounds", type=int, default=0,
                        help="Short wiring check; overrides total_rounds in this output only")
    parser.add_argument("--max-rounds", type=int, default=0,
                        help="Pause after this many committed rounds (0 = full budget)")
    args = parser.parse_args()
    try:
        cfg = read(args.config)
        if args.smoke_rounds:
            if args.resume or args.max_rounds:
                raise ValueError("Do not combine --smoke-rounds with --resume/--max-rounds")
            cfg = smoke_config(cfg, args.smoke_rounds)
        run(cfg, args.output, args.data_dir, args.resume,
            args.max_rounds or None)
    except BaseException as error:
        owner = args.output / "runtime_owner.json"
        if (owner.exists() and read(owner)["pid"] == os.getpid()
                and not isinstance(error, FileExistsError)):
            status(args.output,
                   "stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                   error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()