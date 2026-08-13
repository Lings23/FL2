"""
strategies/fed_strategy.py
---------------------------
Flower Strategy implementations that integrate the defense layer.

FedSecStrategy
    — Base strategy wrapping any BaseDefense
    — Supports FedAvg / FedProx / FedYogi aggregation as the inner optimizer
    — Emits rich per-round metrics

Extension interface
-------------------
Override aggregate_fit() for custom aggregation,
or override configure_fit() to inject per-round config (e.g. proximal μ).
"""

from __future__ import annotations

import logging
import json
import hashlib
import time
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    EvaluateRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from config.config_loader import StrategyConfig, DefenseConfig
from defenses.defense_base import BaseDefense, get_defense, UpdateList
from experiments.trial_plan import (
    TrialPlanError,
    TrialPlanV1,
    derive_seed,
    logical_id_key,
    sha256_json,
)

logger = logging.getLogger(__name__)

CLIENT_METADATA_KEYS = {
    "client_id",
    "is_malicious",
    "attack_active",
    "label_flip_eligible_examples",
    "label_flip_poisoned_examples",
    "label_flip_total_examples",
    "fit_seed",
    "attack_seed",
}


# ── Metrics aggregation helpers ───────────────────────────────────────────────

def _weighted_avg_metrics(results: List[Tuple[int, Dict]]) -> Dict:
    """Weighted average of scalar metrics across clients."""
    total = sum(n for n, _ in results)
    if total == 0:
        return {}
    agg: Dict[str, float] = {}
    for n, m in results:
        for k, v in m.items():
            if k in CLIENT_METADATA_KEYS:
                continue
            agg[k] = agg.get(k, 0.0) + (n / total) * float(v)
    return agg


# ── FedSecStrategy ────────────────────────────────────────────────────────────

class FedSecStrategy(Strategy):
    """
    Modular Flower strategy for security research.

    Parameters
    ----------
    strategy_cfg    : StrategyConfig from config.yaml
    defense_cfg     : DefenseConfig from config.yaml
    initial_params  : initial global model parameters (NDArrays)
    evaluate_fn     : optional server-side evaluation function
                      Signature: (server_round, params, config) -> (loss, metrics)
    num_clients     : total number of clients (used by FoolsGold)
    """

    def __init__(
        self,
        strategy_cfg: StrategyConfig,
        defense_cfg: DefenseConfig,
        initial_params: NDArrays,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, Dict], Optional[Tuple[float, Dict]]]
        ] = None,
        num_clients: int = 10,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 5,
        clients_per_round: int = 5,
        on_fit_config_fn: Optional[Callable[[int], Dict]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict]] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        server_update_fn: Optional[Callable[[NDArrays], NDArrays]] = None,
        sampling_seed: int = 42,
        pairing_mode: str = "legacy",
        sampling_protocol: str = "endpoint_uniform",
        trial_plan: Optional[TrialPlanV1] = None,
        parameter_roles: Optional[Dict[str, str]] = None,
        parameter_names: Optional[Dict[str, str]] = None,
    ):
        self.scfg = strategy_cfg
        self.defense: BaseDefense = get_defense(defense_cfg, num_clients=num_clients)
        self.global_params: NDArrays = initial_params
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.clients_per_round = clients_per_round
        self.on_fit_config_fn = on_fit_config_fn or self._default_fit_config
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.server_update_fn = server_update_fn
        self.sampling_seed = int(sampling_seed)
        self.pairing_mode = str(pairing_mode)
        self.sampling_protocol = str(sampling_protocol)
        self.trial_plan = trial_plan
        self.parameter_roles = {
            str(key): str(value) for key, value in (parameter_roles or {}).items()
        }
        self.parameter_names = {
            str(key): str(value) for key, value in (parameter_names or {}).items()
        }
        if self.pairing_mode == "strict" and self.trial_plan is None:
            raise TrialPlanError("Strict pairing requires a TrialPlanV1")
        if self.trial_plan is not None and (
            self.trial_plan.sampling_protocol != self.sampling_protocol
        ):
            raise TrialPlanError(
                "Configured sampling protocol does not match TrialPlanV1"
            )
        self.last_client_records: List[Dict[str, Any]] = []
        self._planned_partition_ids: Dict[int, List[str]] = {}
        self._planned_evaluate_partition_ids: Dict[int, List[str]] = {}
        self._fit_seed_digests: Dict[int, str] = {}

        # FedYogi / FedAdam server-side state
        self._m: Optional[NDArrays] = None   # first moment
        self._v: Optional[NDArrays] = None   # second moment

        logger.info("Strategy: %s | Defense: %s",
                    strategy_cfg.name, defense_cfg.type)

    # ── Flower Strategy interface ──────────────────────────────────────────────

    def initialize_parameters(
        self, client_manager
    ) -> Optional[Parameters]:
        return ndarrays_to_parameters(self.global_params)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        config = self.on_fit_config_fn(server_round)
        if self.pairing_mode == "strict":
            assert self.trial_plan is not None
            round_plan = self.trial_plan.round(server_round)
            clients = self._resolve_planned_clients(
                client_manager, round_plan["partition_ids"]
            )
            planned_ids = [str(value) for value in round_plan["partition_ids"]]
            self._planned_partition_ids[int(server_round)] = planned_ids
            self._fit_seed_digests[int(server_round)] = str(
                round_plan["fit_seed_digest"]
            )
            return [
                (
                    client,
                    fl.common.FitIns(parameters, {
                        **config,
                        "fit_seed": int(round_plan["fit_seeds"][partition_id]),
                        "attack_seed": int(round_plan["attack_seeds"][partition_id]),
                        "deterministic_client_training": True,
                        "trial_plan_hash": self.trial_plan.trial_plan_hash,
                    }),
                )
                for client, partition_id in zip(clients, planned_ids)
            ]
        fit_ins = fl.common.FitIns(parameters, config)
        if getattr(self.defense, "requires_principal_first_sampling", False):
            clients = self._principal_first_sample(
                client_manager,
                self.clients_per_round,
                self.min_fit_clients,
                self.sampling_seed + int(server_round),
            )
        else:
            clients = self._deterministic_sample(
                client_manager, self.clients_per_round, self.min_fit_clients,
                self.sampling_seed + int(server_round),
            )
        self._planned_partition_ids[int(server_round)] = [
            str(getattr(client, "partition_id", client.cid)) for client in clients
        ]
        return [(c, fit_ins) for c in clients]

    @staticmethod
    def _resolve_planned_clients(client_manager, planned_ids):
        available = getattr(client_manager, "all", lambda: {})()
        by_partition: Dict[str, Any] = {}
        duplicates: set[str] = set()
        for client in available.values():
            raw = getattr(client, "partition_id", client.cid)
            if isinstance(raw, float) and raw.is_integer():
                raw = int(raw)
            partition_id = str(raw)
            if partition_id in by_partition:
                duplicates.add(partition_id)
            by_partition[partition_id] = client
        if duplicates:
            raise TrialPlanError(
                f"Multiple available endpoints claim partition IDs: {sorted(duplicates, key=logical_id_key)}"
            )
        missing = [str(value) for value in planned_ids if str(value) not in by_partition]
        if missing:
            raise TrialPlanError(
                f"Strict trial plan clients are unavailable: {missing}; no replacement sampling is allowed"
            )
        return [by_partition[str(value)] for value in planned_ids]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        config = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
        eval_ins = fl.common.EvaluateIns(parameters, config)
        if self.pairing_mode == "strict":
            assert self.trial_plan is not None
            round_plan = self.trial_plan.round(server_round)
            fit_ids = [
                str(value)
                for value in round_plan["partition_ids"]
            ]
            planned_ids = fit_ids[: min(self.min_evaluate_clients, len(fit_ids))]
            clients = self._resolve_planned_clients(client_manager, planned_ids)
            self._planned_evaluate_partition_ids[int(server_round)] = planned_ids
            return [
                (
                    client,
                    fl.common.EvaluateIns(
                        parameters,
                        {
                            **config,
                            "server_round": int(server_round),
                            "eval_seed": int(round_plan["evaluate_seeds"][partition_id]),
                            "deterministic_client_evaluation": True,
                            "trial_plan_hash": self.trial_plan.trial_plan_hash,
                        },
                    ),
                )
                for client, partition_id in zip(clients, planned_ids)
            ]
        else:
            clients = self._deterministic_sample(
                client_manager, self.min_evaluate_clients, self.min_evaluate_clients,
                self.sampling_seed + 100_000 + int(server_round),
            )
        return [(c, eval_ins) for c in clients]

    @staticmethod
    def _deterministic_sample(client_manager, count: int, minimum: int, seed: int):
        """Select the same clients for a round across defense configurations."""
        available = getattr(client_manager, "all", lambda: {})()
        def logical_id(client) -> tuple[int, Union[int, str]]:
            raw = getattr(client, "partition_id", None)
            if raw is not None:
                try:
                    return (0, int(raw))
                except (TypeError, ValueError):
                    return (1, str(raw))
            return (1, str(client.cid))

        clients = sorted(available.values(), key=logical_id)
        if len(clients) < minimum:
            return client_manager.sample(num_clients=count, min_num_clients=minimum)
        count = min(count, len(clients))
        indices = np.random.default_rng(seed).choice(len(clients), size=count, replace=False)
        return [clients[int(index)] for index in sorted(indices)]

    def _principal_first_sample(
        self,
        client_manager,
        count: int,
        minimum: int,
        seed: int,
    ):
        """Select principals first, then one registered endpoint per principal."""

        available = getattr(client_manager, "all", lambda: {})()
        groups: Dict[str, List[Any]] = {}
        for client in available.values():
            raw = getattr(client, "partition_id", client.cid)
            client_id = str(int(raw)) if isinstance(raw, float) and raw.is_integer() else str(raw)
            principal = str(self.defense.principal_id_for(client_id))  # type: ignore[attr-defined]
            groups.setdefault(principal, []).append(client)
        if len(groups) < int(minimum):
            raise RuntimeError(
                "RTC-v3 principal-first sampling has fewer available principals "
                f"({len(groups)}) than min_fit_clients ({minimum})"
            )
        principal_ids = sorted(groups)
        selected_count = min(int(count), len(principal_ids))
        rng = np.random.default_rng(seed)
        selected_indices = sorted(
            int(index)
            for index in rng.choice(
                len(principal_ids), size=selected_count, replace=False
            )
        )
        selected = []
        for index in selected_indices:
            principal = principal_ids[index]
            endpoints = sorted(
                groups[principal],
                key=lambda client: str(
                    getattr(client, "partition_id", client.cid)
                ),
            )
            endpoint_index = int(rng.integers(0, len(endpoints)))
            selected.append(endpoints[endpoint_index])
        return selected

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if self.pairing_mode == "strict" and failures:
            raise TrialPlanError(
                f"Round {server_round} has {len(failures)} client failure(s); the whole run is invalid"
            )
        if not results:
            if self.pairing_mode == "strict":
                raise TrialPlanError(
                    f"Round {server_round} completed no planned clients"
                )
            return None, {}

        if failures:
            logger.warning("Round %d: %d client failures", server_round, len(failures))

        if self.pairing_mode == "strict":
            planned = self._planned_partition_ids.get(int(server_round), [])
            result_by_partition: Dict[str, Tuple[ClientProxy, FitRes]] = {}
            for item in results:
                partition_id = self._server_partition_id(*item)
                if partition_id in result_by_partition:
                    raise TrialPlanError(
                        f"Round {server_round} returned duplicate partition {partition_id}"
                    )
                result_by_partition[partition_id] = item
            missing = [value for value in planned if value not in result_by_partition]
            extra = [value for value in result_by_partition if value not in set(planned)]
            if missing or extra or len(results) != len(planned):
                raise TrialPlanError(
                    f"Round {server_round} result set differs from plan; missing={missing} extra={extra}"
                )
            # Async completion order must not affect floating-point reductions
            # or defenses whose histories are aligned to the result vector.
            results = [result_by_partition[value] for value in planned]

        # Unpack updates: [(params_list, num_samples), ...]
        updates: UpdateList = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        selected_partition_ids = [
            self._server_partition_id(client, fit_res)
            for client, fit_res in results
        ]
        client_ids = (
            list(selected_partition_ids)
            if self.pairing_mode == "strict"
            else [self._stable_client_id(client, fit_res) for client, fit_res in results]
        )
        malicious_labels = [
            bool(fit_res.metrics.get("is_malicious", False))
            for _, fit_res in results
        ]
        attack_active_labels = [
            bool(fit_res.metrics.get("attack_active", False))
            for _, fit_res in results
        ]
        client_fit_metrics = [dict(fit_res.metrics or {}) for _, fit_res in results]
        if self.pairing_mode == "strict":
            assert self.trial_plan is not None
            round_plan = self.trial_plan.round(server_round)
            reported_fit_seeds = {
                partition_id: int(metrics.get("fit_seed", -1))
                for partition_id, metrics in zip(selected_partition_ids, client_fit_metrics)
            }
            reported_attack_seeds = {
                partition_id: int(metrics.get("attack_seed", -1))
                for partition_id, metrics in zip(selected_partition_ids, client_fit_metrics)
            }
            if sha256_json(reported_fit_seeds) != str(round_plan["fit_seed_digest"]):
                raise TrialPlanError(
                    f"Round {server_round} clients did not report the planned fit random streams"
                )
            if sha256_json(reported_attack_seeds) != str(round_plan["attack_seed_digest"]):
                raise TrialPlanError(
                    f"Round {server_round} clients did not report the planned attack random streams"
                )

        # ── Defense aggregation ───────────────────────────────────────────────
        aggregation_started = time.perf_counter()
        principal_ids = None
        if getattr(self.defense, "requires_principal_first_sampling", False):
            if self.trial_plan is not None:
                principal_ids = [
                    str(value)
                    for value in self.trial_plan.round(server_round)["principal_ids"]
                ]
            else:
                principal_ids = [
                    str(self.defense.principal_id_for(client_id))  # type: ignore[attr-defined]
                    for client_id in client_ids
                ]
        self.defense.set_context(
            server_round,
            client_ids,
            self.global_params,
            principal_ids=principal_ids,
            server_optimizer=self.scfg.name,
            parameter_roles=(
                self.parameter_roles
                if bool(getattr(self.defense, "requires_parameter_roles", False))
                else None
            ),
            parameter_names=(
                self.parameter_names
                if bool(getattr(self.defense, "requires_parameter_roles", False))
                else None
            ),
        )
        if getattr(self.defense, "requires_server_update", False):
            if self.server_update_fn is None:
                raise RuntimeError("FLTrust requires a configured server root update function")
            server_update = self.server_update_fn(self.global_params)
            self.defense.set_server_update(server_update)  # type: ignore[attr-defined]
        defense_seed = derive_seed(
            self.sampling_seed,
            self.trial_plan.trial_plan_hash if self.trial_plan is not None else "legacy",
            self.defense.__class__.__name__,
            "defense",
            int(server_round),
        )
        setattr(self.defense, "_defense_round_seed", defense_seed)
        numpy_state = np.random.get_state()
        try:
            np.random.seed(defense_seed)
            aggregated = self.defense.aggregate(updates)
        finally:
            np.random.set_state(numpy_state)
        aggregation_time = time.perf_counter() - aggregation_started
        self.last_client_records = self._build_client_records(
            server_round,
            updates,
            client_ids,
            malicious_labels,
            attack_active_labels,
            client_fit_metrics,
        )

        # ── Optional server-side optimizer (FedYogi) ──────────────────────────
        name = self.scfg.name.lower()
        if name == "fedyogi":
            aggregated = self._fedyogi_step(aggregated)
        elif name == "fedadam":
            aggregated = self._fedadam_step(aggregated)
        # (FedProx changes happen on client side; FedAvg is the default)

        self.global_params = aggregated

        # Aggregate fit metrics
        fit_metrics: Dict[str, Scalar] = {}
        if results:
            raw = [(r.num_examples, r.metrics) for _, r in results if r.metrics]
            fit_metrics = _weighted_avg_metrics(raw)

        defense_metrics = getattr(self.defense, "last_round_metrics", {})
        for key, value in defense_metrics.items():
            if isinstance(value, (int, float, np.floating)):
                fit_metrics[key] = float(value)
            elif isinstance(value, (str, bool)):
                fit_metrics[key] = value
        fit_metrics.update(self._security_round_metrics(self.last_client_records))
        planned_partition_ids = self._planned_partition_ids.get(int(server_round), [])
        fit_metrics["selected_partition_ids"] = ",".join(selected_partition_ids)
        fit_metrics["planned_partition_ids"] = ",".join(planned_partition_ids)
        fit_metrics["planned_partition_ids_json"] = json.dumps(planned_partition_ids)
        fit_metrics["completed_partition_ids_json"] = json.dumps(selected_partition_ids)
        fit_metrics["pairing_mode"] = self.pairing_mode
        fit_metrics["sampling_protocol"] = self.sampling_protocol
        if self.trial_plan is not None:
            fit_metrics["trial_plan_hash"] = self.trial_plan.trial_plan_hash
            fit_metrics["pairing_group_id"] = self.trial_plan.pairing_group_id
            fit_metrics["fit_seed_digest"] = self._fit_seed_digests[int(server_round)]
        fit_metrics["aggregation_time_seconds"] = float(aggregation_time)
        fit_metrics["defense_random_seed"] = int(defense_seed)

        logger.info("Round %d aggregation done (defense=%s)",
                    server_round, self.defense.__class__.__name__)

        return ndarrays_to_parameters(aggregated), fit_metrics

    @staticmethod
    def _stable_client_id(client: ClientProxy, fit_res: FitRes) -> str:
        # Trust state must be keyed by server-side identity.  A client-reported
        # metrics["client_id"] is useful for logging, but using it here would
        # let an adversarial client reset or spoof temporal history.
        return str(client.cid)

    @staticmethod
    def _server_partition_id(client: ClientProxy, fit_res: FitRes) -> str:
        raw = getattr(client, "partition_id", None)
        if raw is None:
            raw = fit_res.metrics.get("client_id", client.cid)
        if isinstance(raw, float) and raw.is_integer():
            raw = int(raw)
        return str(raw)

    def _build_client_records(
        self,
        server_round: int,
        updates: UpdateList,
        client_ids: List[str],
        malicious_labels: List[bool],
        attack_active_labels: List[bool],
        client_fit_metrics: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        trust_map = getattr(self.defense, "last_client_trusts", {})
        effective_map = getattr(self.defense, "last_client_weights", {})
        aggregation_map = getattr(self.defense, "last_client_aggregation_weights", {})
        rtc_records = {
            str(record.client_id): record
            for record in getattr(self.defense, "_last_records", [])
        }
        cone_assignments = getattr(self.defense, "_last_cone_assignments", {})
        cone_similarities = getattr(self.defense, "_last_cone_similarities", {})
        sketch_exports = getattr(self.defense, "_last_sketches", {})
        semantic_batch = getattr(self.defense, "_last_semantic_batch", None)
        semantic_risks = getattr(self.defense, "_last_semantic_risks", ())
        semantic_q_values = getattr(self.defense, "_last_semantic_q", ())
        semantic_wide_export = bool(
            getattr(self.defense, "_export_sketches", False)
        )
        cone_update_eligible = getattr(
            self.defense, "_last_cone_update_eligible", {}
        )
        cumulative_q = getattr(self.defense, "_last_cumulative_q", {})
        direction_q = getattr(self.defense, "_last_direction_q", {})
        clipped_mask = list(getattr(self.defense, "_last_clipped_mask", []))
        constraint_tags = list(getattr(self.defense, "_last_constraint_tags", []))
        clip_norm = float(getattr(self.defense, "_last_clip_norm", float("inf")))

        records: List[Dict[str, Any]] = []
        for idx, (
            (params, num_examples), cid, is_malicious, attack_active, fit_metrics
        ) in enumerate(
            zip(
                updates,
                client_ids,
                malicious_labels,
                attack_active_labels,
                client_fit_metrics,
            )
        ):
            raw_norm = self._delta_norm(params, self.global_params)
            is_clipped = idx < len(clipped_mask) and bool(clipped_mask[idx])
            clipped_norm = min(raw_norm, clip_norm) if is_clipped else raw_norm
            aggregation_weight = aggregation_map.get(cid)
            rtc_record = rtc_records.get(str(cid))
            impact_norm = (
                float(aggregation_weight) * clipped_norm
                if aggregation_weight is not None
                else None
            )
            flags = []
            if idx < len(constraint_tags) and constraint_tags[idx]:
                flags.append(str(constraint_tags[idx]))
            if is_clipped:
                flags.append("clipped")

            record = {
                "round": int(server_round),
                "cid": cid,
                "is_malicious": bool(is_malicious),
                "attack_active": bool(attack_active),
                "num_examples": int(num_examples),
                "local_epochs": int(float(fit_metrics.get("local_epochs", 1))),
                "label_flip_eligible_examples": fit_metrics.get(
                    "label_flip_eligible_examples"
                ),
                "label_flip_poisoned_examples": fit_metrics.get(
                    "label_flip_poisoned_examples"
                ),
                "label_flip_total_examples": fit_metrics.get(
                    "label_flip_total_examples"
                ),
                "principal_id": getattr(rtc_record, "principal_id", cid),
                "validated_mass": getattr(rtc_record, "validated_mass", None),
                "nominal_mass": getattr(rtc_record, "nominal_mass", None),
                "rtc_v3_residual_norm": getattr(rtc_record, "residual_norm", None),
                "rtc_v3_cone_full": (
                    cone_assignments.get("full", ())[idx]
                    if idx < len(cone_assignments.get("full", ()))
                    else None
                ),
                "rtc_v3_cone_similarity_full": (
                    cone_similarities.get("full", ())[idx]
                    if idx < len(cone_similarities.get("full", ()))
                    else None
                ),
                "rtc_v3_cone_update_eligible_full": (
                    cone_update_eligible.get("full", ())[idx]
                    if idx < len(cone_update_eligible.get("full", ()))
                    else None
                ),
                "rtc_v3_cumulative_q_full": (
                    cumulative_q.get("full", ())[idx]
                    if idx < len(cumulative_q.get("full", ()))
                    else None
                ),
                "rtc_v3_direction_q_full": (
                    direction_q.get("full", ())[idx]
                    if idx < len(direction_q.get("full", ()))
                    else None
                ),
                "trust": trust_map.get(cid),
                "raw_delta_norm": raw_norm,
                "clipped_delta_norm": clipped_norm,
                "effective_weight": effective_map.get(cid),
                "aggregation_weight": aggregation_weight,
                "impact_norm": impact_norm,
                "clipped": is_clipped,
                "capped": "capped" in flags,
                "quarantined": "quarantined" in flags,
                "state": getattr(rtc_record, "state", None),
                "magnitude_risk": getattr(rtc_record, "magnitude_risk", None),
                "direction_risk": getattr(rtc_record, "direction_risk", None),
                "temporal_risk": getattr(rtc_record, "temporal_risk", None),
                "influence_risk": getattr(rtc_record, "influence_risk", None),
                "total_risk": getattr(rtc_record, "total_risk", None),
                "event_risk": getattr(rtc_record, "event_risk", None),
                "flags": ",".join(flags),
            }
            full_sketches = sketch_exports.get("full", ())
            if idx < len(full_sketches):
                for position, value in enumerate(full_sketches[idx]):
                    record[f"rtc_v3_sketch_full_{position:03d}"] = float(value)
            if semantic_batch is not None:
                record["rtc_v3_semantic_head_norm"] = float(
                    semantic_batch.head_norms[idx]
                )
                if semantic_wide_export:
                    for class_index, value in enumerate(semantic_batch.row_norms[idx]):
                        record[
                            f"rtc_v3_semantic_row_norm_{class_index:03d}"
                        ] = float(value)
                    for pair_index, value in enumerate(semantic_batch.raw_pairs[idx]):
                        record[
                            f"rtc_v3_semantic_pair_raw_{pair_index:03d}"
                        ] = float(value)
                pair_index = int(semantic_batch.top_pair_indices[idx])
                source, target = semantic_batch.pairs[pair_index]
                signature = np.asarray(
                    semantic_batch.top_pair_signatures[idx], dtype=np.float32
                )
                record.update(
                    {
                        "semantic_top_source": int(source),
                        "semantic_top_target": int(target),
                        "semantic_raw_score": float(
                            semantic_batch.raw_pairs[idx, pair_index]
                        ),
                        "semantic_z": float(
                            semantic_batch.z_values[idx, pair_index]
                        ),
                        "semantic_risk": (
                            float(semantic_risks[idx])
                            if idx < len(semantic_risks)
                            else None
                        ),
                        "semantic_q": (
                            float(semantic_q_values[idx])
                            if idx < len(semantic_q_values)
                            else None
                        ),
                        "semantic_signature_digest": hashlib.sha256(
                            signature.tobytes(order="C")
                        ).hexdigest(),
                    }
                )
            records.append(record)
        return records

    @staticmethod
    def _delta_norm(params: NDArrays, global_params: NDArrays) -> float:
        total_sq = 0.0
        for param, reference in zip(params, global_params):
            if not np.issubdtype(reference.dtype, np.floating):
                continue
            delta = param.astype(np.float32) - reference.astype(np.float32)
            total_sq += float(np.sum(delta * delta))
        return float(np.sqrt(max(0.0, total_sq)))

    @staticmethod
    def _security_round_metrics(records: List[Dict[str, Any]]) -> Dict[str, Scalar]:
        metrics: Dict[str, Scalar] = {}
        if not records:
            return metrics

        malicious = [record for record in records if record["is_malicious"]]
        active_malicious = [record for record in records if record["attack_active"]]
        benign = [record for record in records if not record["is_malicious"]]
        metrics["selected_malicious_clients"] = len(malicious)
        metrics["selected_active_attackers"] = len(active_malicious)
        metrics["selected_benign_clients"] = len(benign)
        metrics["attack_active"] = float(bool(active_malicious))

        label_flip_records = [
            record for record in records
            if record.get("label_flip_total_examples") is not None
        ]
        if label_flip_records:
            selected_training_exposures = sum(
                int(record["num_examples"]) * max(0, int(record.get("local_epochs", 1)))
                for record in records
            )
            poisoned_exposures = sum(
                int(float(record.get("label_flip_poisoned_examples") or 0))
                * max(0, int(record.get("local_epochs", 1)))
                for record in label_flip_records
                if bool(record.get("attack_active"))
            )
            eligible_exposures = sum(
                int(float(record.get("label_flip_eligible_examples") or 0))
                * max(0, int(record.get("local_epochs", 1)))
                for record in label_flip_records
                if bool(record.get("attack_active"))
            )
            active_malicious_exposures = sum(
                int(record["num_examples"]) * max(0, int(record.get("local_epochs", 1)))
                for record in active_malicious
            )
            metrics.update({
                "label_flip_poisoned_exposures": poisoned_exposures,
                "label_flip_eligible_exposures": eligible_exposures,
                "selected_training_exposures": selected_training_exposures,
                "label_flip_global_exposure_rate": (
                    poisoned_exposures / selected_training_exposures
                    if selected_training_exposures else 0.0
                ),
                "label_flip_eligible_poison_rate": (
                    poisoned_exposures / eligible_exposures
                    if eligible_exposures else 0.0
                ),
                "label_flip_active_malicious_exposure_rate": (
                    poisoned_exposures / active_malicious_exposures
                    if active_malicious_exposures else 0.0
                ),
            })

        if all(record["aggregation_weight"] is not None for record in records):
            metrics["malicious_aggregation_weight_share"] = float(sum(
                float(record["aggregation_weight"])
                for record in malicious
            ))
            metrics["active_attacker_weight_share"] = float(sum(
                float(record["aggregation_weight"])
                for record in active_malicious
            ))

        if all(record["impact_norm"] is not None for record in records):
            total_impact = sum(float(record["impact_norm"]) for record in records)
            malicious_impact = sum(float(record["impact_norm"]) for record in malicious)
            metrics["malicious_impact_share"] = (
                malicious_impact / total_impact if total_impact > 1e-12 else 0.0
            )
            active_impact = sum(
                float(record["impact_norm"]) for record in active_malicious
            )
            metrics["active_attacker_impact_share"] = (
                active_impact / total_impact if total_impact > 1e-12 else 0.0
            )

        malicious_trust = [
            float(record["trust"])
            for record in malicious
            if record["trust"] is not None
        ]
        benign_trust = [
            float(record["trust"])
            for record in benign
            if record["trust"] is not None
        ]
        active_trust = [
            float(record["trust"])
            for record in active_malicious
            if record["trust"] is not None
        ]
        if malicious_trust:
            metrics["malicious_trust_mean"] = float(np.mean(malicious_trust))
        if benign_trust:
            metrics["benign_trust_mean"] = float(np.mean(benign_trust))
        if active_trust and benign_trust:
            comparisons = [
                (
                    1.0
                    if attack_score < benign_score
                    else 0.5
                    if attack_score == benign_score
                    else 0.0
                )
                for attack_score in active_trust
                for benign_score in benign_trust
            ]
            metrics["trust_detection_auc"] = float(np.mean(comparisons))

        malicious_clipped = sum(
            1 for record in malicious if record["clipped"]
        )
        benign_clipped = sum(
            1 for record in benign if record["clipped"]
        )
        active_clipped = sum(
            1 for record in active_malicious if record["clipped"]
        )
        metrics["malicious_clipped_clients"] = malicious_clipped
        metrics["benign_clipped_clients"] = benign_clipped
        total_clipped = sum(1 for record in records if record["clipped"])
        metrics["clip_precision"] = (
            active_clipped / total_clipped if total_clipped else 0.0
        )
        metrics["clip_recall_active_attackers"] = (
            active_clipped / len(active_malicious) if active_malicious else 0.0
        )

        benign_quarantined = sum(
            1 for record in benign if record["quarantined"]
        )
        metrics["benign_quarantined_clients"] = benign_quarantined
        metrics["benign_quarantine_rate"] = (
            benign_quarantined / len(benign) if benign else 0.0
        )
        for state in ("watch", "restricted", "quarantined"):
            benign_count = sum(1 for record in benign if record.get("state") == state)
            malicious_count = sum(1 for record in malicious if record.get("state") == state)
            metrics[f"benign_{state}_rate"] = benign_count / len(benign) if benign else 0.0
            metrics[f"malicious_{state}_rate"] = (
                malicious_count / len(malicious) if malicious else 0.0
            )
        return metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if self.pairing_mode == "strict" and failures:
            raise TrialPlanError(
                f"Round {server_round} has {len(failures)} evaluation client failure(s)"
            )
        if not results:
            return None, {}

        if self.pairing_mode == "strict":
            planned = self._planned_evaluate_partition_ids.get(
                int(server_round), []
            )
            result_by_partition: Dict[str, Tuple[ClientProxy, EvaluateRes]] = {}
            for item in results:
                partition_id = self._server_partition_id(*item)
                if partition_id in result_by_partition:
                    raise TrialPlanError(
                        f"Round {server_round} returned duplicate evaluation "
                        f"partition {partition_id}"
                    )
                result_by_partition[partition_id] = item
            missing = [value for value in planned if value not in result_by_partition]
            extra = [value for value in result_by_partition if value not in set(planned)]
            if missing or extra or len(results) != len(planned):
                raise TrialPlanError(
                    f"Round {server_round} evaluation result set differs from "
                    f"plan; missing={missing} extra={extra}"
                )
            results = [result_by_partition[value] for value in planned]
            assert self.trial_plan is not None
            round_plan = self.trial_plan.round(server_round)
            for partition_id, (_, evaluate_res) in zip(planned, results):
                expected_seed = int(round_plan["evaluate_seeds"][partition_id])
                observed_seed = int(evaluate_res.metrics.get("eval_seed", -1))
                if observed_seed != expected_seed:
                    raise TrialPlanError(
                        f"Round {server_round} evaluation client {partition_id} "
                        "did not report the planned random stream"
                    )

        total = sum(r.num_examples for _, r in results)
        loss_agg = sum(r.loss * r.num_examples for _, r in results) / total
        raw = [(r.num_examples, r.metrics) for _, r in results if r.metrics]
        metrics = _weighted_avg_metrics(raw)
        metrics["loss"] = loss_agg
        if self.pairing_mode == "strict":
            metrics["planned_evaluate_partition_ids"] = ",".join(planned)
            metrics["completed_evaluate_partition_ids"] = ",".join(planned)
            metrics["evaluate_seed_digest"] = sha256_json({
                partition_id: int(round_plan["evaluate_seeds"][partition_id])
                for partition_id in planned
            })

        logger.info(
            "Round %d evaluate | loss=%.4f acc=%.4f",
            server_round, loss_agg, metrics.get("val_accuracy", float("nan"))
        )
        return loss_agg, metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            return None
        params_np = parameters_to_ndarrays(parameters)
        return self.evaluate_fn(server_round, params_np, {})

    # ── Server-side optimizers ────────────────────────────────────────────────

    def _fedyogi_step(self, delta: NDArrays) -> NDArrays:
        """FedYogi server update (Reddi et al. 2020)."""
        β1, β2, η, τ = self.scfg.beta_1, self.scfg.beta_2, self.scfg.eta, self.scfg.tau

        if self._m is None:
            self._m = [np.zeros_like(p) for p in delta]
            self._v = [np.full_like(p, τ ** 2) for p in delta]

        new_params = []
        new_m, new_v = [], []
        for i, (p_global, Δ) in enumerate(zip(self.global_params, delta)):
            # Compute pseudo-gradient (difference)
            g = Δ - p_global
            m = β1 * self._m[i] + (1 - β1) * g           # type: ignore
            v = self._v[i] - (1 - β2) * np.sign(self._v[i] - g ** 2) * g ** 2  # type: ignore
            v = np.maximum(v, τ ** 2)
            new_params.append(p_global + η * m / (np.sqrt(v) + τ))
            new_m.append(m)
            new_v.append(v)

        self._m = new_m
        self._v = new_v
        return new_params

    def _fedadam_step(self, delta: NDArrays) -> NDArrays:
        """FedAdam server update."""
        β1, β2, η, τ = self.scfg.beta_1, self.scfg.beta_2, self.scfg.eta, self.scfg.tau

        if self._m is None:
            self._m = [np.zeros_like(p) for p in delta]
            self._v = [np.zeros_like(p) for p in delta]

        new_params, new_m, new_v = [], [], []
        for i, (p_global, Δ) in enumerate(zip(self.global_params, delta)):
            g = Δ - p_global
            m = β1 * self._m[i] + (1 - β1) * g           # type: ignore
            v = β2 * self._v[i] + (1 - β2) * g ** 2      # type: ignore
            new_params.append(p_global + η * m / (np.sqrt(v) + τ))
            new_m.append(m)
            new_v.append(v)

        self._m, self._v = new_m, new_v
        return new_params

    # ── Default config functions ──────────────────────────────────────────────

    def _default_fit_config(self, server_round: int) -> Dict:
        config: Dict = {"server_round": server_round}
        # Pass proximal mu for FedProx clients
        if self.scfg.name.lower() == "fedprox":
            config["proximal_mu"] = self.scfg.proximal_mu
        return config
