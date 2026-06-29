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

logger = logging.getLogger(__name__)

CLIENT_METADATA_KEYS = {"client_id", "is_malicious", "attack_active"}


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
        self.last_client_records: List[Dict[str, Any]] = []

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
        fit_ins = fl.common.FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.clients_per_round,
            min_num_clients=self.min_fit_clients,
        )
        return [(c, fit_ins) for c in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        config = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
        eval_ins = fl.common.EvaluateIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients,
            min_num_clients=self.min_evaluate_clients,
        )
        return [(c, eval_ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        if failures:
            logger.warning("Round %d: %d client failures", server_round, len(failures))

        # Unpack updates: [(params_list, num_samples), ...]
        updates: UpdateList = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        client_ids = [
            self._stable_client_id(client, fit_res)
            for client, fit_res in results
        ]
        malicious_labels = [
            bool(fit_res.metrics.get("is_malicious", False))
            for _, fit_res in results
        ]
        attack_active_labels = [
            bool(fit_res.metrics.get("attack_active", False))
            for _, fit_res in results
        ]

        # ── Defense aggregation ───────────────────────────────────────────────
        aggregation_started = time.perf_counter()
        self.defense.set_context(server_round, client_ids, self.global_params)
        if hasattr(self.defense, "set_server_update"):
            if self.server_update_fn is None:
                raise RuntimeError("FLTrust requires a configured server root update function")
            server_update = self.server_update_fn(self.global_params)
            self.defense.set_server_update(server_update)  # type: ignore[attr-defined]
        aggregated = self.defense.aggregate(updates)
        aggregation_time = time.perf_counter() - aggregation_started
        self.last_client_records = self._build_client_records(
            server_round,
            updates,
            client_ids,
            malicious_labels,
            attack_active_labels,
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
        fit_metrics.update(self._security_round_metrics(self.last_client_records))
        fit_metrics["aggregation_time_seconds"] = float(aggregation_time)

        logger.info("Round %d aggregation done (defense=%s)",
                    server_round, self.defense.__class__.__name__)

        return ndarrays_to_parameters(aggregated), fit_metrics

    @staticmethod
    def _stable_client_id(client: ClientProxy, fit_res: FitRes) -> str:
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
    ) -> List[Dict[str, Any]]:
        trust_map = getattr(self.defense, "last_client_trusts", {})
        effective_map = getattr(self.defense, "last_client_weights", {})
        aggregation_map = getattr(self.defense, "last_client_aggregation_weights", {})
        clipped_mask = list(getattr(self.defense, "_last_clipped_mask", []))
        constraint_tags = list(getattr(self.defense, "_last_constraint_tags", []))
        clip_norm = float(getattr(self.defense, "_last_clip_norm", float("inf")))

        records: List[Dict[str, Any]] = []
        for idx, ((params, num_examples), cid, is_malicious, attack_active) in enumerate(
            zip(updates, client_ids, malicious_labels, attack_active_labels)
        ):
            raw_norm = self._delta_norm(params, self.global_params)
            is_clipped = idx < len(clipped_mask) and bool(clipped_mask[idx])
            clipped_norm = min(raw_norm, clip_norm) if is_clipped else raw_norm
            aggregation_weight = aggregation_map.get(cid)
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

            records.append({
                "round": int(server_round),
                "cid": cid,
                "is_malicious": bool(is_malicious),
                "attack_active": bool(attack_active),
                "num_examples": int(num_examples),
                "trust": trust_map.get(cid),
                "raw_delta_norm": raw_norm,
                "clipped_delta_norm": clipped_norm,
                "effective_weight": effective_map.get(cid),
                "aggregation_weight": aggregation_weight,
                "impact_norm": impact_norm,
                "clipped": is_clipped,
                "capped": "capped" in flags,
                "quarantined": "quarantined" in flags,
                "flags": ",".join(flags),
            })
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
        return metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total = sum(r.num_examples for _, r in results)
        loss_agg = sum(r.loss * r.num_examples for _, r in results) / total
        raw = [(r.num_examples, r.metrics) for _, r in results if r.metrics]
        metrics = _weighted_avg_metrics(raw)
        metrics["loss"] = loss_agg

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
