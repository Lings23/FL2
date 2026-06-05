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
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union

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


# ── Metrics aggregation helpers ───────────────────────────────────────────────

def _weighted_avg_metrics(results: List[Tuple[int, Dict]]) -> Dict:
    """Weighted average of scalar metrics across clients."""
    total = sum(n for n, _ in results)
    if total == 0:
        return {}
    agg: Dict[str, float] = {}
    for n, m in results:
        for k, v in m.items():
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

        # ── Defense aggregation ───────────────────────────────────────────────
        aggregated = self.defense.aggregate(updates)

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

        logger.info("Round %d aggregation done (defense=%s)",
                    server_round, self.defense.__class__.__name__)

        return ndarrays_to_parameters(aggregated), fit_metrics

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
