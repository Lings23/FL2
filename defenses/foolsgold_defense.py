"""Stateful FoolsGold aggregation."""

from __future__ import annotations

import logging
import json
from typing import Dict

import numpy as np

from config.config_loader import DefenseConfig
from defenses.defense_base import BaseDefense, UpdateList

logger = logging.getLogger(__name__)


class FoolsGoldDefense(BaseDefense):
    """Penalize clients with persistently similar update directions."""

    def __init__(self, cfg: DefenseConfig, num_clients: int = 100):
        super().__init__(cfg)
        params = cfg.custom_params or {}
        self.history_decay = float(params.get("history_decay", 1.0))
        self.history_max_idle_rounds = int(params.get("history_max_idle_rounds", 50))
        self.use_num_examples = bool(params.get("use_num_examples", True))
        self.logit_offset = float(params.get("logit_offset", 0.5))
        self.eps = float(params.get("similarity_eps", 1e-12))
        if not 0.0 < self.history_decay <= 1.0:
            raise ValueError("FoolsGold history_decay must be in (0, 1]")
        if self.history_max_idle_rounds < 0:
            raise ValueError("FoolsGold history_max_idle_rounds must be >= 0")
        self._history: Dict[str, np.ndarray] = {}
        self._last_seen: Dict[str, int] = {}
        self._num_clients = int(num_clients)

    def aggregate(self, updates: UpdateList):
        if not updates:
            raise ValueError("FoolsGold requires at least one client update")
        n = len(updates)
        client_ids = self._client_ids_for(n)
        vectors = np.asarray(
            [self._flatten_delta(params) for params, _ in updates], dtype=np.float32
        )
        if not np.all(np.isfinite(vectors)):
            raise ValueError("FoolsGold received NaN or Inf in a client update")

        self._prune_idle_histories()
        for cid, vector in zip(client_ids, vectors):
            previous = self._history.get(cid)
            if previous is not None and previous.shape != vector.shape:
                logger.warning("FoolsGold model shape changed; clearing contribution history")
                self._history.clear()
                self._last_seen.clear()
                previous = None
            self._history[cid] = (
                vector.copy()
                if previous is None
                else self.history_decay * previous + vector
            )
            self._last_seen[cid] = self._server_round

        histories = np.asarray([self._history[cid] for cid in client_ids], dtype=np.float32)
        alphas = self._foolsgold_weights(histories)
        raw_weights = alphas.astype(np.float64)
        if self.use_num_examples:
            raw_weights *= np.asarray([num for _, num in updates], dtype=np.float64)

        fallback = float(raw_weights.sum()) <= self.eps
        if fallback:
            raw_weights = np.asarray([num for _, num in updates], dtype=np.float64)

        self.last_client_trusts = {
            cid: float(alpha) for cid, alpha in zip(client_ids, alphas)
        }
        self._record_scalar_weights(client_ids, raw_weights)
        self.last_round_metrics = {
            "foolsgold_mean_trust": float(np.mean(alphas)),
            "foolsgold_min_trust": float(np.min(alphas)),
            "foolsgold_zero_weight_clients": float(np.count_nonzero(alphas <= self.eps)),
            "foolsgold_history_clients": float(len(self._history)),
            "foolsgold_fallback": float(fallback),
            "foolsgold_history_partition_ids_json": json.dumps(
                sorted(
                    self._history,
                    key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
                )
            ),
        }
        return self._weighted_average_with_weights(updates, raw_weights)

    def _foolsgold_weights(self, histories: np.ndarray) -> np.ndarray:
        n = histories.shape[0]
        if n == 1:
            return np.ones(1, dtype=np.float32)
        norms = np.linalg.norm(histories, axis=1)
        normalized = np.divide(
            histories,
            norms[:, None],
            out=np.zeros_like(histories),
            where=norms[:, None] > self.eps,
        )
        similarities = normalized @ normalized.T
        similarities = np.clip(similarities, 0.0, 1.0)
        np.fill_diagonal(similarities, 0.0)
        max_similarity = np.max(similarities, axis=1)

        # Pardoning from the FoolsGold algorithm.
        for i in range(n):
            for j in range(n):
                if i != j and max_similarity[i] < max_similarity[j] and max_similarity[j] > self.eps:
                    similarities[i, j] *= max_similarity[i] / max_similarity[j]

        alphas = np.clip(1.0 - np.max(similarities, axis=1), 0.0, 1.0)
        maximum = float(np.max(alphas))
        if maximum > self.eps:
            alphas /= maximum
        logit_input = np.clip(alphas, 1e-6, 1.0 - 1e-6)
        return np.clip(
            np.log(logit_input / (1.0 - logit_input)) + self.logit_offset,
            0.0,
            1.0,
        ).astype(np.float32)

    def _prune_idle_histories(self) -> None:
        if self.history_max_idle_rounds == 0:
            return
        stale = [
            cid for cid, last_round in self._last_seen.items()
            if self._server_round - last_round > self.history_max_idle_rounds
        ]
        for cid in stale:
            self._last_seen.pop(cid, None)
            self._history.pop(cid, None)
