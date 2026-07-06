"""FLTrust aggregation with explicit trusted-server update semantics."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from config.config_loader import DefenseConfig
from defenses.defense_base import BaseDefense, UpdateList


class FLTrustDefense(BaseDefense):
    """Bootstrap client trust from a clean server-side root update."""

    requires_server_update = True

    def __init__(self, cfg: DefenseConfig, **_: object):
        super().__init__(cfg)
        params = cfg.custom_params or {}
        self.server_lr = float(params.get("server_lr", 1.0))
        self.use_num_examples = bool(params.get("use_num_examples", False))
        self.zero_trust_fallback = str(
            params.get("zero_trust_fallback", "server_update")
        ).lower()
        if self.server_lr < 0:
            raise ValueError("FLTrust server_lr must be >= 0")
        if self.zero_trust_fallback not in {"server_update", "keep_global"}:
            raise ValueError("FLTrust zero_trust_fallback must be server_update or keep_global")
        self._server_update: Optional[List[np.ndarray]] = None

    def set_server_update(self, server_update: List[np.ndarray]) -> None:
        """Set the trusted model delta for the current round."""
        self._server_update = [value.copy() for value in server_update]

    def aggregate(self, updates: UpdateList):
        if not updates:
            raise ValueError("FLTrust requires at least one client update")
        if self._server_update is None or self._global_params is None:
            raise RuntimeError("FLTrust requires global context and a server root update")

        server_delta = self._flatten(self._server_update)
        if not np.all(np.isfinite(server_delta)):
            raise ValueError("FLTrust server update contains NaN or Inf")
        server_norm = float(np.linalg.norm(server_delta))
        client_ids = self._client_ids_for(len(updates))
        trust_scores = np.zeros(len(updates), dtype=np.float64)
        scaled_deltas = []

        for idx, (params, _) in enumerate(updates):
            client_delta = self._flatten_delta(params)
            if not np.all(np.isfinite(client_delta)):
                scaled_deltas.append(np.zeros_like(server_delta))
                continue
            client_norm = float(np.linalg.norm(client_delta))
            if server_norm <= 1e-12 or client_norm <= 1e-12:
                scaled_deltas.append(np.zeros_like(client_delta))
                continue
            cosine = float(np.dot(server_delta, client_delta) / (server_norm * client_norm))
            trust_scores[idx] = max(0.0, min(1.0, cosine))
            scaled_deltas.append(client_delta * (server_norm / client_norm))

        weights = trust_scores.copy()
        if self.use_num_examples:
            weights *= np.asarray([num for _, num in updates], dtype=np.float64)
        fallback = server_norm <= 1e-12 or float(weights.sum()) <= 1e-12
        if fallback:
            aggregate_delta = (
                server_delta.copy()
                if self.zero_trust_fallback == "server_update"
                else np.zeros_like(server_delta)
            )
        else:
            aggregate_delta = np.zeros_like(server_delta, dtype=np.float32)
            for weight, delta in zip(weights, scaled_deltas):
                aggregate_delta += float(weight / weights.sum()) * delta

        self.last_client_trusts = {
            cid: float(score) for cid, score in zip(client_ids, trust_scores)
        }
        self._record_scalar_weights(client_ids, weights)
        self.last_round_metrics = {
            "fltrust_mean_trust": float(np.mean(trust_scores)),
            "fltrust_positive_trust_clients": float(np.count_nonzero(trust_scores > 0)),
            "fltrust_zero_trust_clients": float(np.count_nonzero(trust_scores <= 0)),
            "fltrust_server_delta_norm": server_norm,
            "fltrust_fallback": float(fallback),
        }
        return self._apply_delta(aggregate_delta * self.server_lr)

    def _apply_delta(self, flat_delta: np.ndarray):
        result = []
        offset = 0
        assert self._global_params is not None
        for global_param in self._global_params:
            if np.issubdtype(global_param.dtype, np.floating):
                size = global_param.size
                delta = flat_delta[offset:offset + size].reshape(global_param.shape)
                result.append(
                    (global_param.astype(np.float32) + delta).astype(
                        global_param.dtype, copy=False
                    )
                )
                offset += size
            else:
                result.append(global_param.copy())
        return result
