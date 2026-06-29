"""
defenses/defense_base.py
-------------------------
Pluggable defense / robust aggregation framework.

Implemented defenses
--------------------
• krum           — Krum / Multi-Krum (Blanchard et al. 2017)
• trimmed_mean   — Coordinate-wise trimmed mean (Yin et al. 2018)
• median         — Coordinate-wise median
• fltrust        — FLTrust server-side cosine re-weighting (Cao et al. 2020)
• foolsgold      — FoolsGold contribution similarity penalisation (Fung et al. 2018)
• fedavg         — Standard FedAvg (no defense, baseline)

Extension interface
-------------------
1. Subclass BaseDefense and implement aggregate()
2. Register in DEFENSE_REGISTRY at the bottom
3. Set defense.type in config.yaml
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from config.config_loader import DefenseConfig

logger = logging.getLogger(__name__)

# Type alias: a list of (ndarrays, num_samples) tuples
UpdateList = List[Tuple[List[np.ndarray], int]]


# ── Base defense ──────────────────────────────────────────────────────────────

class BaseDefense(ABC):
    """
    Abstract base class for all defense / robust aggregation methods.

    Subclasses must implement:
        aggregate(updates: UpdateList) -> List[np.ndarray]

    `updates` is a list of (parameters, num_samples) tuples received
    from selected clients in a given round.

    Return value: aggregated global parameters as List[np.ndarray].
    """

    def __init__(self, cfg: DefenseConfig):
        self.cfg = cfg
        self._server_round = 0
        self._client_ids: List[str] = []
        self._global_params: Optional[List[np.ndarray]] = None
        self.last_client_trusts: Dict[str, float] = {}
        self.last_client_weights: Dict[str, float] = {}
        self.last_client_aggregation_weights: Dict[str, float] = {}
        self.last_round_metrics: Dict[str, float] = {}

    def set_context(
        self,
        server_round: int,
        client_ids: Sequence[str],
        global_params: Sequence[np.ndarray],
    ) -> None:
        """Optional per-round context for stateful defenses.

        Stateless defenses can ignore this hook. Stateful defenses use it to
        key histories by real client ids and to compute model deltas from the
        previous global parameters.
        """
        self._server_round = int(server_round)
        self._client_ids = [str(cid) for cid in client_ids]
        self._global_params = [p.copy() for p in global_params]

    @abstractmethod
    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
        ...

    # ── Utility helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _flatten(params: List[np.ndarray]) -> np.ndarray:
        """Flatten only floating-point parameters into a 1-D float32 vector.

        Non-floating buffers (e.g. BatchNorm ``num_batches_tracked``, which
        has dtype int64) are skipped so that distance / cosine computations
        are not corrupted by integer values.
        """
        return np.concatenate(
            [p.ravel().astype(np.float32)
             for p in params
             if np.issubdtype(p.dtype, np.floating)]
        )

    @staticmethod
    def _unflatten(flat: np.ndarray, template: List[np.ndarray]) -> List[np.ndarray]:
        """Reconstruct a parameter list from a flat float32 vector.

        Non-floating parameters are copied unchanged from *template* so that
        integer buffers (e.g. BatchNorm step counters) are preserved as-is.
        """
        result: List[np.ndarray] = []
        offset = 0
        for p in template:
            if np.issubdtype(p.dtype, np.floating):
                n = p.size
                result.append(
                    flat[offset: offset + n].reshape(p.shape).astype(p.dtype, copy=False)
                )
                offset += n
            else:
                # Integer / bool buffers: keep the template value unchanged
                result.append(p.copy())
        return result

    @staticmethod
    def _weighted_average(updates: UpdateList) -> List[np.ndarray]:
        """Dtype-safe weighted average (FedAvg numerics).

        The fix for the ``_UFuncOutputCastingError``:
        PyTorch state dicts can contain integer-typed buffers (e.g.
        ``num_batches_tracked`` in BatchNorm layers, dtype int64).
        Accumulating  ``w * p``  (float64) into a ``zeros_like`` array that
        inherited the int64 dtype raises a casting error in NumPy ≥ 1.24.

        Solution: always accumulate in float32, then cast each output tensor
        back to its original dtype at the end — rounding integer buffers.
        """
        total = sum(n for _, n in updates)
        orig_dtypes = [p.dtype for p in updates[0][0]]

        # Always accumulate in float32 to avoid int-dtype casting errors
        agg = [np.zeros_like(p, dtype=np.float32) for p in updates[0][0]]

        for params, n in updates:
            w = n / total
            for i, p in enumerate(params):
                agg[i] += w * p.astype(np.float32)

        # Restore original dtypes; round integer buffers (e.g. step counters)
        result: List[np.ndarray] = []
        for arr, dt in zip(agg, orig_dtypes):
            if np.issubdtype(dt, np.integer):
                result.append(np.round(arr).astype(dt))
            else:
                result.append(arr.astype(dt, copy=False))
        return result

    @staticmethod
    def _weighted_average_with_weights(
        updates: UpdateList,
        weights: Sequence[float],
    ) -> List[np.ndarray]:
        """Average model parameters with explicit non-negative weights."""
        arr = np.asarray(weights, dtype=np.float64)
        arr = np.maximum(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        total = float(arr.sum())
        if total <= 1e-12:
            return BaseDefense._weighted_average(updates)

        orig_dtypes = [p.dtype for p in updates[0][0]]
        agg = [np.zeros_like(p, dtype=np.float32) for p in updates[0][0]]
        for weight, (params, _) in zip(arr, updates):
            for idx, param in enumerate(params):
                agg[idx] += float(weight / total) * param.astype(np.float32)

        result = []
        for value, dtype in zip(agg, orig_dtypes):
            if np.issubdtype(dtype, np.integer):
                result.append(np.round(value).astype(dtype))
            else:
                result.append(value.astype(dtype, copy=False))
        return result

    def _client_ids_for(self, count: int) -> List[str]:
        if len(self._client_ids) == count:
            return list(self._client_ids)
        return [str(idx) for idx in range(count)]

    def _flatten_delta(self, params: Sequence[np.ndarray]) -> np.ndarray:
        if self._global_params is None:
            return self._flatten(list(params))
        chunks = [
            (param.astype(np.float32) - reference.astype(np.float32)).ravel()
            for param, reference in zip(params, self._global_params)
            if np.issubdtype(reference.dtype, np.floating)
        ]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def _record_scalar_weights(
        self,
        client_ids: Sequence[str],
        raw_weights: Sequence[float],
    ) -> None:
        weights = np.maximum(
            np.nan_to_num(
                np.asarray(raw_weights, dtype=np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            0.0,
        )
        total = float(weights.sum())
        normalized = weights / total if total > 1e-12 else np.zeros_like(weights)
        self.last_client_weights = {
            cid: float(weight) for cid, weight in zip(client_ids, weights)
        }
        self.last_client_aggregation_weights = {
            cid: float(weight) for cid, weight in zip(client_ids, normalized)
        }


# ── Standard FedAvg (no defense) ─────────────────────────────────────────────

class FedAvgDefense(BaseDefense):
    """Plain weighted average — reference baseline."""

    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
        self._record_scalar_weights(
            self._client_ids_for(len(updates)),
            [num_samples for _, num_samples in updates],
        )
        return self._weighted_average(updates)


# ── Krum ──────────────────────────────────────────────────────────────────────

class KrumDefense(BaseDefense):
    """
    Krum (Blanchard et al. 2017).
    Selects the update with minimum sum-of-distances to its k nearest neighbours.
    Multi-Krum averages the top-m selected updates.
    """

    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
        n = len(updates)
        f = max(0, int(self.cfg.krum_num_malicious))
        if n < 2 * f + 3:
            raise ValueError(
                f"Krum requires n >= 2f + 3, received n={n}, f={f}"
            )
        max_selected = n - f - 2
        m = max(1, int(self.cfg.krum_num_to_select))
        if m > max_selected:
            raise ValueError(
                f"Multi-Krum requires m <= n - f - 2, received m={m}, "
                f"n={n}, f={f}"
            )

        vectors = np.array([self._flatten(params) for params, _ in updates])

        # Pairwise squared distances
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(vectors[i] - vectors[j]) ** 2
                dist_matrix[i, j] = dist_matrix[j, i] = d

        # Krum score: sum of (n - f - 2) smallest distances
        k = n - f - 2
        scores = np.zeros(n)
        for i in range(n):
            dists = np.sort(dist_matrix[i])
            scores[i] = dists[1:k + 1].sum()   # skip self (0)

        # Select top-m (lowest score = most similar to others)
        selected_idx = np.argsort(scores)[:m].tolist()
        logger.debug("Krum selected clients: %s", selected_idx)

        selected_weights = [
            1.0 if idx in selected_idx else 0.0
            for idx in range(n)
        ]
        self._record_scalar_weights(self._client_ids_for(n), selected_weights)
        return self._weighted_average_with_weights(updates, selected_weights)


# ── Trimmed Mean ──────────────────────────────────────────────────────────────

class TrimmedMeanDefense(BaseDefense):
    """
    Coordinate-wise trimmed mean (Yin et al. 2018).
    Removes the top and bottom `trim_fraction` of values per coordinate.
    """

    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
        beta = self.cfg.trim_fraction
        vectors = np.array([self._flatten(params) for params, _ in updates])
        n = len(vectors)
        k = max(0, int(n * beta))
        if 2 * k >= n:
            raise ValueError(
                f"Trimmed mean requires 2 * floor(n * trim_fraction) < n; "
                f"received n={n}, trim_fraction={beta}"
            )

        # Sort along client axis, trim, then mean
        sorted_v = np.sort(vectors, axis=0)
        trimmed = sorted_v[k:n - k] if n - 2 * k > 0 else sorted_v
        agg_flat = trimmed.mean(axis=0)

        return self._unflatten(agg_flat, updates[0][0])


# ── Coordinate-wise Median ────────────────────────────────────────────────────

class MedianDefense(BaseDefense):
    """Coordinate-wise median."""

    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
        vectors = np.array([self._flatten(params) for params, _ in updates])
        median_flat = np.median(vectors, axis=0)
        return self._unflatten(median_flat, updates[0][0])


# ── FLTrust ───────────────────────────────────────────────────────────────────

class FLTrustDefense(BaseDefense):
    """
    FLTrust (Cao et al. 2020).

    The server maintains a small root dataset and computes a server
    gradient. Client contributions are re-weighted by their cosine
    similarity to the server gradient.

    Usage
    -----
    After construction, call set_server_update(server_params) once per round
    before calling aggregate(). The framework's server strategy should
    call this; see strategies/fedtrust_strategy.py for an example.
    """

    def __init__(self, cfg: DefenseConfig):
        super().__init__(cfg)
        self._server_update: Optional[List[np.ndarray]] = None

    def set_server_update(self, server_update: List[np.ndarray]) -> None:
        self._server_update = server_update

    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
        if self._server_update is None or self._global_params is None:
            raise RuntimeError(
                "FLTrust requires global context and a server root update every round"
            )

        server_delta = self._flatten(self._server_update)
        server_norm = float(np.linalg.norm(server_delta))
        if server_norm < 1e-9:
            logger.warning("FLTrust server root update has near-zero norm; keeping global model.")
            client_ids = self._client_ids_for(len(updates))
            self.last_client_trusts = {cid: 0.0 for cid in client_ids}
            self._record_scalar_weights(client_ids, [0.0 for _ in updates])
            return [param.copy() for param in self._global_params]

        trust_scores = []
        scaled_deltas = []
        for params, _ in updates:
            client_delta = self._flatten_delta(params)
            client_norm = float(np.linalg.norm(client_delta))
            cosine = np.dot(server_delta, client_delta) / (
                server_norm * client_norm + 1e-9
            )
            trust_scores.append(max(0.0, float(cosine)))
            if client_norm <= 1e-9:
                scaled_deltas.append(np.zeros_like(client_delta))
            else:
                scaled_deltas.append(client_delta * (server_norm / client_norm))

        weights = np.asarray(trust_scores, dtype=np.float64)
        total_weight = float(weights.sum())
        aggregate_delta = np.zeros_like(server_delta, dtype=np.float32)
        if total_weight > 1e-9:
            for weight, client_delta in zip(weights, scaled_deltas):
                aggregate_delta += float(weight / total_weight) * client_delta
        else:
            logger.warning("FLTrust: all client trust scores are zero; keeping global model.")

        client_ids = self._client_ids_for(len(updates))
        self.last_client_trusts = {
            cid: float(score) for cid, score in zip(client_ids, trust_scores)
        }
        self._record_scalar_weights(client_ids, weights)

        result: List[np.ndarray] = []
        offset = 0
        for global_param in self._global_params:
            if np.issubdtype(global_param.dtype, np.floating):
                size = global_param.size
                delta = aggregate_delta[offset:offset + size].reshape(global_param.shape)
                value = global_param.astype(np.float32) + delta
                result.append(value.astype(global_param.dtype, copy=False))
                offset += size
            else:
                result.append(global_param.copy())
        return result


# ── FoolsGold ─────────────────────────────────────────────────────────────────

class FoolsGoldDefense(BaseDefense):
    """
    FoolsGold (Fung et al. 2018).

    Penalises clients whose update histories are highly similar
    (indicative of Sybil / collusion attacks).

    State: historical contribution matrix (updated each round).
    """

    def __init__(self, cfg: DefenseConfig, num_clients: int = 100):
        super().__init__(cfg)
        self._history: Dict[str, np.ndarray] = {}
        self._num_clients = num_clients

    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
        n = len(updates)
        client_ids = self._client_ids_for(n)
        vectors = np.asarray(
            [self._flatten_delta(params) for params, _ in updates],
            dtype=np.float32,
        )

        for cid, vector in zip(client_ids, vectors):
            previous = self._history.get(cid)
            self._history[cid] = vector.copy() if previous is None else previous + vector

        histories = np.asarray(
            [self._history[cid] for cid in client_ids], dtype=np.float32
        )
        norms = np.linalg.norm(histories, axis=1, keepdims=True) + 1e-9
        normalized = histories / norms
        similarities = np.einsum("ik,jk->ij", normalized, normalized)
        np.fill_diagonal(similarities, 0.0)

        max_similarity = (
            np.max(similarities, axis=1) if n > 1 else np.zeros(1, dtype=np.float32)
        )
        for i in range(n):
            for j in range(n):
                if i == j or max_similarity[j] <= 1e-9:
                    continue
                if max_similarity[i] < max_similarity[j]:
                    similarities[i, j] *= max_similarity[i] / max_similarity[j]

        alphas = (
            1.0 - np.max(similarities, axis=1)
            if n > 1
            else np.ones(1, dtype=np.float32)
        )
        alphas = np.clip(alphas, 0.0, 1.0)
        if float(alphas.max()) > 1e-9:
            alphas /= float(alphas.max())
        logit_input = np.clip(alphas, 1e-6, 1.0 - 1e-6)
        alphas = np.clip(
            np.log(logit_input / (1.0 - logit_input)) + 0.5,
            0.0,
            1.0,
        )

        weights = alphas.astype(np.float64)
        if float(weights.sum()) <= 1e-9:
            weights = np.ones(n, dtype=np.float64)

        self._record_scalar_weights(client_ids, weights)
        return self._weighted_average_with_weights(updates, weights)


# ── Registry & factory ────────────────────────────────────────────────────────

DEFENSE_REGISTRY: Dict[str, Any] = {
    "none":          FedAvgDefense,
    "fedavg":        FedAvgDefense,
    "krum":          KrumDefense,
    "trimmed_mean":  TrimmedMeanDefense,
    "median":        MedianDefense,
    "fltrust":       FLTrustDefense,
    "foolsgold":     FoolsGoldDefense,
    "time_consistency": "TimeConsistencyDefense",
    # ── Extension point ────────────────────────────────────────────────────
    # "your_defense": YourDefenseClass,
}


def get_defense(cfg: DefenseConfig, **kwargs: Any) -> BaseDefense:
    """Instantiate the defense specified in config."""
    key = cfg.type.lower()
    if key not in DEFENSE_REGISTRY:
        raise ValueError(
            f"Unknown defense {key!r}. Available: {list(DEFENSE_REGISTRY)}"
        )
    cls = DEFENSE_REGISTRY[key]
    if cls == "TimeConsistencyDefense":
        from defenses.time_consistency_defense import TimeConsistencyDefense
        cls = TimeConsistencyDefense
    # Pass extra kwargs (e.g. num_clients for FoolsGold)
    try:
        return cls(cfg, **kwargs)
    except TypeError:
        return cls(cfg)
