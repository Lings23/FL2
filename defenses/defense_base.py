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
from typing import Any, Dict, List, Optional, Tuple, Type

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

    @abstractmethod
    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
        ...

    # ── Utility helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _flatten(params: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([p.ravel() for p in params])

    @staticmethod
    def _unflatten(flat: np.ndarray, template: List[np.ndarray]) -> List[np.ndarray]:
        result, offset = [], 0
        for p in template:
            n = p.size
            result.append(flat[offset:offset + n].reshape(p.shape))
            offset += n
        return result

    @staticmethod
    def _weighted_average(updates: UpdateList) -> List[np.ndarray]:
        """Standard weighted average (FedAvg numerics)."""
        total = sum(n for _, n in updates)
        agg = [np.zeros_like(p) for p in updates[0][0]]
        for params, n in updates:
            w = n / total
            for i, p in enumerate(params):
                agg[i] += w * p
        return agg


# ── Standard FedAvg (no defense) ─────────────────────────────────────────────

class FedAvgDefense(BaseDefense):
    """Plain weighted average — reference baseline."""

    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
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
        m = min(self.cfg.krum_num_to_select, n)
        f = max(0, n - m - 2)                 # assumed Byzantine fraction

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

        selected = [updates[i] for i in selected_idx]
        return self._weighted_average(selected)


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
        k = max(1, int(n * beta))

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
        if self._server_update is None:
            logger.warning("FLTrust: no server update set; falling back to FedAvg.")
            return self._weighted_average(updates)

        sv = self._flatten(self._server_update)
        sv_norm = np.linalg.norm(sv)
        if sv_norm < 1e-9:
            return self._weighted_average(updates)

        weights = []
        for params, _ in updates:
            cv = self._flatten(params)
            cos_sim = np.dot(sv, cv) / (sv_norm * np.linalg.norm(cv) + 1e-9)
            trust_score = max(0.0, cos_sim)    # ReLU
            weights.append(trust_score)

        total_w = sum(weights)
        if total_w < 1e-9:
            logger.warning("FLTrust: all trust scores ≈ 0; equal weighting.")
            return self._weighted_average(updates)

        agg_flat = np.zeros_like(sv)
        for i, (params, _) in enumerate(updates):
            cv = self._flatten(params)
            # Project onto server direction and scale by trust weight
            cv_proj = (np.dot(sv, cv) / (sv_norm ** 2)) * sv
            agg_flat += (weights[i] / total_w) * cv_proj

        return self._unflatten(agg_flat, updates[0][0])


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
        self._history: Optional[np.ndarray] = None
        self._num_clients = num_clients
        self._client_idx_map: Dict[int, int] = {}   # cid → position
        self._counter = 0

    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
        n = len(updates)
        dim = self._flatten(updates[0][0]).shape[0]

        # Lazily init history matrix
        if self._history is None:
            self._history = np.zeros((self._num_clients, dim), dtype=np.float32)

        # Map update positions to client slots
        clients = list(range(n))  # positional IDs for this round
        vectors = np.array([self._flatten(p) for p, _ in updates], dtype=np.float32)

        # Update history
        for i, v in enumerate(vectors):
            slot = i % self._num_clients
            self._history[slot] += v

        # Cosine similarity between histories
        norms = np.linalg.norm(self._history[:n], axis=1, keepdims=True) + 1e-9
        normed = self._history[:n] / norms
        cs_matrix = normed @ normed.T           # n × n cosine similarities

        # Learning rate (contribution weight) penalisation
        alphas = np.ones(n)
        for i in range(n):
            for j in range(i):
                sim = cs_matrix[i, j]
                if sim > 0.5:
                    # Penalise the one with higher norm (larger contributor)
                    if np.linalg.norm(vectors[i]) > np.linalg.norm(vectors[j]):
                        alphas[i] = min(alphas[i], 1 - sim)
                    else:
                        alphas[j] = min(alphas[j], 1 - sim)

        # Normalise and aggregate
        alphas = np.clip(alphas, 0, 1)
        total_w = alphas.sum()
        if total_w < 1e-9:
            return self._weighted_average(updates)

        agg_flat = np.zeros(dim, dtype=np.float32)
        for i, (v, _) in enumerate(zip(vectors, updates)):
            agg_flat += (alphas[i] / total_w) * v

        return self._unflatten(agg_flat, updates[0][0])


# ── Registry & factory ────────────────────────────────────────────────────────

DEFENSE_REGISTRY: Dict[str, Type[BaseDefense]] = {
    "none":          FedAvgDefense,
    "fedavg":        FedAvgDefense,
    "krum":          KrumDefense,
    "trimmed_mean":  TrimmedMeanDefense,
    "median":        MedianDefense,
    "fltrust":       FLTrustDefense,
    "foolsgold":     FoolsGoldDefense,
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
    # Pass extra kwargs (e.g. num_clients for FoolsGold)
    try:
        return cls(cfg, **kwargs)
    except TypeError:
        return cls(cfg)
