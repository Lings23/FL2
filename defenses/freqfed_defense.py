"""FreqFed frequency-domain client filtering.

This implementation follows the FreqFed paper's main path as closely as the
project interface allows: build a low-frequency DCT fingerprint for each client,
compute pairwise cosine distances, cluster clients with HDBSCAN, and aggregate
only the largest non-noise cluster.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
from scipy.fft import dct, dctn

from config.config_loader import DefenseConfig
from defenses.defense_base import BaseDefense, UpdateList

logger = logging.getLogger(__name__)


class FreqFedDefense(BaseDefense):
    """Cluster low-frequency DCT fingerprints and aggregate the majority cluster."""

    def __init__(self, cfg: DefenseConfig, **_: object):
        super().__init__(cfg)
        params = cfg.custom_params or {}
        self.low_frequency_ratio = float(params.get("low_frequency_ratio", 0.5))
        self.low_frequency_shape = str(params.get("low_frequency_shape", "triangular")).lower()
        self.min_cluster_size = int(params.get("min_cluster_size", 3))
        self.min_samples = int(params.get("min_samples", 2))
        self.min_selected_clients = int(params.get("min_selected_clients", 2))
        self.cluster_selection_method = str(params.get("cluster_selection_method", "eom"))
        self.allow_single_cluster = bool(params.get("allow_single_cluster", False))
        self.projection_dim = int(params.get("projection_dim", 0))
        self.fallback = str(params.get("fallback", "median")).lower()
        self.seed = int(params.get("seed", 42))
        if not 0.0 < self.low_frequency_ratio <= 1.0:
            raise ValueError("FreqFed low_frequency_ratio must be in (0, 1]")
        if self.low_frequency_shape not in {"triangular", "rectangle"}:
            raise ValueError("FreqFed low_frequency_shape must be triangular or rectangle")
        if self.min_cluster_size < 2 or self.min_samples < 1:
            raise ValueError("FreqFed clustering sizes must be positive (cluster size >= 2)")
        if self.fallback not in {"median", "fedavg", "keep_global"}:
            raise ValueError("FreqFed fallback must be median, fedavg, or keep_global")

    def aggregate(self, updates: UpdateList):
        if not updates:
            raise ValueError("FreqFed requires at least one client update")
        client_ids = self._client_ids_for(len(updates))
        valid_indices: List[int] = []
        fingerprints: List[np.ndarray] = []
        for idx, (params, _) in enumerate(updates):
            fingerprint = self._frequency_fingerprint(params)
            if fingerprint.size and np.all(np.isfinite(fingerprint)):
                valid_indices.append(idx)
                fingerprints.append(fingerprint)

        labels = np.full(len(updates), -1, dtype=np.int64)
        probabilities = np.zeros(len(updates), dtype=np.float64)
        selected: List[int] = []
        fallback_used = len(valid_indices) < self.min_cluster_size
        fallback_reason = "insufficient_valid_fingerprints" if fallback_used else ""
        if not fallback_used:
            try:
                local_labels, local_probabilities = self._cluster(np.asarray(fingerprints))
                labels[valid_indices] = local_labels
                probabilities[valid_indices] = local_probabilities
                selected = self._select_cluster(labels, probabilities, fingerprints, valid_indices)
                fallback_used = len(selected) < self.min_selected_clients
                if fallback_used:
                    fallback_reason = "selected_cluster_too_small"
            except (ImportError, ValueError) as exc:
                logger.warning("FreqFed clustering unavailable/invalid; using %s fallback: %s", self.fallback, exc)
                fallback_used = True
                fallback_reason = f"clustering_error:{type(exc).__name__}"

        if fallback_used:
            result = self._fallback_aggregate(updates)
            if self.fallback == "fedavg":
                raw_weights = np.asarray([num for _, num in updates], dtype=np.float64)
                self._record_scalar_weights(client_ids, raw_weights)
            else:
                # Coordinate-wise median and keep-global do not have meaningful
                # scalar per-client influence weights. Leaving these maps empty
                # prevents downstream reports from claiming all attackers had
                # zero influence.
                raw_weights = np.zeros(len(updates), dtype=np.float64)
                self.last_client_weights = {}
                self.last_client_aggregation_weights = {}
            self.last_client_trusts = {}
        else:
            raw_weights = np.asarray(
                [num if idx in selected else 0.0 for idx, (_, num) in enumerate(updates)],
                dtype=np.float64,
            )
            result = self._weighted_average_with_weights(updates, raw_weights)
            self.last_client_trusts = {
                cid: float(probabilities[idx]) if idx in selected else 0.0
                for idx, cid in enumerate(client_ids)
            }
            self._record_scalar_weights(client_ids, raw_weights)

        cluster_labels = {int(label) for label in labels if label >= 0}
        self.last_round_metrics = {
            "freqfed_selected_clients": float(len(selected) if not fallback_used else 0),
            "freqfed_rejected_clients": float(len(updates) - len(selected) if not fallback_used else 0),
            "freqfed_noise_clients": float(np.count_nonzero(labels < 0)),
            "freqfed_cluster_count": float(len(cluster_labels)),
            "freqfed_selected_ratio": float(len(selected) / len(updates)) if not fallback_used else 0.0,
            "freqfed_fallback": float(fallback_used),
            "aggregation_mode": (
                f"{self.fallback}_fallback" if fallback_used else "cluster"
            ),
            "freqfed_fallback_reason": fallback_reason,
            "freqfed_low_frequency_ratio": float(self.low_frequency_ratio),
            "freqfed_low_frequency_shape": self.low_frequency_shape,
            "freqfed_distance_metric": "precomputed_cosine",
        }
        return result

    def _frequency_fingerprint(self, params: Sequence[np.ndarray]) -> np.ndarray:
        chunks = []
        for param in params:
            if not np.issubdtype(param.dtype, np.floating) or param.ndim == 0:
                continue
            value = param.astype(np.float32, copy=False)
            if value.ndim == 1:
                spectrum = dct(value, type=2, norm="ortho")
                low = self._low_frequency_1d(spectrum)
            else:
                matrix = value.reshape(-1, value.shape[-1])
                spectrum = dctn(matrix, type=2, norm="ortho")
                low = self._low_frequency_2d(spectrum)
            norm = float(np.linalg.norm(low))
            chunks.append(low / norm if norm > 1e-12 else np.zeros_like(low))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        fingerprint = np.concatenate(chunks).astype(np.float32, copy=False)
        return self._project(fingerprint)

    def _low_frequency_1d(self, spectrum: np.ndarray) -> np.ndarray:
        count = max(1, int(np.ceil(spectrum.shape[0] * self.low_frequency_ratio)))
        return np.asarray(spectrum[:count], dtype=np.float32)

    def _low_frequency_2d(self, spectrum: np.ndarray) -> np.ndarray:
        rows, cols = spectrum.shape
        if self.low_frequency_shape == "rectangle":
            row_count = max(1, int(np.ceil(rows * self.low_frequency_ratio)))
            col_count = max(1, int(np.ceil(cols * self.low_frequency_ratio)))
            return np.asarray(spectrum[:row_count, :col_count].ravel(), dtype=np.float32)

        # FreqFed's paper extracts a low-frequency triangular region from the
        # DCT coefficient matrix (i + j below a cutoff), rather than a simple
        # rectangular crop.  The ratio makes that cutoff configurable while the
        # default path stays triangular.
        cutoff = max(0, int(np.floor(max(rows, cols) * self.low_frequency_ratio)))
        row_idx, col_idx = np.indices((rows, cols))
        mask = (row_idx + col_idx) <= cutoff
        return np.asarray(spectrum[mask], dtype=np.float32)

    def _project(self, vector: np.ndarray) -> np.ndarray:
        if self.projection_dim <= 0 or vector.size <= self.projection_dim:
            return vector
        rng = np.random.default_rng(self.seed)
        buckets = rng.integers(0, self.projection_dim, size=vector.size)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=vector.size)
        projected = np.zeros(self.projection_dim, dtype=np.float32)
        np.add.at(projected, buckets, vector * signs)
        return projected / np.sqrt(self.projection_dim)

    def _cluster(self, fingerprints: np.ndarray):
        try:
            import hdbscan
        except ImportError as exc:
            raise ImportError("install the 'hdbscan' package to enable FreqFed") from exc
        distances = self._cosine_distance_matrix(fingerprints)
        model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="precomputed",
            algorithm="generic",
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
        )
        labels = model.fit_predict(distances)
        probabilities = getattr(model, "probabilities_", np.ones(len(labels)))
        return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)

    def _cosine_distance_matrix(self, fingerprints: np.ndarray) -> np.ndarray:
        if fingerprints.ndim != 2:
            raise ValueError("FreqFed fingerprints must be a 2D matrix")
        normalized = fingerprints.astype(np.float64, copy=False)
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        normalized = normalized / np.maximum(norms, 1e-12)
        distances = 1.0 - (normalized @ normalized.T)
        distances = np.clip(distances, 0.0, 2.0)
        np.fill_diagonal(distances, 0.0)
        return distances.astype(np.float64, copy=False)

    def _select_cluster(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        fingerprints: Sequence[np.ndarray],
        valid_indices: Sequence[int],
    ) -> List[int]:
        candidates = []
        for label in sorted({int(value) for value in labels if value >= 0}):
            members = np.flatnonzero(labels == label).tolist()
            mean_probability = float(np.mean(probabilities[members]))
            candidates.append((-len(members), label, -mean_probability, members))
        return min(candidates)[-1] if candidates else []

    def _fallback_aggregate(self, updates: UpdateList):
        if self.fallback == "fedavg":
            return self._weighted_average(updates)
        if self.fallback == "keep_global":
            if self._global_params is None:
                return self._weighted_average(updates)
            return [param.copy() for param in self._global_params]
        vectors = np.asarray([self._flatten(params) for params, _ in updates])
        return self._unflatten(np.median(vectors, axis=0), updates[0][0])
