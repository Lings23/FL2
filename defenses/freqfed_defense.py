"""FreqFed frequency-domain client filtering."""

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
        self.low_frequency_ratio = float(params.get("low_frequency_ratio", 0.2))
        self.min_cluster_size = int(params.get("min_cluster_size", 3))
        self.min_samples = int(params.get("min_samples", 2))
        self.min_selected_clients = int(params.get("min_selected_clients", 2))
        self.cluster_selection_method = str(params.get("cluster_selection_method", "eom"))
        self.allow_single_cluster = bool(params.get("allow_single_cluster", False))
        self.projection_dim = int(params.get("projection_dim", 4096))
        self.fallback = str(params.get("fallback", "median")).lower()
        self.seed = int(params.get("seed", 42))
        if not 0.0 < self.low_frequency_ratio <= 1.0:
            raise ValueError("FreqFed low_frequency_ratio must be in (0, 1]")
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
                count = max(1, int(np.ceil(value.shape[0] * self.low_frequency_ratio)))
                low = spectrum[:count]
            else:
                matrix = value.reshape(-1, value.shape[-1])
                spectrum = dctn(matrix, type=2, norm="ortho")
                rows = max(1, int(np.ceil(matrix.shape[0] * self.low_frequency_ratio)))
                cols = max(1, int(np.ceil(matrix.shape[1] * self.low_frequency_ratio)))
                low = spectrum[:rows, :cols].ravel()
            norm = float(np.linalg.norm(low))
            chunks.append(low / norm if norm > 1e-12 else np.zeros_like(low))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        fingerprint = np.concatenate(chunks).astype(np.float32, copy=False)
        return self._project(fingerprint)

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
        model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="cosine",
            algorithm="generic",
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
        )
        # hdbscan's generic cosine backend expects a double-precision distance
        # matrix on some platforms (notably its Windows wheels).
        labels = model.fit_predict(np.asarray(fingerprints, dtype=np.float64))
        probabilities = getattr(model, "probabilities_", np.ones(len(labels)))
        return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)

    def _select_cluster(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        fingerprints: Sequence[np.ndarray],
        valid_indices: Sequence[int],
    ) -> List[int]:
        candidates = []
        fp_by_index = {idx: fp for idx, fp in zip(valid_indices, fingerprints)}
        for label in sorted({int(value) for value in labels if value >= 0}):
            members = np.flatnonzero(labels == label).tolist()
            matrix = np.asarray([fp_by_index[idx] for idx in members])
            normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            similarity = normalized @ normalized.T
            mean_distance = float(np.mean(1.0 - similarity))
            mean_probability = float(np.mean(probabilities[members]))
            candidates.append((-len(members), -mean_probability, mean_distance, label, members))
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
