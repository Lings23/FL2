"""Round-aware time-consistency defense (RTC-v2).

The public class name remains ``TimeConsistencyDefense`` so existing configs,
strategy code, and experiment scripts can keep using ``type=time_consistency``.
Internally the old multi-scale/FFT/attention design has been replaced by a
smaller, deployable pipeline:

1. build per-client round records from deltas to the previous global model;
2. score magnitude, direction, temporal, and influence risks;
3. update a client state machine and EWMA trust;
4. aggregate with norm clipping, per-client influence caps, and safe fallback.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import numpy as np

from config.config_loader import DefenseConfig
from defenses.defense_base import BaseDefense, UpdateList
from defenses.rtc import ClientHistory, InfluenceAggregator, RiskScorer, RoundRecord, build_round_metrics

logger = logging.getLogger(__name__)


class TimeConsistencyDefense(BaseDefense):
    """RTC-v2 stateful robust aggregation defense."""

    def __init__(self, cfg: DefenseConfig, num_clients: int = 100):
        super().__init__(cfg)
        params = dict(cfg.custom_params or {})
        self.num_clients = int(num_clients)
        self.eps = float(params.get("eps", 1e-9))
        self.projection_dim = int(params.get("projection_dim", 2048))
        self.max_history = int(params.get("max_history", params.get("history_max_history", 30)))
        self.offline_reset_rounds = int(params.get("offline_reset_rounds", 10))
        self.initial_trust = float(params.get("initial_trust", params.get("trust_initial", 1.0)))
        self.log_client_weights = bool(params.get("log_client_weights", True))
        self.params = params

        self.scorer = RiskScorer(params)
        self.aggregator = InfluenceAggregator(params, eps=self.eps)

        self._states: Dict[str, ClientHistory] = {}
        self._projection_indices: np.ndarray | None = None
        self._last_records: List[RoundRecord] = []
        self._last_clipped_mask: List[bool] = []
        self._last_constraint_tags: List[str] = []
        self._last_clip_norm: float = 0.0
        self._last_clipped_clients: int = 0
        self._last_quarantined_clients: int = 0
        self._last_capped_clients: int = 0
        self._fft_periodic_penalty_clients: int = 0
        self._direction_penalty_clients: int = 0

        logger.info(
            "TimeConsistencyDefense RTC-v2 init | projection_dim=%d max_history=%d",
            self.projection_dim,
            self.max_history,
        )

    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
        if not updates:
            return []
        if self._global_params is None:
            logger.warning("TimeConsistencyDefense missing global context; falling back to FedAvg.")
            return self._weighted_average(updates)

        client_ids = self._client_ids_for(len(updates))
        records = self._build_records(updates, client_ids)
        for record in records:
            self._get_state(record.client_id)

        self.scorer.score_records(records, self._states, self._server_round)
        for record in records:
            history = self._states[record.client_id]
            self.scorer.update_state_and_trust(
                record, history, self.params, server_round=self._server_round
            )

        self.aggregator.assign_weights(records)
        aggregated = self.aggregator.aggregate(updates, records, self._global_params)

        self._record_bookkeeping(records)
        for record in records:
            self._states[record.client_id].append(
                server_round=self._server_round,
                norm=record.norm,
                impact=record.impact_proxy,
                signature=record.signature,
                feature=record.feature_vector,
                risk=record.total_risk,
                event_risk=record.event_risk,
                state=record.state,
                dominant_frequency=record.dominant_frequency,
            )
        self._record_round_metrics(records)
        return aggregated

    def _build_records(self, updates: UpdateList, client_ids: Sequence[str]) -> List[RoundRecord]:
        raw_weights = [float(max(0, int(num_examples))) for _, num_examples in updates]
        records: List[RoundRecord] = []
        for idx, ((params, num_examples), cid) in enumerate(zip(updates, client_ids)):
            delta = self._floating_delta(params, self._global_params or [])
            signature = self._signature(delta)
            norm = self._delta_norm_from_params(params)
            records.append(
                RoundRecord(
                    client_id=str(cid),
                    params_index=idx,
                    num_examples=int(num_examples),
                    delta_flat=delta,
                    signature=signature,
                    norm=float(norm),
                    raw_weight=raw_weights[idx] if idx < len(raw_weights) else 1.0,
                )
            )
        return records

    def _get_state(self, cid: str, default_initial_trust: float | None = None) -> ClientHistory:
        init = self.initial_trust if default_initial_trust is None else float(default_initial_trust)
        state = self._states.get(str(cid))
        if state is None:
            state = ClientHistory(client_id=str(cid), max_history=self.max_history, final_trust=init)
            state.last_seen_round = self._server_round
            self._states[str(cid)] = state
            return state
        if state.last_seen_round > 0 and self._server_round - state.last_seen_round > self.offline_reset_rounds:
            state.reset(server_round=self._server_round, initial_trust=init)
        return state

    def _floating_delta(
        self,
        params: Sequence[np.ndarray],
        reference_params: Sequence[np.ndarray],
    ) -> np.ndarray:
        chunks = []
        for param, ref in zip(params, reference_params):
            if not np.issubdtype(ref.dtype, np.floating):
                continue
            chunks.append(param.astype(np.float32).ravel() - ref.astype(np.float32).ravel())
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.nan_to_num(np.concatenate(chunks).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _signature(self, flat_delta: np.ndarray) -> np.ndarray:
        if flat_delta.size == 0:
            return np.zeros(1, dtype=np.float32)
        if self._projection_indices is None or self._projection_indices.size == 0 or self._projection_indices[-1] >= flat_delta.size:
            if flat_delta.size <= self.projection_dim:
                self._projection_indices = np.arange(flat_delta.size, dtype=np.int64)
            else:
                self._projection_indices = np.linspace(0, flat_delta.size - 1, num=self.projection_dim, dtype=np.int64)
        return flat_delta[self._projection_indices].astype(np.float32, copy=False)

    def _delta_norm_from_params(self, params: Sequence[np.ndarray]) -> float:
        assert self._global_params is not None
        total_sq = 0.0
        for param, ref in zip(params, self._global_params):
            if not np.issubdtype(ref.dtype, np.floating):
                continue
            diff = param.astype(np.float32) - ref.astype(np.float32)
            total_sq += float(np.sum(diff * diff))
        return float(np.sqrt(max(0.0, total_sq)))

    # Compatibility helpers used by existing tests and experiments.
    def _delta_norms(self, updates: UpdateList) -> np.ndarray:
        return np.asarray([self._delta_norm_from_params(params) for params, _ in updates], dtype=np.float64)

    def _compute_clip_norm(self, norms: np.ndarray) -> float:
        return self.aggregator.compute_clip_norm(norms)

    def _clip_delta(self, delta: np.ndarray, norm: float, clip_norm: float) -> np.ndarray:
        return self.aggregator.clip_delta(delta, norm, clip_norm)

    def _aggregate_with_effective_weights(self, updates: UpdateList, effective_weights: Sequence[float]) -> List[np.ndarray]:
        if self._global_params is None:
            return self._weighted_average(updates)
        client_ids = self._client_ids_for(len(updates))
        records = self._build_records(updates, client_ids)
        weights = np.asarray(effective_weights, dtype=np.float64)
        total = float(np.sum(np.maximum(weights, 0.0)))
        if total <= self.eps:
            weights = np.ones(len(records), dtype=np.float64) / max(1, len(records))
        else:
            weights = np.maximum(weights, 0.0) / total
        for record, weight in zip(records, weights):
            record.effective_weight = float(weight)
            record.aggregation_weight = float(weight)
        result = self.aggregator.aggregate(updates, records, self._global_params)
        self._record_bookkeeping(records)
        return result

    def _apply_trust_weight_constraints(
        self,
        trust_scores: Sequence[float],
        effective_weights: Sequence[float],
    ) -> List[float]:
        records = [
            RoundRecord(
                client_id=str(idx),
                params_index=idx,
                num_examples=1,
                delta_flat=np.zeros(1, dtype=np.float32),
                signature=np.zeros(1, dtype=np.float32),
                norm=0.0,
                raw_weight=float(weight),
                trust=float(trust),
            )
            for idx, (trust, weight) in enumerate(zip(trust_scores, effective_weights))
        ]
        quarantine_threshold = float(self.params.get("quarantine_threshold", 0.25))
        restricted_threshold = float(self.params.get("low_trust_threshold", self.params.get("restricted_threshold", 0.35)))
        low_trust_weight_cap = float(self.params.get("low_trust_weight_cap", 0.01))
        weights = np.maximum(np.asarray(effective_weights, dtype=np.float64), 0.0)
        total = float(weights.sum())
        if total <= self.eps:
            return weights.tolist()
        normalized = weights / total
        constrained = normalized.copy()
        tags = ["" for _ in records]
        fixed = np.zeros_like(constrained, dtype=bool)
        for idx, trust in enumerate(trust_scores):
            if trust < quarantine_threshold:
                constrained[idx] = 0.0
                tags[idx] = "quarantined"
                fixed[idx] = True
            elif trust < restricted_threshold:
                constrained[idx] = min(constrained[idx], low_trust_weight_cap)
                tags[idx] = "capped"
                fixed[idx] = True
        if float(constrained.sum()) <= self.eps:
            # RTC-v2's safe fallback is "no trusted mass"; do not restore
            # original FedAvg weights in this compatibility helper.
            constrained = np.zeros_like(constrained)
        elif fixed.any() and (~fixed).any():
            fixed_sum = float(constrained[fixed].sum())
            remaining = max(0.0, 1.0 - fixed_sum)
            free_base = normalized[~fixed]
            free_sum = float(free_base.sum())
            if free_sum > self.eps:
                constrained[~fixed] = free_base / free_sum * remaining
            else:
                constrained[~fixed] = remaining / int((~fixed).sum())
        else:
            constrained = constrained / float(constrained.sum())
        self._last_constraint_tags = tags
        self._last_quarantined_clients = int(sum(tag == "quarantined" for tag in tags))
        self._last_capped_clients = int(sum(tag == "capped" for tag in tags))
        return (constrained * total).tolist()

    def _frequency_features(self, values: List[float]) -> tuple[float, float, float, float]:
        """Compatibility-only FFT summary; RTC-v2 scoring uses true round gaps."""
        if len(values) < 3:
            return 0.0, 0.0, 0.0, 0.0
        window = np.asarray(values, dtype=np.float32)
        centered = window - float(np.mean(window))
        spectrum = np.fft.rfft(centered)
        psd = np.abs(spectrum) ** 2
        if psd.shape[0] <= 1:
            return 0.0, 0.0, 0.0, 0.0
        psd = psd[1:]
        total = float(psd.sum())
        if total <= self.eps:
            return 0.0, 0.0, 0.0, 0.0
        prob = psd / (total + self.eps)
        entropy = -float(np.sum(prob * np.log(prob + self.eps))) / np.log(len(prob) + self.eps)
        dominant_idx = int(np.argmax(prob))
        dominant_energy = float(prob[dominant_idx])
        low_cut = max(1, int(np.ceil(len(prob) * 0.25)))
        low_freq_power = float(prob[:low_cut].sum())
        dominant_frequency = float((dominant_idx + 1) / max(1, len(window)))
        return float(1.0 - entropy), dominant_energy, low_freq_power, dominant_frequency

    def _record_bookkeeping(self, records: Sequence[RoundRecord]) -> None:
        self._last_records = list(records)
        self._last_clipped_mask = [bool(r.clipped) for r in records]
        self._last_constraint_tags = [
            "quarantined" if r.quarantined else "capped" if r.capped else ""
            for r in records
        ]
        self._last_clip_norm = self.aggregator._last_clip_norm
        self._last_clipped_clients = int(sum(r.clipped for r in records))
        self._last_quarantined_clients = int(sum(r.quarantined for r in records))
        self._last_capped_clients = int(sum(r.capped for r in records))
        self._fft_periodic_penalty_clients = int(sum(r.temporal_risk >= 0.7 for r in records))
        self._direction_penalty_clients = int(sum(r.direction_risk >= 0.7 for r in records))

    def _record_round_metrics(self, records: Sequence[RoundRecord]) -> None:
        self.last_client_trusts = {r.client_id: float(r.trust) for r in records}
        self.last_client_weights = {r.client_id: float(r.effective_weight) for r in records}
        self.last_client_aggregation_weights = {r.client_id: float(r.aggregation_weight) for r in records}
        self.last_round_metrics = build_round_metrics(
            records,
            clip_norm=self._last_clip_norm,
            fallback_used=self.aggregator._last_fallback_used,
            fallback_reason=self.aggregator._last_fallback_reason,
        )
        if self.log_client_weights:
            self._log_client_weights(records)

    def _log_client_weights(self, records: Sequence[RoundRecord]) -> None:
        rows = sorted(records, key=lambda r: r.aggregation_weight)
        details = []
        for record in rows:
            flags = []
            if record.quarantined:
                flags.append("quarantined")
            elif record.capped:
                flags.append("capped")
            if record.clipped:
                flags.append("clipped")
            if record.fallback_used:
                flags.append("fallback")
            suffix = f" flags={','.join(flags)}" if flags else ""
            details.append(
                f"cid={record.client_id} trust={record.trust:.4f} "
                f"risk={record.total_risk:.4f} state={record.state} "
                f"effective_weight={record.effective_weight:.4f} "
                f"aggregation_weight={record.aggregation_weight:.4f}{suffix}"
            )
        logger.info("TimeConsistency round %d client weights | %s", self._server_round, "; ".join(details))
