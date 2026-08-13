"""Real-server-round temporal state for RTC-v3 semantic evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PairState:
    statistic: float = 0.0
    last_observed_round: int = 0
    low_evidence_streak: int = 0
    observation_count: int = 0


@dataclass(frozen=True)
class SemanticTemporalTransition:
    next_states: Mapping[tuple[str, int], PairState]
    client_risks: np.ndarray
    client_q: np.ndarray
    synchronized_pairs: tuple[int, ...]


class SemanticTemporalEvidence:
    """Maintain every principal/class-pair state using elapsed server rounds."""

    def __init__(self, config: Mapping[str, Any], pair_count: int):
        self.config = dict(config)
        self.pair_count = int(pair_count)
        if self.pair_count <= 0:
            raise ValueError("semantic temporal evidence requires ordered pairs")
        self.decay_rate = float(self.config.get("decay_rate", 0.0))
        self.kappa = float(self.config.get("kappa", 1.0))
        self.threshold = float(self.config.get("threshold", 1.0))
        self.eta = float(self.config.get("eta", 1.0))
        self.q_min = float(self.config.get("q_min", 0.5))
        self.recovery_threshold = float(
            self.config.get("recovery_threshold", 0.25)
        )
        self.recovery_observations = int(
            self.config.get("recovery_observations", 3)
        )
        self.recovery_factor = float(self.config.get("recovery_factor", 0.5))
        self.min_history_observations = int(
            self.config.get("min_history_observations", 1)
        )
        if (
            self.decay_rate < 0.0
            or self.kappa < 0.0
            or self.threshold < 0.0
            or self.eta <= 0.0
            or not 0.0 < self.q_min < 1.0
            or self.recovery_threshold < 0.0
            or self.recovery_observations <= 0
            or not 0.0 <= self.recovery_factor <= 1.0
            or self.min_history_observations <= 0
        ):
            raise ValueError("invalid semantic temporal configuration")
        sync = self.config.get("synchronization", {})
        if not isinstance(sync, Mapping):
            raise ValueError("semantic synchronization config must be a mapping")
        self.sync_enabled = bool(sync.get("enabled", True))
        self.sync_min_principals = int(sync.get("min_distinct_principals", 2))
        self.sync_evidence_threshold = float(sync.get("evidence_threshold", 1.0))
        self.sync_similarity_threshold = float(sync.get("similarity_threshold", 0.9))
        self.sync_multiplier = float(sync.get("multiplier", 1.0))
        if (
            self.sync_min_principals < 2
            or self.sync_evidence_threshold < 0.0
            or not -1.0 <= self.sync_similarity_threshold <= 1.0
            or self.sync_multiplier < 1.0
        ):
            raise ValueError("invalid semantic synchronization configuration")
        self._states: dict[tuple[str, int], PairState] = {}

    def prepare(
        self,
        *,
        server_round: int,
        principal_ids: Sequence[str],
        z_values: np.ndarray,
        top_pair_indices: Sequence[int],
        top_pair_signatures: Sequence[np.ndarray],
        client_ids: Sequence[str] | None = None,
    ) -> SemanticTemporalTransition:
        round_value = int(server_round)
        if round_value <= 0:
            raise ValueError("semantic time requires positive server rounds")
        z = np.asarray(z_values, dtype=np.float64)
        if z.shape != (len(principal_ids), self.pair_count):
            raise ValueError("semantic z_values shape does not match clients and pairs")
        if not np.all(np.isfinite(z)) or np.any(z < 0.0):
            raise ValueError("semantic z_values must be finite and non-negative")

        principals = tuple(str(value) for value in principal_ids)
        clients = (
            tuple(str(value) for value in client_ids)
            if client_ids is not None
            else tuple(str(index) for index in range(len(principals)))
        )
        if len(clients) != len(principals):
            raise ValueError("semantic client_ids must align with principal_ids")
        unique_principals = tuple(sorted(set(principals)))
        principal_z = {
            principal: np.max(z[np.asarray([p == principal for p in principals])], axis=0)
            for principal in unique_principals
        }
        synchronized: set[int] = set()
        if self.sync_enabled and self.sync_multiplier > 1.0:
            groups: dict[int, list[int]] = {}
            for index, pair_index in enumerate(top_pair_indices):
                if z[index, int(pair_index)] >= self.sync_evidence_threshold:
                    groups.setdefault(int(pair_index), []).append(index)
            for pair_index, indices in groups.items():
                representatives: dict[str, int] = {}
                for index in indices:
                    principal = principals[index]
                    previous = representatives.get(principal)
                    if previous is None or (
                        float(z[index, pair_index]), clients[index]
                    ) > (
                        float(z[previous, pair_index]), clients[previous]
                    ):
                        representatives[principal] = index
                selected = list(representatives.values())
                if len(selected) < self.sync_min_principals:
                    continue
                signatures = np.stack(
                    [np.asarray(top_pair_signatures[index], dtype=np.float64) for index in selected]
                )
                similarities = signatures @ signatures.T
                off_diagonal = similarities[np.triu_indices(len(selected), k=1)]
                if off_diagonal.size and float(np.min(off_diagonal)) >= self.sync_similarity_threshold:
                    synchronized.add(pair_index)
                    for principal in representatives:
                        principal_z[principal][pair_index] *= self.sync_multiplier

        next_states = dict(self._states)
        principal_risk: dict[str, float] = {}
        for principal in unique_principals:
            max_risk = 0.0
            for pair_index, evidence in enumerate(principal_z[principal]):
                key = (principal, pair_index)
                previous = self._states.get(key, PairState())
                if previous.last_observed_round > round_value:
                    raise ValueError("semantic checkpoint is ahead of the server round")
                elapsed = (
                    round_value - previous.last_observed_round
                    if previous.last_observed_round > 0
                    else 0
                )
                decayed = previous.statistic * math.exp(-self.decay_rate * elapsed)
                statistic = max(0.0, decayed + float(evidence) - self.kappa)
                low_streak = (
                    previous.low_evidence_streak + 1
                    if float(evidence) <= self.recovery_threshold
                    else 0
                )
                if low_streak >= self.recovery_observations:
                    statistic *= self.recovery_factor
                    low_streak = 0
                state = PairState(
                    statistic=statistic,
                    last_observed_round=round_value,
                    low_evidence_streak=low_streak,
                    observation_count=previous.observation_count + 1,
                )
                next_states[key] = state
                excess = max(0.0, statistic - self.threshold)
                risk = (
                    1.0 - math.exp(-self.eta * excess)
                    if state.observation_count >= self.min_history_observations
                    else 0.0
                )
                max_risk = max(max_risk, risk)
            principal_risk[principal] = max_risk
        client_risks = np.asarray(
            [principal_risk[principal] for principal in principals], dtype=np.float64
        )
        client_q = np.maximum(self.q_min, 1.0 - client_risks)
        return SemanticTemporalTransition(
            next_states=next_states,
            client_risks=client_risks,
            client_q=client_q,
            synchronized_pairs=tuple(sorted(synchronized)),
        )

    def commit(self, transition: SemanticTemporalTransition) -> None:
        self._states = dict(transition.next_states)

    def state_dict(self) -> dict[str, Any]:
        return {
            f"{principal}\u001f{pair_index}": {
                "statistic": state.statistic,
                "last_observed_round": state.last_observed_round,
                "low_evidence_streak": state.low_evidence_streak,
                "observation_count": state.observation_count,
            }
            for (principal, pair_index), state in sorted(self._states.items())
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        states: dict[tuple[str, int], PairState] = {}
        for raw_key, raw_state in payload.items():
            principal, separator, pair_text = str(raw_key).partition("\u001f")
            if not separator or not principal or not isinstance(raw_state, Mapping):
                raise ValueError("invalid semantic temporal checkpoint entry")
            pair_index = int(pair_text)
            if not 0 <= pair_index < self.pair_count:
                raise ValueError("semantic checkpoint has an unknown class pair")
            state = PairState(
                statistic=float(raw_state["statistic"]),
                last_observed_round=int(raw_state["last_observed_round"]),
                low_evidence_streak=int(raw_state["low_evidence_streak"]),
                observation_count=int(raw_state["observation_count"]),
            )
            if (
                not np.isfinite(state.statistic)
                or state.statistic < 0.0
                or state.last_observed_round <= 0
                or state.low_evidence_streak < 0
                or state.observation_count <= 0
            ):
                raise ValueError("invalid semantic temporal checkpoint state")
            states[(principal, pair_index)] = state
        self._states = states
