"""State containers for round-aware time-consistency aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ClientHistory:
    """Per-client RTC-v2 history keyed by server-side client identity."""

    client_id: str
    max_history: int
    rounds: List[int] = field(default_factory=list)
    norm_history: List[float] = field(default_factory=list)
    impact_history: List[float] = field(default_factory=list)
    effective_influence_history: List[float] = field(default_factory=list)
    signature_history: List[np.ndarray] = field(default_factory=list)
    risk_history: List[float] = field(default_factory=list)
    event_risk_history: List[float] = field(default_factory=list)
    feature_history: List[np.ndarray] = field(default_factory=list)
    state_history: List[str] = field(default_factory=list)
    participation_count: int = 0
    last_seen_round: int = 0
    final_trust: float = 1.0
    state: str = "normal"
    cooldown_until: int = 0
    low_risk_streak: int = 0
    dominant_freq_history: List[float] = field(default_factory=list)

    def append(
        self,
        *,
        server_round: int,
        norm: float,
        impact: float,
        effective_influence: float,
        signature: np.ndarray,
        feature: np.ndarray,
        risk: float,
        event_risk: float,
        state: str,
        dominant_frequency: float,
    ) -> None:
        self.participation_count += 1
        self.last_seen_round = int(server_round)
        self.rounds.append(int(server_round))
        self.norm_history.append(float(norm))
        self.impact_history.append(float(impact))
        self.effective_influence_history.append(float(effective_influence))
        self.signature_history.append(signature.astype(np.float32, copy=True))
        self.feature_history.append(feature.astype(np.float32, copy=True))
        self.risk_history.append(float(risk))
        self.event_risk_history.append(float(event_risk))
        self.state_history.append(str(state))
        self.dominant_freq_history.append(float(dominant_frequency))
        self._trim()

    def reset(self, *, server_round: int, initial_trust: float) -> None:
        self.rounds.clear()
        self.norm_history.clear()
        self.impact_history.clear()
        self.effective_influence_history.clear()
        self.signature_history.clear()
        self.risk_history.clear()
        self.event_risk_history.clear()
        self.feature_history.clear()
        self.state_history.clear()
        self.dominant_freq_history.clear()
        self.participation_count = 0
        self.last_seen_round = int(server_round)
        self.final_trust = float(initial_trust)
        self.state = "normal"
        self.cooldown_until = 0
        self.low_risk_streak = 0

    def _trim(self) -> None:
        overflow = len(self.rounds) - self.max_history
        if overflow <= 0:
            return
        del self.rounds[:overflow]
        del self.norm_history[:overflow]
        del self.impact_history[:overflow]
        del self.effective_influence_history[:overflow]
        del self.signature_history[:overflow]
        del self.risk_history[:overflow]
        del self.event_risk_history[:overflow]
        del self.feature_history[:overflow]
        del self.state_history[:overflow]
        del self.dominant_freq_history[:overflow]


@dataclass
class RoundRecord:
    """One selected client's current-round update and defense decisions."""

    client_id: str
    params_index: int
    num_examples: int
    delta_flat: np.ndarray
    signature: np.ndarray
    norm: float
    raw_weight: float
    raw_weight_share: float = 0.0
    norm_share: float = 0.0
    impact_proxy: float = 0.0
    magnitude_risk: float = 0.0
    direction_risk: float = 0.0
    temporal_risk: float = 0.0
    influence_risk: float = 0.0
    influence_attempt: float = 0.0
    influence_effective: float = 0.0
    normalized_exposure: float = 0.0
    total_risk: float = 0.0
    event_risk: float = 0.0
    trust: float = 1.0
    state: str = "normal"
    effective_weight: float = 0.0
    aggregation_weight: float = 0.0
    weight_cap: float = 0.0
    zero_mass: float = 0.0
    client_budget_remaining: float = float("inf")
    direction_budget_remaining: float = float("inf")
    direction_group: str = ""
    clipped: bool = False
    capped: bool = False
    quarantined: bool = False
    fallback_used: bool = False
    dominant_frequency: float = 0.0

    @property
    def feature_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.direction_risk,
                self.magnitude_risk,
                self.influence_risk,
                self.total_risk,
                self.temporal_risk,
                self.dominant_frequency,
                self.norm_share,
            ],
            dtype=np.float32,
        )
