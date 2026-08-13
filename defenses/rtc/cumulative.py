"""Optional one-sided principal evidence for RTC-v3 Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from defenses.rtc.weighted_stats import weighted_median


@dataclass(frozen=True)
class CumulativeObservation:
    principal_id: str
    resolution: str
    raw_residual: float
    robust_scale: float
    z: float
    statistic: float
    q: float


@dataclass(frozen=True)
class CumulativeTransition:
    observations: Mapping[tuple[str, str], CumulativeObservation]


class OneSidedCumulativeEvidence:
    def __init__(self, config: Mapping[str, object]) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", False))
        self._state: dict[tuple[str, str], float] = {}

    def _resolution_config(self, resolution: str) -> Mapping[str, object]:
        resolutions = self.config.get("resolutions", {})
        if not isinstance(resolutions, Mapping):
            raise ValueError("cumulative.resolutions must be a mapping")
        config = resolutions.get(resolution)
        if not isinstance(config, Mapping):
            raise ValueError(
                f"cumulative calibration missing for resolution {resolution!r}"
            )
        required = {
            "scale_center",
            "scale_lower",
            "scale_upper",
            "kappa",
            "threshold",
            "eta",
            "q_min",
        }
        missing = required.difference(config)
        if missing:
            raise ValueError(
                f"cumulative calibration missing fields for {resolution!r}: "
                f"{sorted(missing)}"
            )
        values = {key: float(config[key]) for key in required}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("cumulative calibration values must be finite")
        if (
            values["scale_center"] <= 0.0
            or values["scale_lower"] <= 0.0
            or values["scale_upper"] < values["scale_lower"]
            or values["kappa"] < 0.0
            or values["threshold"] < 0.0
            or values["eta"] < 0.0
            or not 0.0 < values["q_min"] <= 1.0
        ):
            raise ValueError("invalid cumulative calibration range")
        return config

    def prepare(
        self,
        *,
        principal_ids: Sequence[str],
        nominal_masses: np.ndarray,
        raw_residuals: Mapping[str, np.ndarray],
        force: bool = False,
    ) -> CumulativeTransition:
        if not self.enabled and not force:
            return CumulativeTransition(observations={})
        principals = sorted(set(str(value) for value in principal_ids))
        groups = {
            principal: np.asarray(
                [str(value) == principal for value in principal_ids], dtype=bool
            )
            for principal in principals
        }
        observations: dict[tuple[str, str], CumulativeObservation] = {}
        for resolution, client_residuals in raw_residuals.items():
            config = self._resolution_config(resolution)
            principal_raw: dict[str, float] = {}
            principal_mass: dict[str, float] = {}
            for principal, mask in groups.items():
                mass = float(nominal_masses[mask].sum(dtype=np.float64))
                principal_mass[principal] = mass
                principal_raw[principal] = (
                    float(
                        np.dot(
                            nominal_masses[mask],
                            np.asarray(client_residuals, dtype=np.float64)[mask],
                        )
                    )
                    / (mass + np.finfo(np.float64).eps)
                    if mass > 0.0
                    else 0.0
                )
            eligible = [
                principal for principal in principals if principal_mass[principal] > 0.0
            ]
            if len(eligible) >= 3:
                center = weighted_median(
                    [principal_raw[principal] for principal in eligible],
                    [principal_mass[principal] for principal in eligible],
                    principal_ids=eligible,
                    client_ids=eligible,
                )
            else:
                center = float(config["scale_center"])
            scale = float(
                np.clip(
                    center,
                    float(config["scale_lower"]),
                    float(config["scale_upper"]),
                )
            )
            for principal in principals:
                z = principal_raw[principal] / (
                    scale + np.finfo(np.float64).eps
                )
                key = (principal, resolution)
                statistic = max(
                    0.0,
                    self._state.get(key, 0.0) + z - float(config["kappa"]),
                )
                threshold = float(config["threshold"])
                if statistic <= threshold:
                    q = 1.0
                else:
                    q = max(
                        float(config["q_min"]),
                        float(
                            np.exp(
                                -float(config["eta"])
                                * (statistic - threshold)
                            )
                        ),
                    )
                observations[key] = CumulativeObservation(
                    principal_id=principal,
                    resolution=resolution,
                    raw_residual=principal_raw[principal],
                    robust_scale=scale,
                    z=z,
                    statistic=statistic,
                    q=min(1.0, q),
                )
        return CumulativeTransition(observations=observations)

    def commit(self, transition: CumulativeTransition) -> None:
        for key, observation in transition.observations.items():
            self._state[key] = float(observation.statistic)

    def statistic(self, principal_id: str, resolution: str) -> float:
        return float(self._state.get((str(principal_id), str(resolution)), 0.0))

    def state_dict(self) -> dict[tuple[str, str], float]:
        return dict(self._state)

    def load_state_dict(
        self, state: Mapping[tuple[str, str], float]
    ) -> None:
        checked: dict[tuple[str, str], float] = {}
        for raw_key, raw_value in state.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError("invalid cumulative checkpoint key")
            value = float(raw_value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("invalid cumulative checkpoint value")
            checked[(str(raw_key[0]), str(raw_key[1]))] = value
        self._state = checked
