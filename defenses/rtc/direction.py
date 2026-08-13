"""Optional joint direction-persistence evidence for RTC-v3 Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DirectionObservation:
    principal_id: str
    resolution: str
    cone_id: str
    repeat_count: int
    z: float


@dataclass(frozen=True)
class DirectionTransition:
    observations: Mapping[tuple[str, str], DirectionObservation]
    next_history: Mapping[tuple[str, str], tuple[str, ...]]


class DirectionPersistenceEvidence:
    def __init__(self, config: Mapping[str, object]) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", False))
        self._history: dict[tuple[str, str], tuple[str, ...]] = {}

    def _resolution_config(self, resolution: str) -> Mapping[str, object]:
        configs = self.config.get("resolutions", {})
        if not isinstance(configs, Mapping):
            raise ValueError("direction.resolutions must be a mapping")
        config = configs.get(resolution)
        if not isinstance(config, Mapping):
            raise ValueError(
                f"direction calibration missing for resolution {resolution!r}"
            )
        required = {
            "z_threshold",
            "history_length",
            "min_matches",
            "server_window",
            "principal_window",
            "server_usage_threshold",
            "principal_usage_threshold",
            "q_value",
        }
        missing = required.difference(config)
        if missing:
            raise ValueError(
                f"direction calibration missing fields for {resolution!r}: "
                f"{sorted(missing)}"
            )
        history_length = int(config["history_length"])
        min_matches = int(config["min_matches"])
        if history_length <= 0 or not 1 <= min_matches <= history_length:
            raise ValueError("direction history/min_matches are invalid")
        q_value = float(config["q_value"])
        if not 0.0 < q_value <= 1.0:
            raise ValueError("direction q_value must be in (0, 1]")
        for key in (
            "z_threshold",
            "server_usage_threshold",
            "principal_usage_threshold",
        ):
            value = float(config[key])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"direction {key} must be finite and non-negative")
        if int(config["server_window"]) <= 0 or int(config["principal_window"]) <= 0:
            raise ValueError("direction windows must be positive")
        return config

    def prepare(
        self,
        *,
        principal_ids: Sequence[str],
        nominal_masses: np.ndarray,
        cone_assignments: Mapping[str, Sequence[str]],
        z_values: Mapping[tuple[str, str], float],
    ) -> DirectionTransition:
        if not self.enabled:
            return DirectionTransition(observations={}, next_history=dict(self._history))
        principals = sorted(set(str(value) for value in principal_ids))
        observations: dict[tuple[str, str], DirectionObservation] = {}
        next_history = dict(self._history)
        for resolution, assignments in cone_assignments.items():
            config = self._resolution_config(resolution)
            length = int(config["history_length"])
            for principal in principals:
                indices = [
                    index
                    for index, value in enumerate(principal_ids)
                    if str(value) == principal
                ]
                cone_masses: dict[str, float] = {}
                for index in indices:
                    cone_id = str(assignments[index])
                    if cone_id == "overflow":
                        continue
                    cone_masses[cone_id] = cone_masses.get(cone_id, 0.0) + float(
                        nominal_masses[index]
                    )
                if cone_masses:
                    cone_id = min(
                        cone_masses,
                        key=lambda candidate: (-cone_masses[candidate], candidate),
                    )
                else:
                    cone_id = "overflow"
                key = (principal, resolution)
                previous = self._history.get(key, ())
                updated = (*previous, cone_id)[-length:]
                next_history[key] = updated
                repeat_count = (
                    sum(value == cone_id for value in updated)
                    if cone_id != "overflow"
                    else 0
                )
                observations[key] = DirectionObservation(
                    principal_id=principal,
                    resolution=resolution,
                    cone_id=cone_id,
                    repeat_count=repeat_count,
                    z=float(z_values.get(key, 0.0)),
                )
        return DirectionTransition(
            observations=observations,
            next_history=next_history,
        )

    def q(
        self,
        transition: DirectionTransition,
        *,
        principal_id: str,
        resolution: str,
        server_usage: float,
        principal_usage: float,
    ) -> float:
        if not self.enabled:
            return 1.0
        config = self._resolution_config(resolution)
        observation = transition.observations.get(
            (str(principal_id), str(resolution))
        )
        if observation is None:
            return 1.0
        conditions = (
            observation.z > float(config["z_threshold"]),
            observation.cone_id != "overflow",
            observation.repeat_count >= int(config["min_matches"]),
            float(server_usage) > float(config["server_usage_threshold"]),
            float(principal_usage) > float(config["principal_usage_threshold"]),
        )
        return float(config["q_value"]) if all(conditions) else 1.0

    def server_window(self, resolution: str) -> int:
        return int(self._resolution_config(resolution)["server_window"])

    def principal_window(self, resolution: str) -> int:
        return int(self._resolution_config(resolution)["principal_window"])

    def commit(self, transition: DirectionTransition) -> None:
        self._history = {
            key: tuple(values) for key, values in transition.next_history.items()
        }

    def state_dict(self) -> dict[tuple[str, str], tuple[str, ...]]:
        return dict(self._history)

    def load_state_dict(
        self, state: Mapping[tuple[str, str], Sequence[str]]
    ) -> None:
        checked: dict[tuple[str, str], tuple[str, ...]] = {}
        for raw_key, values in state.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError("invalid direction checkpoint key")
            checked[(str(raw_key[0]), str(raw_key[1]))] = tuple(
                str(value) for value in values
            )
        self._history = checked
