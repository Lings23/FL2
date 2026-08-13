"""Stable, round-atomic cone prototype lifecycle for RTC-v3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from defenses.rtc.sketch import sketch_hash


@dataclass(frozen=True)
class Prototype:
    cone_id: str
    vector: np.ndarray
    last_match_round: int


@dataclass(frozen=True)
class PrototypeTransition:
    assignments: tuple[str, ...]
    similarities: tuple[float, ...]
    next_prototypes: tuple[Prototype, ...]
    next_id: int
    sketches: tuple[np.ndarray, ...] = ()
    nominal_masses: tuple[float, ...] = ()
    principal_ids: tuple[str, ...] = ()


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else np.zeros_like(vector)


def _slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate unit vectors while preserving a hard angular step bound."""

    left = _unit(start)
    right = _unit(end)
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= np.finfo(np.float64).eps:
        return left
    if angle >= math.pi - 1e-10:
        # The direction of an antipodal interpolation is undefined.  A stable
        # orthogonal direction makes the result deterministic.
        basis = np.zeros_like(left)
        basis[int(np.argmin(np.abs(left)))] = 1.0
        orthogonal = _unit(basis - float(np.dot(basis, left)) * left)
        value = math.cos(fraction * angle) * left + math.sin(
            fraction * angle
        ) * orthogonal
        return _unit(value)
    denominator = math.sin(angle)
    value = (
        math.sin((1.0 - fraction) * angle) / denominator * left
        + math.sin(fraction * angle) / denominator * right
    )
    return _unit(value)


class StablePrototypeBank:
    """Frozen assignment with optional, tightly controlled prototype updates.

    V2 banks are initialized from clean calibration prototypes and never create,
    replace or retire a prototype online.  The legacy path remains readable for
    old manifests and tests and preserves its exact-sketch-hash creation rule.
    """

    def __init__(
        self,
        *,
        max_prototypes: int = 16,
        match_threshold: float = 0.85,
        momentum: float = 0.8,
        retirement_rounds: int = 16,
        initial_prototypes: Sequence[Prototype] = (),
        controlled: bool = False,
        update_threshold: float | None = None,
        max_angular_drift: float = math.pi,
        min_update_principals: int = 2,
    ) -> None:
        if max_prototypes <= 0 or not -1.0 <= match_threshold <= 1.0:
            raise ValueError("invalid prototype capacity or threshold")
        if not 0.0 <= momentum < 1.0 or retirement_rounds <= 0:
            raise ValueError("invalid prototype momentum or retirement period")
        threshold = match_threshold if update_threshold is None else update_threshold
        if not match_threshold <= threshold <= 1.0:
            raise ValueError("update_threshold must be at least match_threshold")
        if not 0.0 <= max_angular_drift <= math.pi:
            raise ValueError("max_angular_drift must be in [0, pi]")
        if min_update_principals < 2:
            raise ValueError("min_update_principals must be at least two")
        checked: list[Prototype] = []
        seen: set[str] = set()
        dimension: int | None = None
        for prototype in initial_prototypes:
            cone_id = str(prototype.cone_id)
            vector = _unit(prototype.vector)
            if not cone_id or cone_id == "overflow" or cone_id in seen:
                raise ValueError("offline prototype IDs must be unique and non-overflow")
            if not np.all(np.isfinite(vector)) or not np.any(vector):
                raise ValueError("offline prototype vectors must be finite and nonzero")
            if dimension is None:
                dimension = int(vector.size)
            if int(vector.size) != dimension:
                raise ValueError("offline prototype vectors must share one dimension")
            seen.add(cone_id)
            checked.append(Prototype(cone_id, vector, int(prototype.last_match_round)))
        if len(checked) > max_prototypes:
            raise ValueError("offline prototype count exceeds configured capacity")
        if controlled and not checked:
            raise ValueError("controlled prototype bank requires offline prototypes")
        self.max_prototypes = int(max_prototypes)
        self.match_threshold = float(match_threshold)
        self.update_threshold = float(threshold)
        self.momentum = float(momentum)
        self.retirement_rounds = int(retirement_rounds)
        self.max_angular_drift = float(max_angular_drift)
        self.min_update_principals = int(min_update_principals)
        self.controlled = bool(controlled)
        self._prototypes = tuple(sorted(checked, key=lambda item: item.cone_id))
        self._next_id = len(self._prototypes)
        self.last_update_metrics: dict[str, Any] = {
            "candidate_count": 0.0,
            "applied_count": 0.0,
            "max_angular_drift": 0.0,
            "drift_clipped_count": 0.0,
            "principal_support_by_cone": {},
            "applied_cone_ids": (),
        }

    @property
    def prototypes(self) -> tuple[Prototype, ...]:
        return self._prototypes

    def prepare(
        self,
        *,
        server_round: int,
        sketches: Sequence[np.ndarray],
        nominal_masses: Sequence[float],
        client_ids: Sequence[str],
        principal_ids: Sequence[str] | None = None,
    ) -> PrototypeTransition:
        principals = tuple(str(value) for value in (principal_ids or client_ids))
        if not (
            len(sketches)
            == len(nominal_masses)
            == len(client_ids)
            == len(principals)
        ):
            raise ValueError("prototype inputs must have the same length")
        vectors = tuple(_unit(np.asarray(value, dtype=np.float64)) for value in sketches)
        frozen = tuple(self._prototypes)
        assignments: list[str] = []
        similarities: list[float] = []
        matched: dict[str, list[int]] = {}
        unmatched: list[int] = []
        for index, vector in enumerate(vectors):
            if frozen and float(np.linalg.norm(vector)) > 0.0:
                values = [float(np.dot(vector, prototype.vector)) for prototype in frozen]
                best = min(
                    range(len(values)),
                    key=lambda position: (-values[position], frozen[position].cone_id),
                )
                similarity = values[best]
                if similarity >= self.match_threshold:
                    cone_id = frozen[best].cone_id
                    assignments.append(cone_id)
                    similarities.append(similarity)
                    matched.setdefault(cone_id, []).append(index)
                    continue
                similarities.append(similarity)
            else:
                similarities.append(-1.0)
            assignments.append("overflow")
            unmatched.append(index)

        # V2 transitions preserve the exact offline ID set.  The eligible
        # subset is applied only after all aggregation constraints are known.
        if self.controlled:
            return PrototypeTransition(
                assignments=tuple(assignments),
                similarities=tuple(similarities),
                next_prototypes=frozen,
                next_id=self._next_id,
                sketches=vectors,
                nominal_masses=tuple(float(value) for value in nominal_masses),
                principal_ids=principals,
            )

        # Legacy compatibility is deliberately byte-for-byte semantic with the
        # original V1 baseline.  Formal V2 manifests always take the controlled
        # return above and therefore never create prototypes from sketch hashes.
        next_values: list[Prototype] = []
        for prototype in frozen:
            indices = matched.get(prototype.cone_id, [])
            if indices:
                masses = np.asarray([nominal_masses[index] for index in indices])
                total = float(masses.sum(dtype=np.float64))
                mean = _unit(
                    sum(mass * vectors[index] for mass, index in zip(masses, indices))
                    / max(total, np.finfo(np.float64).eps)
                )
                next_values.append(
                    Prototype(
                        prototype.cone_id,
                        _unit(self.momentum * prototype.vector + (1.0 - self.momentum) * mean),
                        int(server_round),
                    )
                )
            elif int(server_round) - prototype.last_match_round < self.retirement_rounds:
                next_values.append(prototype)
        grouped: dict[str, dict[str, Any]] = {}
        for index in unmatched:
            if not np.any(vectors[index]):
                continue
            digest = sketch_hash(vectors[index])
            entry = grouped.setdefault(
                digest,
                {
                    "mass": 0.0,
                    "vector": vectors[index],
                    "client": str(client_ids[index]),
                },
            )
            entry["mass"] = float(entry["mass"]) + float(nominal_masses[index])
            entry["client"] = min(str(entry["client"]), str(client_ids[index]))
        groups = sorted(
            grouped.items(),
            key=lambda item: (
                -float(item[1]["mass"]),
                item[0],
                str(item[1]["client"]),
            ),
        )
        next_id = self._next_id
        for _, entry in groups[: max(0, self.max_prototypes - len(next_values))]:
            vector = _unit(np.asarray(entry["vector"], dtype=np.float64))
            next_values.append(Prototype(f"cone:{next_id}", vector, int(server_round)))
            next_id += 1
        return PrototypeTransition(
            assignments=tuple(assignments),
            similarities=tuple(similarities),
            next_prototypes=tuple(sorted(next_values, key=lambda item: item.cone_id)),
            next_id=next_id,
        )

    def commit(
        self,
        transition: PrototypeTransition,
        eligible_mask: Sequence[bool] | None = None,
    ) -> None:
        if not self.controlled:
            self._prototypes = tuple(
                Prototype(item.cone_id, np.asarray(item.vector).copy(), item.last_match_round)
                for item in transition.next_prototypes
            )
            self._next_id = int(transition.next_id)
            return
        eligible = np.asarray(
            eligible_mask if eligible_mask is not None else [False] * len(transition.assignments),
            dtype=bool,
        )
        if eligible.size != len(transition.assignments):
            raise ValueError("prototype eligibility mask has the wrong length")
        candidates = eligible & np.asarray(
            [
                assignment != "overflow" and similarity >= self.update_threshold
                for assignment, similarity in zip(
                    transition.assignments, transition.similarities
                )
            ],
            dtype=bool,
        )
        updated: list[Prototype] = []
        applied = 0
        applied_cone_ids: list[str] = []
        clipped = 0
        maximum = 0.0
        principal_support_by_cone: dict[str, int] = {}
        for prototype in self._prototypes:
            indices = [
                index
                for index, assignment in enumerate(transition.assignments)
                if candidates[index] and assignment == prototype.cone_id
            ]
            principals = sorted({transition.principal_ids[index] for index in indices})
            principal_support_by_cone[prototype.cone_id] = len(principals)
            if len(principals) < self.min_update_principals:
                updated.append(prototype)
                continue
            per_principal: list[np.ndarray] = []
            for principal in principals:
                positions = [
                    index for index in indices if transition.principal_ids[index] == principal
                ]
                masses = np.asarray(
                    [transition.nominal_masses[index] for index in positions],
                    dtype=np.float64,
                )
                per_principal.append(
                    _unit(
                        sum(
                            mass * transition.sketches[index]
                            for mass, index in zip(masses, positions)
                        )
                    )
                )
            mean = _unit(sum(per_principal))
            proposed = _unit(
                self.momentum * prototype.vector + (1.0 - self.momentum) * mean
            )
            angle = math.acos(
                float(np.clip(np.dot(prototype.vector, proposed), -1.0, 1.0))
            )
            if angle > self.max_angular_drift and angle > 0.0:
                proposed = _slerp(
                    prototype.vector, proposed, self.max_angular_drift / angle
                )
                angle = self.max_angular_drift
                clipped += 1
            maximum = max(maximum, angle)
            applied += 1
            applied_cone_ids.append(prototype.cone_id)
            updated.append(Prototype(prototype.cone_id, proposed, prototype.last_match_round))
        self._prototypes = tuple(sorted(updated, key=lambda item: item.cone_id))
        self.last_update_metrics = {
            "candidate_count": float(np.count_nonzero(candidates)),
            "applied_count": float(applied),
            "max_angular_drift": float(maximum),
            "drift_clipped_count": float(clipped),
            "principal_support_by_cone": principal_support_by_cone,
            "applied_cone_ids": tuple(applied_cone_ids),
        }

    def snapshot(self) -> Mapping[str, np.ndarray]:
        return {item.cone_id: item.vector.copy() for item in self._prototypes}

    def state_dict(self) -> dict[str, object]:
        return {
            "next_id": self._next_id,
            "controlled": self.controlled,
            "prototypes": tuple(self._prototypes),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if bool(state.get("controlled", self.controlled)) != self.controlled:
            raise ValueError("prototype checkpoint mode mismatch")
        prototypes = state.get("prototypes", ())
        if not isinstance(prototypes, (tuple, list)):
            raise ValueError("invalid prototype checkpoint")
        checked: list[Prototype] = []
        for prototype in prototypes:
            if not isinstance(prototype, Prototype):
                raise ValueError("invalid prototype checkpoint entry")
            vector = _unit(np.asarray(prototype.vector, dtype=np.float64))
            if not np.all(np.isfinite(vector)) or not np.any(vector):
                raise ValueError("prototype checkpoint contains invalid vectors")
            checked.append(Prototype(prototype.cone_id, vector, int(prototype.last_match_round)))
        if len(checked) > self.max_prototypes:
            raise ValueError("prototype checkpoint exceeds configured capacity")
        if self.controlled and {item.cone_id for item in checked} != {
            item.cone_id for item in self._prototypes
        }:
            raise ValueError("controlled prototype checkpoint changes offline IDs")
        self._prototypes = tuple(sorted(checked, key=lambda item: item.cone_id))
        self._next_id = int(state.get("next_id", len(checked)))
