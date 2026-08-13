"""Principal-participation-axis exposure ledger for RTC-v3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class PrincipalParticipation:
    server_round: int
    nominal_mass: float
    profile: str
    exposures: Mapping[str, float]
    solver_failed: bool = False


class PrincipalExposureLedger:
    def __init__(self) -> None:
        self._events: dict[str, list[PrincipalParticipation]] = {}

    def participation_count(self, principal_id: str) -> int:
        return len(self._events.get(str(principal_id), ()))

    def prior_events(
        self,
        principal_id: str,
        window: int,
    ) -> tuple[PrincipalParticipation, ...]:
        width = int(window)
        if width <= 0:
            raise ValueError("principal participation window must be positive")
        events = self._events.get(str(principal_id), ())
        return tuple(events[-max(0, width - 1) :]) if width > 1 else ()

    def history(
        self,
        principal_id: str,
        exposure_key: str,
        window: int,
    ) -> float:
        return float(
            sum(
                float(event.exposures.get(str(exposure_key), 0.0))
                for event in self.prior_events(principal_id, window)
            )
        )

    def nominal_mass_with_current(
        self,
        principal_id: str,
        window: int,
        current_nominal_mass: float,
    ) -> float:
        return float(
            sum(event.nominal_mass for event in self.prior_events(principal_id, window))
            + float(current_nominal_mass)
        )

    def profiles_with_current(
        self,
        principal_id: str,
        window: int,
        current_profile: str,
    ) -> set[str]:
        return {
            *(event.profile for event in self.prior_events(principal_id, window)),
            str(current_profile),
        }

    def commit(
        self,
        events: Mapping[str, PrincipalParticipation],
    ) -> None:
        prepared: list[tuple[str, PrincipalParticipation]] = []
        for principal_id, event in events.items():
            principal = str(principal_id)
            existing = self._events.get(principal, ())
            if existing and event.server_round <= existing[-1].server_round:
                raise ValueError(
                    "principal participation commits must advance server round"
                )
            if event.nominal_mass < 0.0 or not math.isfinite(event.nominal_mass):
                raise ValueError("principal nominal mass must be non-negative")
            if any(
                float(value) < 0.0 or not math.isfinite(float(value))
                for value in event.exposures.values()
            ):
                raise ValueError("principal exposure must be non-negative")
            prepared.append((principal, event))
        for principal, event in prepared:
            self._events.setdefault(principal, []).append(event)

    def snapshot(self) -> dict[str, tuple[PrincipalParticipation, ...]]:
        return {
            principal: tuple(events)
            for principal, events in self._events.items()
        }

    def load_snapshot(
        self,
        snapshot: Mapping[str, tuple[PrincipalParticipation, ...]],
    ) -> None:
        candidate = PrincipalExposureLedger()
        for principal, events in snapshot.items():
            previous_round = 0
            for event in events:
                if not isinstance(event, PrincipalParticipation):
                    raise ValueError("invalid principal ledger checkpoint event")
                if event.server_round <= previous_round:
                    raise ValueError("principal checkpoint rounds must increase")
                if event.nominal_mass < 0.0 or not math.isfinite(event.nominal_mass):
                    raise ValueError("invalid principal checkpoint nominal mass")
                if any(
                    float(value) < 0.0 or not math.isfinite(float(value))
                    for value in event.exposures.values()
                ):
                    raise ValueError("invalid principal checkpoint exposure")
                previous_round = event.server_round
            candidate._events[str(principal)] = list(events)
        self._events = candidate._events
