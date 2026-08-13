"""Server-round exposure ledger for RTC-v3."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Mapping


class ServerExposureLedger:
    """Store verified exposure by real server round.

    Missing rounds contribute zero, but they still advance server time because
    window lookup is based on the numeric round, not append count.
    """

    def __init__(self) -> None:
        self._values: dict[str, dict[str, dict[int, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._last_committed_round = 0

    def history(
        self,
        resolution: str,
        exposure_type: str,
        server_round: int,
        window: int,
    ) -> float:
        round_value = int(server_round)
        width = int(window)
        if round_value <= 0 or width <= 0:
            raise ValueError("server round and window must be positive")
        values = self._values[str(resolution)][str(exposure_type)]
        start = max(1, round_value - width + 1)
        return float(
            sum(values.get(previous, 0.0) for previous in range(start, round_value))
        )

    def commit(
        self,
        server_round: int,
        exposures: Mapping[str, Mapping[str, float]],
    ) -> None:
        round_value = int(server_round)
        if round_value <= self._last_committed_round:
            raise ValueError(
                "server exposure commits must use strictly increasing server rounds"
            )
        prepared: list[tuple[str, str, float]] = []
        for resolution, typed_values in exposures.items():
            for exposure_type, raw_value in typed_values.items():
                value = float(raw_value)
                if value < 0.0 or value != value or value == float("inf"):
                    raise ValueError("committed exposure must be finite and non-negative")
                prepared.append((str(resolution), str(exposure_type), value))
        for resolution, exposure_type, value in prepared:
            self._values[resolution][exposure_type][round_value] = value
        self._last_committed_round = round_value

    def snapshot(self) -> dict[str, dict[str, dict[int, float]]]:
        return {
            resolution: {
                exposure_type: dict(round_values)
                for exposure_type, round_values in typed.items()
            }
            for resolution, typed in self._values.items()
        }

    def load_snapshot(
        self, snapshot: Mapping[str, Mapping[str, Mapping[int, float]]]
    ) -> None:
        candidate = ServerExposureLedger()
        maximum_round = 0
        for resolution, typed in snapshot.items():
            for exposure_type, round_values in typed.items():
                for raw_round, raw_value in round_values.items():
                    round_value = int(raw_round)
                    value = float(raw_value)
                    if round_value <= 0 or value < 0.0 or not math.isfinite(value):
                        raise ValueError("invalid server ledger checkpoint")
                    candidate._values[str(resolution)][str(exposure_type)][
                        round_value
                    ] = value
                    maximum_round = max(maximum_round, round_value)
        candidate._last_committed_round = maximum_round
        self._values = candidate._values
        self._last_committed_round = candidate._last_committed_round
