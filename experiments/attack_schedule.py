"""Defense-independent round schedules for strict attack experiments.

The schedule is an exogenous part of the attack condition.  In particular it
must not depend on a defense's observations, otherwise nominally paired runs
would receive different attacks.  Random-looking schedules therefore use a
SHA-256 stream derived only from the attack condition and base seed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


PERIODIC_SCHEDULES = {
    "continuous_1_0": (1, 0),
    "short_1_1": (1, 1),
    "long_3_3": (3, 3),
}
SPARSE_RANDOM_SCHEDULE = "sparse_random_0.25"
SUPPORTED_SCHEDULES = {*PERIODIC_SCHEDULES, SPARSE_RANDOM_SCHEDULE}


def _sha256_unit_interval(payload: Mapping[str, Any]) -> float:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return integer / float(1 << 64)


def attack_schedule_active(
    *,
    schedule: str,
    round_number: int,
    start_round: int,
    end_round: int,
    base_seed: int,
    attack: str,
    malicious_fraction: float,
) -> bool:
    """Return the preregistered global attack state for one server round."""
    round_number = int(round_number)
    start_round = max(1, int(start_round))
    end_round = int(end_round)
    if round_number < start_round or (end_round >= 0 and round_number > end_round):
        return False
    if schedule in PERIODIC_SCHEDULES:
        on_rounds, off_rounds = PERIODIC_SCHEDULES[schedule]
        cycle = on_rounds + off_rounds
        return ((round_number - start_round) % cycle) < on_rounds
    if schedule != SPARSE_RANDOM_SCHEDULE:
        raise ValueError(f"Unsupported attack schedule: {schedule}")
    draw = _sha256_unit_interval(
        {
            "stream": "strict-sparse-attack-schedule-v1",
            "base_seed": int(base_seed),
            "attack": str(attack),
            "malicious_fraction": float(malicious_fraction),
            "start_round": start_round,
            "end_round": end_round,
            "round": round_number,
        }
    )
    return draw < 0.25


def build_attack_schedule(
    *,
    schedule: str,
    rounds: int,
    start_round: int,
    end_round: int,
    base_seed: int,
    attack: str,
    malicious_fraction: float,
) -> tuple[bool, ...]:
    return tuple(
        attack_schedule_active(
            schedule=schedule,
            round_number=round_number,
            start_round=start_round,
            end_round=end_round,
            base_seed=base_seed,
            attack=attack,
            malicious_fraction=malicious_fraction,
        )
        for round_number in range(1, int(rounds) + 1)
    )

