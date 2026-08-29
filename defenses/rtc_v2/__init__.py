"""Legacy RTC-V2 compatibility API.

RTC-V2 is not a default or formal experiment defense.  This package exists
only to reproduce historical runs explicitly via ``rtc_v2_legacy``.
"""

from defenses.rtc.aggregation import InfluenceAggregator
from defenses.rtc.history import ClientHistory, RoundRecord
from defenses.rtc.metrics import build_round_metrics
from defenses.rtc.scoring import RiskScorer
from defenses.time_consistency_defense import TimeConsistencyDefense

__all__ = [
    "ClientHistory",
    "InfluenceAggregator",
    "RiskScorer",
    "RoundRecord",
    "TimeConsistencyDefense",
    "build_round_metrics",
]
