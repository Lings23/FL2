"""Compatibility exports for historical RTC module paths.

New integrations should import the promoted API from :mod:`defenses.rtc_v3`
or the frozen historical API from :mod:`defenses.rtc_v2`.
"""

from .history import ClientHistory, RoundRecord
from .scoring import RiskScorer
from .aggregation import InfluenceAggregator
from .metrics import build_round_metrics
from .calibration import CalibrationManifest, build_manifest
from .v3 import RTCv3Defense

__all__ = [
    "ClientHistory",
    "RoundRecord",
    "RiskScorer",
    "InfluenceAggregator",
    "build_round_metrics",
    "CalibrationManifest",
    "build_manifest",
    "RTCv3Defense",
]
