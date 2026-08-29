"""Canonical public API for the promoted RTC-V3 defense.

Implementation modules remain under :mod:`defenses.rtc` so historical import
paths and checkpoints keep working.  New code should import from this package.
"""

from defenses.rtc.calibration import CalibrationManifest
from defenses.rtc.v3 import RTCv3Defense, RTCv3ClientRecord

__all__ = ["CalibrationManifest", "RTCv3ClientRecord", "RTCv3Defense"]
