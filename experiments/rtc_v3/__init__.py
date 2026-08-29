"""Canonical RTC-V3 experiment lifecycle.

The flat ``rtc_v3_*`` modules remain compatibility implementations.  Prefer
the concise package entry points for new automation and documentation.
"""

from experiments.rtc_fedavg_comparison import RTC_V3_PROMOTED_MANIFEST

__all__ = ["RTC_V3_PROMOTED_MANIFEST"]
