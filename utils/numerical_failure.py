"""Structured numerical failures that survive Flower/Ray exception wrapping."""
from __future__ import annotations

import json
from typing import Any

MARKER = "FEDSEC_NUMERICAL_V1:"


class NumericalFailure(FloatingPointError):
    def __init__(self, reason: str, stage: str, detail: str, round_number: int | None = None):
        self.evidence = {"schema_version": 1, "numerical_state": reason,
                         "stage": stage, "detail": detail, "round": round_number}
        super().__init__(MARKER + json.dumps(self.evidence, separators=(",", ":")))

    def __reduce__(self):
        e = self.evidence
        return type(self), (e['numerical_state'], e['stage'], e['detail'], e['round'])


def numerical_evidence(value: Any) -> dict | None:
    """Read an explicit structured marker, never infer from incidental wording."""
    if isinstance(value, NumericalFailure):
        return dict(value.evidence)
    if isinstance(value, BaseException):
        for nested in (value.__cause__, value.__context__):
            if nested is not None and nested is not value:
                found = numerical_evidence(nested)
                if found:
                    return found
    text = str(value)
    start = text.find(MARKER)
    if start >= 0:
        try:
            result, _ = json.JSONDecoder().raw_decode(text[start + len(MARKER):])
            if result.get("schema_version") == 1 and result.get("numerical_state"):
                return result
        except (ValueError, TypeError, AttributeError):
            pass
    return None
