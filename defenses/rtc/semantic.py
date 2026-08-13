"""Classifier-row semantic evidence for RTC-v3.

The implementation is deliberately data-free: it inspects only the classifier
head residual supplied by each client and clean-calibrated manifest constants.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


CLASSIFIER_WEIGHT_ROLE = "classifier_weight"
CLASSIFIER_BIAS_ROLE = "classifier_bias"


def ordered_class_pairs(num_classes: int) -> tuple[tuple[int, int], ...]:
    if int(num_classes) < 2:
        raise ValueError("semantic classifier contract requires at least two classes")
    return tuple(
        (source, target)
        for source in range(int(num_classes))
        for target in range(int(num_classes))
        if source != target
    )


def select_top_pair_indices(
    *,
    client_pair_indices: Sequence[int],
    exposure_coefficients: Sequence[float],
    nominal_masses: Sequence[float],
    pairs: Sequence[tuple[int, int]],
    top_k: int,
) -> tuple[int, ...]:
    """Select deterministic exposure-heavy pairs with numeric pair tie-breaks."""

    indices = np.asarray(client_pair_indices, dtype=np.int64)
    coefficients = np.asarray(exposure_coefficients, dtype=np.float64)
    masses = np.asarray(nominal_masses, dtype=np.float64)
    if indices.shape != coefficients.shape or indices.shape != masses.shape:
        raise ValueError("semantic Top-K inputs must have matching client shapes")
    if (
        np.any(indices < 0)
        or np.any(indices >= len(pairs))
        or not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(masses))
        or np.any(coefficients < 0.0)
        or np.any(masses < 0.0)
        or not 1 <= int(top_k) < len(pairs)
    ):
        raise ValueError("invalid semantic Top-K inputs")
    exposure = np.bincount(
        indices,
        weights=coefficients * masses,
        minlength=len(pairs),
    )
    ranked = sorted(
        range(len(pairs)),
        key=lambda index: (-float(exposure[index]), pairs[index][0], pairs[index][1]),
    )
    return tuple(ranked[: int(top_k)])


def classifier_contract_hash(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(contract),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ClassifierHeadLayout:
    weight_index: int
    bias_index: int | None
    num_classes: int
    feature_count: int

    @classmethod
    def build(
        cls,
        params: Sequence[np.ndarray],
        roles: Mapping[int | str, str],
    ) -> "ClassifierHeadLayout":
        normalized = {str(key): str(value).lower() for key, value in roles.items()}
        weight_indices = [
            index
            for index in range(len(params))
            if normalized.get(str(index)) == CLASSIFIER_WEIGHT_ROLE
        ]
        bias_indices = [
            index
            for index in range(len(params))
            if normalized.get(str(index)) == CLASSIFIER_BIAS_ROLE
        ]
        if len(weight_indices) != 1 or len(bias_indices) > 1:
            raise ValueError(
                "RTC-v3 semantic head requires exactly one classifier_weight "
                "and at most one classifier_bias"
            )
        weight_index = weight_indices[0]
        weight = np.asarray(params[weight_index])
        if weight.ndim != 2 or not np.issubdtype(weight.dtype, np.floating):
            raise ValueError("classifier_weight must be a floating CxD tensor")
        num_classes, feature_count = (int(value) for value in weight.shape)
        if num_classes < 2 or feature_count < 1:
            raise ValueError("classifier_weight has an invalid class-row shape")
        bias_index = bias_indices[0] if bias_indices else None
        if bias_index is not None:
            bias = np.asarray(params[bias_index])
            if (
                bias.ndim != 1
                or bias.shape[0] != num_classes
                or not np.issubdtype(bias.dtype, np.floating)
            ):
                raise ValueError("classifier_bias must be a floating vector of C rows")
        return cls(weight_index, bias_index, num_classes, feature_count)

    def contract(
        self,
        params: Sequence[np.ndarray],
        parameter_names: Mapping[int | str, str] | None = None,
    ) -> dict[str, Any]:
        names = {
            str(key): str(value) for key, value in (parameter_names or {}).items()
        }
        entries = []
        for index, role in (
            (self.weight_index, CLASSIFIER_WEIGHT_ROLE),
            (self.bias_index, CLASSIFIER_BIAS_ROLE),
        ):
            if index is None:
                continue
            array = np.asarray(params[index])
            entries.append(
                {
                    "index": int(index),
                    "role": role,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "name": names.get(str(index), ""),
                }
            )
        contract: dict[str, Any] = {
            "class_axis": 0,
            "num_classes": self.num_classes,
            "parameters": entries,
            "semantic_resolutions": {
                "head": [
                    int(self.weight_index),
                    *(
                        [int(self.bias_index)]
                        if self.bias_index is not None
                        else []
                    ),
                ],
                "classifier_weight_rows": [
                    {
                        "class_index": class_index,
                        "parameter_index": int(self.weight_index),
                        "axis": 0,
                    }
                    for class_index in range(self.num_classes)
                ],
                "classifier_bias_entries": (
                    [
                        {
                            "class_index": class_index,
                            "parameter_index": int(self.bias_index),
                            "axis": 0,
                        }
                        for class_index in range(self.num_classes)
                    ]
                    if self.bias_index is not None
                    else []
                ),
            },
        }
        contract["contract_hash"] = classifier_contract_hash(contract)
        return contract

    def validate_contract(
        self,
        params: Sequence[np.ndarray],
        expected: Mapping[str, Any],
        parameter_names: Mapping[int | str, str] | None = None,
    ) -> None:
        actual = self.contract(params, parameter_names)
        if dict(expected) != actual:
            raise ValueError(
                "RTC-v3 classifier-head contract mismatch: "
                f"expected hash {expected.get('contract_hash')!r}, "
                f"got {actual.get('contract_hash')!r}"
            )


@dataclass(frozen=True)
class SemanticBatch:
    pairs: tuple[tuple[int, int], ...]
    raw_pairs: np.ndarray
    z_values: np.ndarray
    normalized_head_norms: np.ndarray
    row_norms: np.ndarray
    head_norms: np.ndarray
    top_pair_indices: np.ndarray
    top_pair_signatures: tuple[np.ndarray, ...]


def hard_exposure_coefficients(
    risks: Sequence[float],
    normalized_head_norms: Sequence[float],
    risk_floor: float,
) -> np.ndarray:
    """Map continuous risk to non-negative hard-ledger exposure.

    Risk and q remain unchanged outside this function. The floor only removes
    calibrated sub-watch jitter from hard rolling-window accounting.
    """

    risk_values = np.asarray(risks, dtype=np.float64)
    head_values = np.asarray(normalized_head_norms, dtype=np.float64)
    floor = float(risk_floor)
    if (
        risk_values.shape != head_values.shape
        or not np.all(np.isfinite(risk_values))
        or not np.all(np.isfinite(head_values))
        or np.any(risk_values < 0.0)
        or np.any(risk_values > 1.0)
        or np.any(head_values < 0.0)
        or not np.isfinite(floor)
        or not 0.0 <= floor <= 1.0
    ):
        raise ValueError("invalid semantic hard-exposure inputs")
    return np.maximum(0.0, risk_values - floor) * head_values


def extract_semantic_batch(
    *,
    layout: ClassifierHeadLayout,
    residual_arrays: Sequence[Sequence[np.ndarray]],
    floating_indices: Sequence[int],
    phase_config: Mapping[str, Any],
) -> SemanticBatch:
    """Vectorize all ordered class-pair features for one server round."""

    position_by_index = {
        int(parameter_index): position
        for position, parameter_index in enumerate(floating_indices)
    }
    try:
        weight_position = position_by_index[layout.weight_index]
        bias_position = (
            position_by_index[layout.bias_index]
            if layout.bias_index is not None
            else None
        )
    except KeyError as exc:
        raise ValueError("semantic classifier tensor is not floating") from exc
    rows = []
    for client_residuals in residual_arrays:
        weight = np.asarray(client_residuals[weight_position], dtype=np.float64)
        if weight.shape != (layout.num_classes, layout.feature_count):
            raise ValueError("semantic classifier residual shape changed at runtime")
        if bias_position is not None:
            bias = np.asarray(
                client_residuals[bias_position], dtype=np.float64
            ).reshape(layout.num_classes, 1)
            weight = np.concatenate((weight, bias), axis=1)
        rows.append(weight)
    row_tensor = np.stack(rows, axis=0)
    row_norms = np.linalg.norm(row_tensor, axis=2)
    row_scales = np.asarray(phase_config.get("row_scales", []), dtype=np.float64)
    if row_scales.shape == ():
        row_scales = np.full(layout.num_classes, float(row_scales))
    if (
        row_scales.shape != (layout.num_classes,)
        or not np.all(np.isfinite(row_scales))
        or np.any(row_scales <= 0.0)
    ):
        raise ValueError("semantic row_scales must contain one positive value per class")
    normalized_rows = row_norms / row_scales.reshape(1, -1)
    pairs = ordered_class_pairs(layout.num_classes)
    sources = np.asarray([source for source, _ in pairs], dtype=np.int64)
    targets = np.asarray([target for _, target in pairs], dtype=np.int64)
    raw_pairs = normalized_rows[:, targets] - normalized_rows[:, sources]
    pair_centers = np.asarray(phase_config.get("pair_centers", []), dtype=np.float64)
    pair_scales = np.asarray(phase_config.get("pair_scales", []), dtype=np.float64)
    expected_shape = (len(pairs),)
    if pair_centers.shape != expected_shape or not np.all(np.isfinite(pair_centers)):
        raise ValueError("semantic pair_centers do not match the ordered-pair contract")
    if (
        pair_scales.shape != expected_shape
        or not np.all(np.isfinite(pair_scales))
        or np.any(pair_scales <= 0.0)
    ):
        raise ValueError("semantic pair_scales must be finite and positive")
    z_values = np.maximum(
        0.0,
        (raw_pairs - pair_centers.reshape(1, -1))
        / pair_scales.reshape(1, -1),
    )
    top_pair_indices = np.argmax(z_values, axis=1)
    signatures = []
    for client_index, pair_index in enumerate(top_pair_indices):
        source, target = pairs[int(pair_index)]
        signature = row_tensor[client_index, target] - row_tensor[client_index, source]
        norm = float(np.linalg.norm(signature))
        signatures.append(
            signature / norm if norm > np.finfo(np.float64).eps else np.zeros_like(signature)
        )
    head_scale = float(phase_config.get("head_exposure_scale", float("nan")))
    if not np.isfinite(head_scale) or head_scale <= 0.0:
        raise ValueError("semantic head_exposure_scale must be finite and positive")
    head_norms = np.linalg.norm(row_tensor.reshape(len(rows), -1), axis=1)
    normalized_head_norms = head_norms / head_scale
    return SemanticBatch(
        pairs=pairs,
        raw_pairs=raw_pairs,
        z_values=z_values,
        normalized_head_norms=normalized_head_norms,
        row_norms=row_norms,
        head_norms=head_norms,
        top_pair_indices=top_pair_indices,
        top_pair_signatures=tuple(signatures),
    )
