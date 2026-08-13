"""Deterministic validated-mass-weighted robust statistics for RTC-v3."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _validated_inputs(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value_array.size != weight_array.size:
        raise ValueError("values and weights must have the same length")
    if value_array.size == 0:
        raise ValueError("weighted statistic requires at least one observation")
    if not np.all(np.isfinite(value_array)):
        raise ValueError("weighted statistic values must be finite")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weighted statistic weights must be finite and non-negative")
    if float(weight_array.sum(dtype=np.float64)) <= 0.0:
        raise ValueError("weighted statistic requires positive total weight")
    return value_array, weight_array


def weighted_quantile_higher(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    quantile: float,
    *,
    principal_ids: Sequence[str] | None = None,
    client_ids: Sequence[str] | None = None,
) -> float:
    """Return the first observed value whose cumulative mass reaches ``q``.

    Sorting is deterministic by ``(value, principal_id, client_id)``.  The
    returned result is always an observation; no interpolation is performed.
    """

    value_array, weight_array = _validated_inputs(values, weights)
    q = float(quantile)
    if not np.isfinite(q) or q < 0.0 or q > 1.0:
        raise ValueError("quantile must be finite and in [0, 1]")
    size = value_array.size
    principals = np.asarray(
        [str(value) for value in principal_ids]
        if principal_ids is not None
        else [""] * size,
        dtype=str,
    )
    clients = np.asarray(
        [str(value) for value in client_ids]
        if client_ids is not None
        else [str(index) for index in range(size)],
        dtype=str,
    )
    if principals.size != size or clients.size != size:
        raise ValueError("stable identifier lengths must match values")

    order = np.lexsort((clients, principals, value_array))
    sorted_values = value_array[order]
    sorted_weights = weight_array[order]
    positive = sorted_weights > 0.0
    sorted_values = sorted_values[positive]
    sorted_weights = sorted_weights[positive]
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    target = q * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, sorted_values.size - 1)
    return float(sorted_values[index])


def weighted_median(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    *,
    principal_ids: Sequence[str] | None = None,
    client_ids: Sequence[str] | None = None,
) -> float:
    return weighted_quantile_higher(
        values,
        weights,
        0.5,
        principal_ids=principal_ids,
        client_ids=client_ids,
    )


def weighted_median_and_mad(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    *,
    principal_ids: Sequence[str] | None = None,
    client_ids: Sequence[str] | None = None,
) -> tuple[float, float]:
    median = weighted_median(
        values,
        weights,
        principal_ids=principal_ids,
        client_ids=client_ids,
    )
    mad = weighted_median(
        np.abs(np.asarray(values, dtype=np.float64) - median),
        weights,
        principal_ids=principal_ids,
        client_ids=client_ids,
    )
    return median, mad


def weighted_coordinate_median(
    arrays: Sequence[np.ndarray],
    weights: Sequence[float] | np.ndarray,
    *,
    principal_ids: Sequence[str] | None = None,
    client_ids: Sequence[str] | None = None,
    block_size: int = 262_144,
) -> np.ndarray:
    """Compute the exact coordinate-wise weighted higher median in blocks.

    The client axis is tiny while a model tensor can contain millions of
    coordinates.  Sorting a whole tensor at once creates a large temporary;
    calling :func:`weighted_median` once per coordinate is prohibitively slow.
    This implementation therefore vectorizes over bounded coordinate blocks.
    """

    if not arrays:
        raise ValueError("coordinate median requires at least one array")
    first = np.asarray(arrays[0])
    if not np.issubdtype(first.dtype, np.floating):
        raise TypeError("coordinate median requires floating arrays")
    flattened: list[np.ndarray] = []
    for array in arrays:
        candidate = np.asarray(array)
        if candidate.shape != first.shape:
            raise ValueError("coordinate median arrays must share a shape")
        if not np.issubdtype(candidate.dtype, np.floating):
            raise TypeError("coordinate median requires floating arrays")
        values = np.asarray(candidate, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("weighted statistic values must be finite")
        flattened.append(values)

    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(arrays) != weight_array.size:
        raise ValueError("array and weight counts must match")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weighted statistic weights must be finite and non-negative")
    positive = weight_array > 0.0
    if not np.any(positive):
        raise ValueError("weighted statistic requires positive total weight")
    if int(block_size) <= 0:
        raise ValueError("coordinate median block_size must be positive")

    size = weight_array.size
    principals = np.asarray(
        [str(value) for value in principal_ids]
        if principal_ids is not None
        else [""] * size,
        dtype=str,
    )
    clients = np.asarray(
        [str(value) for value in client_ids]
        if client_ids is not None
        else [str(index) for index in range(size)],
        dtype=str,
    )
    if principals.size != size or clients.size != size:
        raise ValueError("stable identifier lengths must match values")

    # Remove zero-mass clients once.  Pre-ordering by the identifier keys and
    # then using a stable value sort is exactly the reference lexsort order.
    identifier_order = np.lexsort((clients, principals))
    identifier_order = identifier_order[positive[identifier_order]]
    ordered_weights = weight_array[identifier_order]
    equal_weights = bool(np.all(ordered_weights == ordered_weights[0]))
    lower_middle = (ordered_weights.size - 1) // 2
    target = 0.5 * float(ordered_weights.sum(dtype=np.float64))

    output = np.empty(first.size, dtype=np.float64)
    for start in range(0, first.size, int(block_size)):
        stop = min(first.size, start + int(block_size))
        block = np.stack(
            [values[start:stop] for values in flattened], axis=0
        )[identifier_order]
        if equal_weights:
            # weighted_quantile_higher(q=.5) selects the lower observation for
            # an even number of equal-mass clients, not their arithmetic mean.
            output[start:stop] = np.partition(
                block, lower_middle, axis=0
            )[lower_middle]
            continue

        order = np.argsort(block, axis=0, kind="stable")
        sorted_weights = np.take_along_axis(
            np.broadcast_to(ordered_weights[:, None], block.shape), order, axis=0
        )
        cumulative = np.cumsum(sorted_weights, axis=0, dtype=np.float64)
        selected = np.argmax(cumulative >= target, axis=0)
        sorted_values = np.take_along_axis(block, order, axis=0)
        output[start:stop] = np.take_along_axis(
            sorted_values, selected[None, :], axis=0
        )[0]
    return output.reshape(first.shape)


def weighted_coordinate_median_reference(
    arrays: Sequence[np.ndarray],
    weights: Sequence[float] | np.ndarray,
    *,
    principal_ids: Sequence[str] | None = None,
    client_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Original scalar implementation retained as a correctness oracle."""

    if not arrays:
        raise ValueError("coordinate median requires at least one array")
    first = np.asarray(arrays[0])
    if not np.issubdtype(first.dtype, np.floating):
        raise TypeError("coordinate median requires floating arrays")
    for array in arrays[1:]:
        candidate = np.asarray(array)
        if candidate.shape != first.shape:
            raise ValueError("coordinate median arrays must share a shape")
        if not np.issubdtype(candidate.dtype, np.floating):
            raise TypeError("coordinate median requires floating arrays")
    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(arrays) != weight_array.size:
        raise ValueError("array and weight counts must match")
    stacked = np.stack(
        [np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays],
        axis=1,
    )
    output = np.empty(stacked.shape[0], dtype=np.float64)
    for coordinate, row in enumerate(stacked):
        output[coordinate] = weighted_median(
            row,
            weight_array,
            principal_ids=principal_ids,
            client_ids=client_ids,
        )
    return output.reshape(first.shape)
