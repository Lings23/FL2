"""Deterministic, versioned signed CountSketch for RTC-v3 cone routing."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import threading
import time
from typing import Any, Sequence

import numpy as np


LEGACY_BLAKE2B_V1 = "legacy_blake2b_v1"
SPLITMIX64_V1 = "splitmix64_v1"
SUPPORTED_SKETCH_ALGORITHMS = frozenset({LEGACY_BLAKE2B_V1, SPLITMIX64_V1})

_CACHE_MAX_ENTRIES = 16
_CACHE_MAX_BYTES = 256 * 1024 * 1024
_CACHE: "OrderedDict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, int]]" = OrderedDict()
_CACHE_BYTES = 0
_CACHE_HITS = 0
_CACHE_MISSES = 0
_CACHE_EVICTIONS = 0
_CACHE_PRECOMPUTE_SECONDS = 0.0
_CACHE_LOCK = threading.RLock()


def _hash64(seed: int, coordinate: int, domain: bytes) -> int:
    payload = (
        int(seed).to_bytes(8, "little", signed=True)
        + int(coordinate).to_bytes(8, "little", signed=False)
        + domain
    )
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    )


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized unsigned SplitMix64 with platform-independent wraparound."""

    values = np.asarray(values, dtype=np.uint64)
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _mapping_key(
    arrays: Sequence[np.ndarray],
    *,
    dimension: int,
    seed: int,
    algorithm: str,
) -> tuple[Any, ...]:
    # Shapes bind the cache to the ordered model/resolution layout, rather than
    # accidentally sharing an entry between differently structured tensors.
    layout = tuple(tuple(int(value) for value in np.asarray(array).shape) for array in arrays)
    return (str(algorithm), int(seed), int(dimension), layout)


def _build_mapping(
    total: int,
    *,
    dimension: int,
    seed: int,
    algorithm: str,
) -> tuple[np.ndarray, np.ndarray]:
    bucket_dtype = np.uint16 if dimension <= np.iinfo(np.uint16).max else np.uint32
    if algorithm == LEGACY_BLAKE2B_V1:
        buckets = np.fromiter(
            (_hash64(seed, coordinate, b"bucket") % dimension for coordinate in range(total)),
            dtype=bucket_dtype,
            count=total,
        )
        signs = np.fromiter(
            (1 if (_hash64(seed, coordinate, b"sign") & 1) else -1 for coordinate in range(total)),
            dtype=np.int8,
            count=total,
        )
        return buckets, signs

    if algorithm != SPLITMIX64_V1:
        raise ValueError(f"unsupported CountSketch algorithm: {algorithm!r}")
    buckets = np.empty(total, dtype=bucket_dtype)
    signs = np.empty(total, dtype=np.int8)
    seed_bits = np.uint64(int(seed) & ((1 << 64) - 1))
    chunk_size = 1_048_576
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        coordinates = np.arange(start, stop, dtype=np.uint64)
        bucket_hash = _splitmix64(
            coordinates ^ seed_bits ^ np.uint64(0x243F6A8885A308D3)
        )
        sign_hash = _splitmix64(
            coordinates ^ seed_bits ^ np.uint64(0x13198A2E03707344)
        )
        buckets[start:stop] = (bucket_hash % np.uint64(dimension)).astype(
            bucket_dtype, copy=False
        )
        signs[start:stop] = np.where(
            (sign_hash & np.uint64(1)) != 0, 1, -1
        ).astype(np.int8, copy=False)
    return buckets, signs


def _cached_mapping(
    arrays: Sequence[np.ndarray],
    *,
    dimension: int,
    seed: int,
    algorithm: str,
) -> tuple[np.ndarray, np.ndarray]:
    global _CACHE_BYTES, _CACHE_HITS, _CACHE_MISSES
    global _CACHE_EVICTIONS, _CACHE_PRECOMPUTE_SECONDS

    key = _mapping_key(
        arrays, dimension=dimension, seed=seed, algorithm=algorithm
    )
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE_HITS += 1
            _CACHE.move_to_end(key)
            return cached[0], cached[1]
        _CACHE_MISSES += 1

        started = time.perf_counter()
        total = sum(int(np.asarray(array).size) for array in arrays)
        buckets, signs = _build_mapping(
            total, dimension=dimension, seed=seed, algorithm=algorithm
        )
        _CACHE_PRECOMPUTE_SECONDS += time.perf_counter() - started
        entry_bytes = int(buckets.nbytes + signs.nbytes)
        while _CACHE and (
            len(_CACHE) >= _CACHE_MAX_ENTRIES
            or _CACHE_BYTES + entry_bytes > _CACHE_MAX_BYTES
        ):
            _, (_, _, removed_bytes) = _CACHE.popitem(last=False)
            _CACHE_BYTES -= removed_bytes
            _CACHE_EVICTIONS += 1
        # Oversized single layouts remain usable but are not cached.
        if entry_bytes <= _CACHE_MAX_BYTES:
            _CACHE[key] = (buckets, signs, entry_bytes)
            _CACHE_BYTES += entry_bytes
        return buckets, signs


def clear_sketch_cache() -> None:
    """Clear mappings and counters; intended for tests and isolated benchmarks."""

    global _CACHE_BYTES, _CACHE_HITS, _CACHE_MISSES
    global _CACHE_EVICTIONS, _CACHE_PRECOMPUTE_SECONDS
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE_BYTES = 0
        _CACHE_HITS = 0
        _CACHE_MISSES = 0
        _CACHE_EVICTIONS = 0
        _CACHE_PRECOMPUTE_SECONDS = 0.0


def sketch_cache_stats() -> dict[str, float]:
    with _CACHE_LOCK:
        return {
            "hits": float(_CACHE_HITS),
            "misses": float(_CACHE_MISSES),
            "evictions": float(_CACHE_EVICTIONS),
            "entries": float(len(_CACHE)),
            "bytes": float(_CACHE_BYTES),
            "precompute_seconds": float(_CACHE_PRECOMPUTE_SECONDS),
        }


def signed_count_sketch(
    arrays: Sequence[np.ndarray],
    *,
    dimension: int = 256,
    seed: int = 0,
    algorithm: str = LEGACY_BLAKE2B_V1,
) -> np.ndarray:
    """Compute a deterministic CountSketch under an explicit algorithm version.

    Missing versions deliberately select the original BLAKE2b mapping so old
    calibration manifests retain their exact behavior.
    """

    dimension = int(dimension)
    if dimension <= 0:
        raise ValueError("CountSketch dimension must be positive")
    algorithm = str(algorithm)
    if algorithm not in SUPPORTED_SKETCH_ALGORITHMS:
        raise ValueError(f"unsupported CountSketch algorithm: {algorithm!r}")
    values_by_array: list[np.ndarray] = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("CountSketch input must be finite")
        values_by_array.append(values)

    buckets, signs = _cached_mapping(
        arrays, dimension=dimension, seed=int(seed), algorithm=algorithm
    )
    sketch = np.zeros(dimension, dtype=np.float64)
    offset = 0
    for values in values_by_array:
        stop = offset + values.size
        local_buckets = buckets[offset:stop]
        local_weights = signs[offset:stop].astype(np.float64) * values
        if algorithm == LEGACY_BLAKE2B_V1:
            # np.add.at preserves the reference coordinate accumulation order.
            np.add.at(sketch, local_buckets, local_weights)
        else:
            sketch += np.bincount(
                local_buckets,
                weights=local_weights,
                minlength=dimension,
            )
        offset = stop
    norm = float(np.linalg.norm(sketch))
    if norm > 0.0:
        sketch /= norm
    return sketch


def signed_count_sketch_reference(
    arrays: Sequence[np.ndarray],
    *,
    dimension: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Original BLAKE2b scalar implementation retained as an oracle."""

    if int(dimension) <= 0:
        raise ValueError("CountSketch dimension must be positive")
    sketch = np.zeros(int(dimension), dtype=np.float64)
    coordinate = 0
    for array in arrays:
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("CountSketch input must be finite")
        for value in values:
            bucket = _hash64(seed, coordinate, b"bucket") % int(dimension)
            sign = 1.0 if (_hash64(seed, coordinate, b"sign") & 1) else -1.0
            sketch[bucket] += sign * float(value)
            coordinate += 1
    norm = float(np.linalg.norm(sketch))
    if norm > 0.0:
        sketch /= norm
    return sketch


def sketch_hash(sketch: np.ndarray) -> str:
    canonical = np.asarray(sketch, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(canonical).hexdigest()
