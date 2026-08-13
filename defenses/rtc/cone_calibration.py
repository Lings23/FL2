"""Deterministic clean-only calibration for RTC-v3 offline stable cones."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class CalibratedConePhase:
    profile: str
    match_threshold: float
    update_threshold: float
    max_angular_drift: float
    prototypes: tuple[Mapping[str, Any], ...]
    coverage: float
    round_link_threshold: float


def _unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0.0)


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else np.zeros_like(vector)


def _complete_labels(vectors: np.ndarray, similarity: float) -> np.ndarray:
    if len(vectors) <= 1:
        return np.ones(len(vectors), dtype=np.int64)
    distances = pdist(vectors, metric="cosine")
    distances = np.nan_to_num(distances, nan=2.0, posinf=2.0, neginf=0.0)
    hierarchy = linkage(distances, method="complete", optimal_ordering=True)
    return fcluster(hierarchy, t=max(0.0, 1.0 - similarity), criterion="distance")


def _nearest_other_principal(frame: pd.DataFrame, vectors: np.ndarray) -> list[float]:
    values: list[float] = []
    for (_, _), positions in frame.groupby(
        ["calibration_seed", "round"], sort=True
    ).indices.items():
        indices = np.asarray(sorted(positions), dtype=np.int64)
        local = vectors[indices]
        similarities = local @ local.T
        principals = frame.iloc[indices]["principal_id"].astype(str).to_numpy()
        for row in range(len(indices)):
            mask = principals != principals[row]
            if np.any(mask):
                values.append(float(np.max(similarities[row, mask])))
    if not values:
        raise ValueError("clean cone calibration has no cross-principal neighbours")
    return values


def _candidate_centroids(
    frame: pd.DataFrame,
    vectors: np.ndarray,
    similarity: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for (seed, server_round), positions in frame.groupby(
        ["calibration_seed", "round"], sort=True
    ).indices.items():
        indices = np.asarray(sorted(positions), dtype=np.int64)
        labels = _complete_labels(vectors[indices], similarity)
        for label in sorted(set(labels.tolist())):
            members = indices[labels == label]
            principals = sorted(
                set(frame.iloc[members]["principal_id"].astype(str))
            )
            if len(principals) < 2:
                continue
            masses = frame.iloc[members]["nominal_mass"].to_numpy(dtype=np.float64)
            center = _unit(np.sum(masses[:, None] * vectors[members], axis=0))
            if not np.any(center):
                continue
            candidates.append(
                {
                    "vector": center,
                    "mass": float(masses.sum()),
                    "principals": set(principals),
                    "rounds": {(int(seed), int(server_round))},
                    "first_key": (
                        int(seed),
                        int(server_round),
                        min(frame.iloc[members]["principal_id"].astype(str)),
                    ),
                }
            )
    return candidates


def _phase_centers(
    frame: pd.DataFrame,
    vectors: np.ndarray,
    similarity: float,
    max_prototypes: int,
) -> list[dict[str, Any]]:
    candidates = _candidate_centroids(frame, vectors, similarity)
    if not candidates:
        raise ValueError("clean within-round clustering produced no supported cones")
    candidate_vectors = np.stack([item["vector"] for item in candidates])
    labels = _complete_labels(candidate_vectors, similarity)
    centers: list[dict[str, Any]] = []
    for label in sorted(set(labels.tolist())):
        positions = np.flatnonzero(labels == label)
        masses = np.asarray([candidates[index]["mass"] for index in positions])
        vector = _unit(
            np.sum(masses[:, None] * candidate_vectors[positions], axis=0)
        )
        principals: set[str] = set()
        rounds: set[tuple[int, int]] = set()
        for index in positions:
            principals.update(candidates[index]["principals"])
            rounds.update(candidates[index]["rounds"])
        centers.append(
            {
                "vector": vector,
                "support_mass": float(masses.sum()),
                "support_principals": len(principals),
                "support_rounds": len(rounds),
                "first_key": min(candidates[index]["first_key"] for index in positions),
            }
        )
    centers.sort(
        key=lambda item: (
            -item["support_mass"],
            -item["support_principals"],
            -item["support_rounds"],
            item["first_key"],
        )
    )
    return centers[:max_prototypes]


def _nearest(vectors: np.ndarray, centers: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([np.asarray(item["vector"], dtype=np.float64) for item in centers])
    similarities = vectors @ matrix.T
    best = np.argmax(similarities, axis=1)
    return best, similarities[np.arange(len(vectors)), best]


def _align_ids(
    phases: list[tuple[str, list[dict[str, Any]], float]],
) -> list[tuple[str, list[dict[str, Any]], float]]:
    next_id = 0
    previous: list[dict[str, Any]] = []
    result: list[tuple[str, list[dict[str, Any]], float]] = []
    for profile, centers, threshold in phases:
        assigned: dict[int, str] = {}
        if previous and centers:
            left = np.stack([item["vector"] for item in previous])
            right = np.stack([item["vector"] for item in centers])
            rows, columns = linear_sum_assignment(-(left @ right.T))
            for row, column in zip(rows, columns):
                if float(np.dot(left[row], right[column])) >= threshold:
                    assigned[int(column)] = str(previous[row]["cone_id"])
        for index, center in enumerate(centers):
            if index not in assigned:
                assigned[index] = f"cone:{next_id}"
                next_id += 1
            center["cone_id"] = assigned[index]
        centers.sort(key=lambda item: item["cone_id"])
        result.append((profile, centers, threshold))
        previous = centers
    return result


def _clean_update_drift(
    frame: pd.DataFrame,
    vectors: np.ndarray,
    centers: Sequence[Mapping[str, Any]],
    update_threshold: float,
    momentum: float,
) -> float:
    current = {str(item["cone_id"]): _unit(item["vector"]) for item in centers}
    observed: list[float] = []
    for (_, _), positions in frame.groupby(
        ["calibration_seed", "round"], sort=True
    ).indices.items():
        indices = np.asarray(sorted(positions), dtype=np.int64)
        ordered = [current[key] for key in sorted(current)]
        ids = sorted(current)
        best, similarities = _nearest(
            vectors[indices], [{"vector": value} for value in ordered]
        )
        for cone_position, cone_id in enumerate(ids):
            selected = [
                int(indices[row])
                for row in range(len(indices))
                if int(best[row]) == cone_position
                and float(similarities[row]) >= update_threshold
            ]
            principals = sorted(set(frame.iloc[selected]["principal_id"].astype(str)))
            if len(principals) < 2:
                continue
            per_principal = []
            for principal in principals:
                members = [
                    index
                    for index in selected
                    if str(frame.iloc[index]["principal_id"]) == principal
                ]
                per_principal.append(_unit(np.sum(vectors[members], axis=0)))
            mean = _unit(np.sum(per_principal, axis=0))
            proposed = _unit(momentum * current[cone_id] + (1.0 - momentum) * mean)
            angle = math.acos(
                float(np.clip(np.dot(current[cone_id], proposed), -1.0, 1.0))
            )
            observed.append(angle)
            current[cone_id] = proposed
    return float(np.quantile(observed, 0.99)) if observed else 0.0


def calibrate_offline_cones(
    clients: pd.DataFrame,
    *,
    profile_names: Sequence[str],
    dimension: int = 256,
    max_prototypes: int = 16,
    momentum: float = 0.98,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    sketch_columns = [f"rtc_v3_sketch_full_{index:03d}" for index in range(dimension)]
    missing = [column for column in sketch_columns if column not in clients]
    if missing:
        raise ValueError(
            "clean calibration is missing sketch exports: " + ", ".join(missing[:3])
        )
    ordered = clients.sort_values(
        ["calibration_seed", "round", "principal_id", "cid"],
        kind="mergesort",
    ).reset_index(drop=True)
    vectors = _unit_rows(ordered[sketch_columns].to_numpy(dtype=np.float64))
    if np.any(np.linalg.norm(vectors, axis=1) <= 0.0):
        raise ValueError("clean calibration contains zero residual sketches")

    preliminary: list[tuple[str, list[dict[str, Any]], float]] = []
    thresholds: dict[str, tuple[float, float]] = {}
    for profile in profile_names:
        mask = ordered["profile"].astype(str).to_numpy() == str(profile)
        phase_frame = ordered.loc[mask].reset_index(drop=True)
        phase_vectors = vectors[mask]
        neighbour = _nearest_other_principal(phase_frame, phase_vectors)
        round_threshold = float(np.quantile(neighbour, 0.01))
        held_out_similarities: list[float] = []
        seeds = sorted(set(phase_frame["calibration_seed"].astype(int)))
        for held_out in seeds:
            training_mask = phase_frame["calibration_seed"].astype(int).to_numpy() != held_out
            evaluation_mask = ~training_mask
            centers = _phase_centers(
                phase_frame.loc[training_mask].reset_index(drop=True),
                phase_vectors[training_mask],
                round_threshold,
                max_prototypes,
            )
            _, similarity = _nearest(phase_vectors[evaluation_mask], centers)
            held_out_similarities.extend(similarity.tolist())
        if not held_out_similarities:
            raise ValueError(f"phase {profile!r} has no out-of-fold similarities")
        match = float(np.quantile(held_out_similarities, 0.01))
        update = max(match, float(np.quantile(held_out_similarities, 0.25)))
        thresholds[str(profile)] = (match, update)
        final_centers = _phase_centers(
            phase_frame, phase_vectors, round_threshold, max_prototypes
        )
        preliminary.append((str(profile), final_centers, match))

    aligned = _align_ids(preliminary)
    assignments = ordered[
        ["calibration_seed", "round", "profile", "cid", "principal_id", "nominal_mass", "rtc_v3_residual_norm"]
    ].copy()
    assignments["cone_id"] = "overflow"
    assignments["similarity"] = -1.0
    phase_payload: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for profile, centers, round_threshold in aligned:
        mask = ordered["profile"].astype(str).to_numpy() == profile
        phase_vectors = vectors[mask]
        best, similarity = _nearest(phase_vectors, centers)
        match, update = thresholds[profile]
        matched = similarity >= match
        ids = [str(item["cone_id"]) for item in centers]
        assignment_values = np.asarray(
            [ids[int(index)] if keep else "overflow" for index, keep in zip(best, matched)],
            dtype=object,
        )
        assignments.loc[mask, "cone_id"] = assignment_values
        assignments.loc[mask, "similarity"] = similarity
        coverage = float(np.mean(matched))
        if coverage < 0.95:
            raise ValueError(
                f"offline cone coverage for {profile!r} is {coverage:.4f}, below 0.95"
            )
        phase_frame = ordered.loc[mask].reset_index(drop=True)
        drift = _clean_update_drift(
            phase_frame, phase_vectors, centers, update, momentum
        )
        phase_payload[profile] = {
            "round_link_threshold": float(round_threshold),
            "match_threshold": match,
            "update_threshold": update,
            "max_angular_drift": drift,
            "prototypes": [
                {
                    "cone_id": str(item["cone_id"]),
                    "vector": np.asarray(item["vector"], dtype=np.float64).tolist(),
                    "support_mass": float(item["support_mass"]),
                    "support_principals": int(item["support_principals"]),
                    "support_rounds": int(item["support_rounds"]),
                }
                for item in centers
            ],
        }
        diagnostics[profile] = {
            "prototype_count": len(centers),
            "coverage": coverage,
            "overflow_rate": 1.0 - coverage,
            "similarity_min": float(np.min(similarity)),
            "similarity_q01": float(np.quantile(similarity, 0.01)),
            "similarity_q25": float(np.quantile(similarity, 0.25)),
            "max_angular_drift": drift,
        }
    config = {
        "enabled": True,
        "mode": "offline_controlled_v1",
        "dimension": int(dimension),
        "seed": 42,
        "sketch_algorithm_version": "splitmix64_v1",
        "coordinate_median_block_size": 262144,
        "max_prototypes": int(max_prototypes),
        "momentum": float(momentum),
        "min_update_principals": 2,
        "phases": phase_payload,
        "clustering": {
            "algorithm": "deterministic_complete_linkage_cosine_v1",
            "round_threshold_quantile": 0.01,
            "match_threshold_quantile": 0.01,
            "update_threshold_quantile": 0.25,
            "cross_fit": "leave_one_calibration_seed_out",
            "minimum_cluster_principals": 2,
        },
    }
    return config, assignments, diagnostics
