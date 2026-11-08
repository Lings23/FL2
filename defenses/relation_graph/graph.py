"""Lagged conditional residuals and corroboration on the client graph."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np


@dataclass(frozen=True)
class EdgeObservation:
    left: str
    right: str
    backbone: float
    task: float
    score: float | None = None

    @property
    def pair(self):
        return self.left, self.right

    def __post_init__(self):
        if not isinstance(self.left, str) or not isinstance(self.right, str) or not self.left < self.right:
            raise ValueError("Edge endpoints must be distinct, canonical principal IDs")
        if any(type(value) not in (float, int) or not math.isfinite(value) or not -1 <= value <= 1
               for value in (self.backbone, self.task)):
            raise ValueError("Edge similarities must be finite and in [-1, 1]")
        if self.score is not None and (type(self.score) not in (float, int) or not math.isfinite(self.score)):
            raise ValueError("Edge score must be finite or missing")


@dataclass(frozen=True)
class RoundObservation:
    round_number: int
    edges: tuple[EdgeObservation, ...]

    def __post_init__(self):
        if type(self.round_number) is not int or self.round_number < 1:
            raise ValueError("Invalid observation round")
        pairs = [edge.pair for edge in self.edges]
        if pairs != sorted(set(pairs)):
            raise ValueError("Graph observations must contain sorted, unique edges")


def role_relations(base, uploads, indices):
    """Average per-tensor cosines over tensors informative for both endpoints."""
    size = len(uploads)
    similarity = np.zeros((size, size), dtype=np.float64)
    support = np.zeros((size, size), dtype=np.int32)
    for index in indices:
        anchor = base[index].astype(np.float64).reshape(-1)
        with np.errstate(over="raise", invalid="raise"):
            delta = np.stack([params[index].astype(np.float64).reshape(-1) - anchor for params in uploads])
        amplitude = np.max(np.abs(delta), axis=1)
        informative = amplitude > 0
        # Scale before taking norms to keep finite large and tiny vectors usable.
        np.divide(delta, amplitude[:, None], out=delta, where=informative[:, None])
        norm = np.linalg.norm(delta, axis=1)
        np.divide(delta, norm[:, None], out=delta, where=informative[:, None])
        similarity += delta @ delta.T
        support += informative[:, None] & informative[None, :]
    np.divide(similarity, support, out=similarity, where=support > 0)
    return np.clip(similarity, -1.0, 1.0), support > 0


def observe_edges(identities, base, uploads, backbone_indices, task_indices):
    backbone, backbone_observed = role_relations(base, uploads, backbone_indices)
    task, task_observed = role_relations(base, uploads, task_indices)
    return tuple(
        EdgeObservation(identities[left], identities[right], float(backbone[left, right]), float(task[left, right]))
        for left, right in combinations(range(len(identities)), 2)
        if backbone_observed[left, right] and task_observed[left, right]
    )


@dataclass(frozen=True)
class GraphAssessment:
    observation: RoundObservation
    state: str
    risks: dict[str, float]
    groups: list[dict]
    edges: list[dict]
    calibrated_fraction: float


def _bin(value, count):
    return min(count - 1, int((value + 1.0) * count / 2.0))


def _conditional_reference(history, settings):
    samples = defaultdict(list)
    by_pair = defaultdict(list)
    by_pair_bin = defaultdict(list)
    all_tasks = []
    pos_tasks = []
    for frame in history:
        for edge in frame.edges:
            band = _bin(edge.backbone, settings.backbone_bins)
            samples[band].append((frame.round_number, edge))
            all_tasks.append(edge.task)
            if edge.task >= 0.0:
                pos_tasks.append(edge.task)
            by_pair[edge.pair].append((frame.round_number, edge.score))
            by_pair_bin[edge.pair, band].append((edge.task, edge.score))

    global_center = float(np.median(all_tasks)) if all_tasks else 0.0
    pos_pool = np.asarray(
        pos_tasks if len(pos_tasks) >= settings.min_reference_pairs else all_tasks)
    pos_center = float(np.median(pos_pool)) if len(pos_pool) else 0.0
    pos_scale = (
        max(settings.residual_scale_floor,
            1.4826 * float(np.median(np.abs(pos_pool - pos_center))))
        if len(pos_pool) else settings.residual_scale_floor
    )
    total_pos = len(pos_pool)

    reference = {}
    for band, entries in samples.items():
        rounds = len({number for number, _ in entries})
        pairs = len({edge.pair for _, edge in entries})
        if rounds < settings.min_reference_rounds or pairs < settings.min_reference_pairs:
            continue
        values = np.asarray([edge.task for _, edge in entries])
        pos_values = values[values >= 0.0]
        bin_center = float(np.median(values))
        bin_scale = max(settings.residual_scale_floor,
                        1.4826 * float(np.median(np.abs(values - bin_center))))
        if bin_center < global_center and global_center > 0.0:
            # Negative conflict edges must not drag a bin's reference below the
            # global task level; keep the bin calibrated against the positive pool.
            center = max(bin_center, global_center)
            scale = pos_scale
        elif len(pos_values) >= settings.min_reference_pairs and len(values) > len(pos_values):
            # Mixed positive/negative bin: calibrate the positive population only.
            center = float(np.median(pos_values))
            scale = max(settings.residual_scale_floor,
                        1.4826 * float(np.median(np.abs(pos_values - center))))
        else:
            center = bin_center
            scale = bin_scale

        pos_share = len(pos_values) / max(1, total_pos)
        if pos_share < 0.20 and center > pos_center and pos_center > 0.0:
            # A sparse high-backbone bin dominated by a small clique must not
            # calibrate its own reference center; blend toward the global
            # positive baseline in proportion to its positive mass.
            w = pos_share / 0.20
            center = w * center + (1.0 - w) * pos_center
            scale = w * scale + (1.0 - w) * pos_scale

        reference[band] = (center, scale, rounds, pairs)
    return reference, by_pair, by_pair_bin


def _edge_strength(score, previous, round_number, settings):
    # The current calibrated observation has unit mass. Missing edges have no mass.
    observed_mass = 1.0
    exceeds = score > settings.edge_z_threshold
    excess_mass = float(exceeds)
    confirmations = int(exceeds)
    for previous_round, previous_score in previous:
        if previous_score is None:
            continue
        weight = 2.0 ** (-(round_number - previous_round) / settings.edge_half_life)
        observed_mass += weight
        if previous_score > settings.edge_z_threshold:
            excess_mass += weight
            confirmations += 1
    fraction = excess_mass / observed_mass
    strength = 0.0
    if exceeds and confirmations >= settings.min_edge_confirmations and fraction >= settings.min_exceedance_fraction:
        confidence = min(1.0, excess_mass / settings.min_edge_confirmations)
        strength = confidence * fraction * (1.0 - settings.edge_z_threshold / score)
    return strength, confirmations, observed_mass, fraction


def _prune_to_dense_core(component, neighbors, min_group_density):
    """Peel low-degree bridge nodes until the remaining component is dense."""
    comp = set(component)
    while len(comp) > 3:
        links = sum(len(neighbors[m] & comp) for m in comp) // 2
        density = 2 * links / (len(comp) * (len(comp) - 1))
        if density >= min_group_density:
            break
        min_deg = min(len(neighbors[m] & comp) for m in comp)
        victim = sorted(m for m in comp if len(neighbors[m] & comp) == min_deg)[-1]
        comp.remove(victim)
    return comp


def _structure_risks(identities, strengths, scored, settings):
    neighbors = {identity: set() for identity in identities}
    ordinary_edges = {edge.pair for edge in scored
                      if edge.score is not None and edge.score <= settings.edge_z_threshold}
    task_by_pair = {edge.pair: edge.task for edge in scored}
    for left, right in strengths:
        neighbors[left].add(right)
        neighbors[right].add(left)
    risks = dict.fromkeys(identities, 0.0)
    groups = []
    visited = set()
    for identity in identities:
        if identity in visited or not neighbors[identity]:
            continue
        component = {identity}
        pending = [identity]
        while pending:
            new = neighbors[pending.pop()] - component
            component.update(new)
            pending.extend(sorted(new))
        visited.update(component)
        component = _prune_to_dense_core(
            component, neighbors, settings.min_group_density)
        if len(component) < 3:
            # A single anomalous edge is an edge-level event, not subgraph collusion.
            continue
        members = sorted(component)
        links = sum(len(neighbors[member] & component) for member in members) // 2
        density = 2 * links / (len(members) * (len(members) - 1))
        outside = set(identities) - component
        if density < settings.min_group_density or not outside:
            continue
        # A missing boundary cannot distinguish a local group from round-wide drift.
        ordinary_boundary = sum(tuple(sorted((member, other))) in ordinary_edges
                                for member in members for other in outside)
        boundary_fraction = ordinary_boundary / (len(members) * len(outside))
        if not ordinary_boundary or boundary_fraction < settings.min_boundary_fraction:
            continue
        internal_tasks = [task_by_pair[tuple(sorted((u, v)))]
                          for u, v in combinations(members, 2)
                          if tuple(sorted((u, v))) in task_by_pair]
        boundary_tasks = [task_by_pair[tuple(sorted((u, w)))]
                          for u in members for w in outside
                          if tuple(sorted((u, w))) in task_by_pair]
        if internal_tasks and boundary_tasks:
            contrast = float(np.median(internal_tasks) - np.median(boundary_tasks))
            if contrast < 0.25:
                continue
        groups.append(dict(principals=members, edges=links, density=density,
                           boundary_fraction=boundary_fraction))
        for member in members:
            incident = [strengths[tuple(sorted((member, other)))]
                        for other in sorted(neighbors[member] & component)]
            risks[member] = math.fsum(incident) / (len(members) - 1)
    return risks, groups


def assess(round_number, identities, current_edges, history, settings):
    """Score the current graph against past observations before admitting it to history."""
    reference, by_pair, by_pair_bin = _conditional_reference(history, settings)
    scored = []
    evidence = []
    strengths = {}
    for edge in current_edges:
        band = _bin(edge.backbone, settings.backbone_bins)
        stats = reference.get(band)
        row = dict(left=edge.left, right=edge.right, backbone=edge.backbone, task=edge.task,
                   reference_center=None, reference_scale=None, reference_rounds=0, reference_pairs=0,
                   pair_reference_observations=0, residual=None, score=None, confirmations=0,
                   effective_observations=0.0, exceedance_fraction=None, strength=0.0)
        if stats is not None:
            center, scale, rounds, pairs = stats
            pair_entries = by_pair_bin.get((edge.pair, band), [])
            valid_tasks = [task for task, score in pair_entries
                           if score is None or score <= settings.edge_z_threshold]
            if valid_tasks:
                shrink = len(valid_tasks) / (len(valid_tasks) + settings.pair_shrinkage)
                shift = shrink * (float(np.median(valid_tasks)) - center)
                headroom = max(0.0, 1.0 - max(0.0, center) ** 2)
                effective_limit = settings.pair_shift_limit * headroom
                center += float(np.clip(shift, -effective_limit, effective_limit))
            residual = edge.task - center
            headroom_factor = max(0.15, 1.0 - max(0.0, center) ** 2)
            effective_scale = max(settings.residual_scale_floor * headroom_factor,
                                  scale * headroom_factor)
            score = residual / effective_scale
            strength, count, mass, fraction = _edge_strength(score, by_pair[edge.pair], round_number, settings)
            row.update(reference_center=center, reference_scale=scale, reference_rounds=rounds,
                       reference_pairs=pairs, pair_reference_observations=len(pair_entries),
                       residual=residual, score=score, confirmations=count,
                       effective_observations=mass, exceedance_fraction=fraction,
                       strength=strength)
            if strength > 0:
                strengths[edge.pair] = strength
        scored.append(EdgeObservation(edge.left, edge.right, edge.backbone, edge.task, row["score"]))
        evidence.append(row)

    risks, groups = _structure_risks(identities, strengths, scored, settings)
    possible = len(identities) * (len(identities) - 1) // 2
    calibrated = sum(edge.score is not None for edge in scored)
    coverage = calibrated / possible if possible else 0.0
    unmatched_excess = any(edge.score is not None and edge.score > settings.edge_z_threshold for edge in scored)
    if groups:
        state = "anomalous_structure"
    elif not calibrated or coverage < settings.min_calibrated_fraction or unmatched_excess:
        state = "insufficient_evidence"
    else:
        state = "no_anomaly_evidence"
    return GraphAssessment(RoundObservation(round_number, tuple(scored)), state, risks, groups, evidence, coverage)
