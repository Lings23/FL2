"""R0 read-only, trainable-only geometry. No labels, RNG, caps or state updates."""
from __future__ import annotations

import json
from typing import Mapping, Sequence
import numpy as np

from defenses.rtc.weighted_stats import weighted_coordinate_median

VERSION = 'rtc_r0_trainable_geometry.v1'


def observe_direction(*, global_params, updates, aggregated, client_ids: Sequence[str],
                      trainable_indices, parameter_names: Mapping[str, str],
                      weights, clip_factors, anchor_mass: float, principal_ids=None, f: int = 3, m: int = 5,
                      reference_mode: str = 'multi_krum'):
    """Reference: deterministic Multi-Krum on clipped trainable deltas, top m mean.

    Norms/projections use trainable tensors only. Layer diagnostics use the same
    selected set, not independent layer-wise client selection. Gram matrices keep
    peak temporary storage to one tensor. Actual output includes dtype rounding;
    reconstructed client + accepted-anchor terms are checked separately.
    """
    n = len(updates)
    if reference_mode not in ('multi_krum', 'geometric_median'):
        raise ValueError('invalid observation reference mode')
    indices = sorted(trainable_indices) if trainable_indices is not None else []
    ids = [str(cid) for cid in client_ids]
    principals = ids if principal_ids is None else [str(p) for p in principal_ids]
    w, factors = np.asarray(weights, dtype=float), np.asarray(clip_factors, dtype=float)
    if (not indices or len(set(ids)) != n or len(ids) != n or len(principals) != n or w.shape != (n,) or
            factors.shape != (n,) or not np.isfinite(w).all() or not np.isfinite(factors).all() or
            (w < 0).any() or (factors < 0).any() or (factors > 1).any() or
            not np.isfinite(anchor_mass) or not 0 <= anchor_mass <= 1):
        raise ValueError('invalid R0 observation layout, identities or mass')
    if f < 0 or m < 1:
        raise ValueError('invalid reference parameters')
    grams, names, raw_sq = [], [], np.zeros(n)
    anchor_sq = aggregate_sq = 0.0
    anchor_dots, aggregate_dots = np.zeros(n), np.zeros(n)
    reconstruction_sq = 0.0
    for index in indices:
        if index < 0 or index >= len(global_params) or str(index) not in parameter_names:
            raise ValueError('R0 requires explicit trainable index/name metadata')
        base = np.asarray(global_params[index], dtype=np.float64)
        if not np.issubdtype(np.asarray(global_params[index]).dtype, np.floating):
            raise ValueError('trainable tensor must be floating')
        raw = np.stack([np.asarray(params[index], dtype=np.float64) - base for params, _ in updates])
        if raw.shape[1:] != base.shape or not np.isfinite(raw).all():
            raise ValueError('invalid trainable delta')
        raw = raw.reshape(n, -1)
        raw_sq += np.einsum('ij,ij->i', raw, raw)
        clipped = raw * factors[:, None]
        gram = clipped @ clipped.T
        grams.append(gram)
        names.append(parameter_names[str(index)])
        anchor = (weighted_coordinate_median([row for row in clipped], w,
                    principal_ids=principals, client_ids=ids) if w.sum() > 0 and anchor_mass > 0
                  else np.zeros(clipped.shape[1]))
        actual = (np.asarray(aggregated[index], dtype=np.float64) - base).reshape(-1)
        if not np.isfinite(actual).all():
            raise ValueError('invalid aggregate')
        anchor_dots += clipped @ anchor
        aggregate_dots += clipped @ actual
        anchor_sq += float(anchor @ anchor)
        aggregate_sq += float(actual @ actual)
        error = actual - (w @ clipped + anchor_mass * anchor)
        reconstruction_sq += float(error @ error)
    gram = sum(grams)
    diag = np.maximum(np.diag(gram), 0)
    distances = np.maximum(diag[:, None] + diag[None, :] - 2 * gram, 0)
    scores = np.full(n, np.nan)
    selected = []
    if n >= 2 * f + 3 and m <= n - f - 2:
        for i in range(n):
            scores[i] = np.sort(np.delete(distances[i], i))[:n-f-2].sum()
        selected = sorted(range(n), key=lambda i: (scores[i], ids[i]))[:m]
    alpha = np.zeros(n)
    if selected:
        alpha[selected] = 1 / len(selected)
    extra_summary, loo_cosines = {}, [None] * n
    version = VERSION
    reference_converged = True
    minimum_reference_norm = 1e-12
    if reference_mode == 'geometric_median':
        from defenses.rtc.geometric_reference import geometric_reference, RELATIVE_MIN_NORM
        alpha, convergence = geometric_reference(gram)
        reference_converged = convergence['converged']
        minimum_reference_norm = max(1e-12, RELATIVE_MIN_NORM * convergence['scale'])
        selected = list(range(n))
        version = 'rtc_r0b_geometric_geometry.v1'
        loo_valid = 0
        for i in range(n):
            loo, info = geometric_reference(gram, omit=i)
            rn = float(np.sqrt(max(0, loo @ gram @ loo)))
            if info['converged'] and rn > max(1e-12, RELATIVE_MIN_NORM * info['scale']) and diag[i] > 1e-24:
                loo_cosines[i] = float(np.clip((gram @ loo)[i] / (np.sqrt(diag[i]) * rn), -1, 1))
                loo_valid += 1
        extra_summary = {
            'rtc_r0_reference_mode': reference_mode,
            'rtc_r0_reference_converged': convergence['converged'],
            'rtc_r0_reference_iterations': convergence['iterations'],
            'rtc_r0_reference_relative_step': convergence['relative_step'],
            'rtc_r0_reference_weights_json': json.dumps(alpha.tolist()),
            'rtc_r0_geometry_client_ids_json': json.dumps(ids),
            'rtc_r0_gram_json': json.dumps(gram.tolist(), allow_nan=False),
            'rtc_r0_loo_valid_count': loo_valid,
        }
    reference_norm = float(np.sqrt(max(0, alpha @ gram @ alpha)))
    valid = bool(selected and reference_converged and reference_norm > minimum_reference_norm)
    norms = np.sqrt(diag)
    raw_norms = np.sqrt(raw_sq)
    projections = gram @ alpha / reference_norm if valid else np.zeros(n)
    cosines = np.divide(projections, norms, out=np.zeros(n), where=norms > 1e-12)
    layer_rows = [{} for _ in ids]
    for name, layer in zip(names, grams):
        ln = np.sqrt(np.maximum(np.diag(layer), 0))
        rn = float(np.sqrt(max(0, alpha @ layer @ alpha)))
        for i in range(n):
            layer_rows[i][name] = {'norm': float(ln[i]), 'reference_norm': rn,
                'cosine': float(np.clip((layer @ alpha)[i] / (ln[i] * rn), -1, 1))
                          if valid and ln[i] > 1e-12 and rn > 1e-12 else None}
    records = []
    for i, cid in enumerate(ids):
        records.append({'rtc_r0_version': version, 'rtc_r0_reference_valid': valid,
            'rtc_r0_reference_selected': i in selected,
            'rtc_r0_trainable_raw_norm': float(raw_norms[i]),
            'rtc_r0_trainable_clipped_norm': float(norms[i]),
            'rtc_r0_clip_factor': float(factors[i]),
            'rtc_r0_krum_score': float(scores[i]) if selected and reference_mode == 'multi_krum' else None,
            'rtc_r0_cosine': float(np.clip(cosines[i], -1, 1)) if valid and norms[i] > 1e-12 else None,
            'rtc_r0_projection': float(projections[i]) if valid else None,
            'rtc_r0_weighted_projection': float(w[i] * projections[i]) if valid else None,
            'rtc_r0_negative_contribution': float(w[i] * max(0, -projections[i])) if valid else None,
            'rtc_r0_layers_json': json.dumps(layer_rows[i], sort_keys=True, allow_nan=False)})
        if reference_mode == 'geometric_median':
            records[-1].update(rtc_r0_loo_cosine=loo_cosines[i], rtc_r0_reference_weight=float(alpha[i]))
    summary = {'rtc_r0_version': version, 'rtc_r0_observe_only': True,
        'rtc_r0_reference_valid': valid, 'rtc_r0_reference_norm': reference_norm,
        'rtc_r0_reference_f': f, 'rtc_r0_reference_m': m,
        'rtc_r0_reference_ids_json': json.dumps([ids[i] for i in selected]),
        'rtc_r0_trainable_tensor_count': len(indices),
        'rtc_r0_client_sum_norm': float(np.sqrt(max(0, w @ gram @ w))),
        'rtc_r0_anchor_contribution_norm': float(anchor_mass * np.sqrt(anchor_sq)),
        'rtc_r0_actual_aggregate_norm': float(np.sqrt(aggregate_sq)),
        'rtc_r0_reconstruction_error': float(np.sqrt(reconstruction_sq))}
    summary.update(extra_summary)
    if valid:
        summary.update(rtc_r0_client_projection=float(w @ projections),
                       rtc_r0_anchor_projection=float(anchor_mass * (alpha @ anchor_dots) / reference_norm),
                       rtc_r0_actual_projection=float((alpha @ aggregate_dots) / reference_norm))
    return records, summary
