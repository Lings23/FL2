"""R1 leave-one-out spectral reference and immediate cap. No identity labels.

References average unit updates; aggregation still uses the original clipped
updates. Insufficient support/confidence abstains. This is an empirical filter,
not a Byzantine safety guarantee, especially under non-IID sampling.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 'rtc_r1_spectral_loo.v1'


def load_calibration(source):
    payload = json.loads(Path(source).read_text(encoding='utf-8-sig'))
    if set(payload).difference({'corroboration'}) != {'version', 'f', 'minimum_gap', 'cosine_threshold', 'cap_multiplier', 'provenance'}:
        raise ValueError('invalid spectral calibration keys')
    if payload['version'] != VERSION or type(payload['f']) is not int or payload['f'] != 3:
        raise ValueError('invalid spectral calibration version/f')
    for key, lo, hi in [('minimum_gap', 0, 2), ('cosine_threshold', -1, 0), ('cap_multiplier', 0, 1)]:
        value = payload[key]
        if type(value) not in (float, int) or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'invalid spectral calibration {key}')
    if payload['cap_multiplier'] >= 1 or payload['minimum_gap'] <= 0:
        raise ValueError('ineffective spectral calibration')
    if 'corroboration' in payload:
        from defenses.rtc.corroboration import validate
        validate(payload['corroboration'])
    return payload


def calibration_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def score_gram(gram, f=3):
    g = np.asarray(gram, dtype=float)
    if type(f) is not int or f < 0:
        raise ValueError('invalid f')
    if g.ndim != 2 or g.shape[0] != g.shape[1] or not np.isfinite(g).all():
        raise ValueError('invalid Gram matrix')
    n = len(g)
    scale = max(1., float(np.max(np.abs(g), initial=0)))
    if not np.allclose(g, g.T, atol=1e-10 * scale, rtol=0) or (n and np.linalg.eigvalsh(g).min() < -1e-8 * scale):
        raise ValueError('Gram must be symmetric positive semidefinite')
    norm = np.sqrt(np.maximum(np.diag(g), 0))
    cosine = np.divide(g, np.outer(norm, norm), out=np.zeros_like(g), where=np.outer(norm, norm) > 1e-24)
    results = []
    for i in range(n):
        keep = np.array([j for j in range(n) if j != i and norm[j] > 1e-12], dtype=int)
        row = {'valid': False, 'cosine': None, 'gap': None, 'reference_indices': [], 'reference_weights': []}
        results.append(row)
        if len(keep) < 2 * f + 3 or norm[i] <= 1e-12:
            row['reason'] = 'insufficient_nonzero_clients'; continue
        sub = cosine[np.ix_(keep, keep)].copy()
        np.fill_diagonal(sub, 0)
        eigenvalues, eigenvectors = np.linalg.eigh(sub)
        v = eigenvectors[:, -1]
        positive, negative = int((v > 1e-8).sum()), int((v < -1e-8).sum())
        gap = float((eigenvalues[-1] - eigenvalues[-2]) / max(abs(eigenvalues[-1]), 1e-12))
        row['gap'] = gap
        if eigenvalues[-1] <= 1e-12 or gap <= 1e-8 or positive == negative:
            row['reason'] = 'ambiguous_spectrum_or_orientation'; continue
        if positive < negative:
            v = -v
        chosen = v > 1e-8
        if int(chosen.sum()) < len(keep) - f:
            row['reason'] = 'insufficient_majority_support'; continue
        ref_ids = keep[chosen]
        weights = v[chosen] / v[chosen].sum()
        ref_norm = float(np.sqrt(max(0, weights @ cosine[np.ix_(ref_ids, ref_ids)] @ weights)))
        if ref_norm <= 1e-6:
            row['reason'] = 'small_reference'; continue
        row.update(valid=True, cosine=float(np.clip(cosine[i, ref_ids] @ weights / ref_norm, -1, 1)),
                   reference_indices=ref_ids.tolist(), reference_weights=weights.tolist(), reason='available')
    return results


def measure(clipped, positions, calibration, mode):
    if mode not in ('observe', 'cap') or not positions or len(set(positions)) != len(positions):
        raise ValueError('invalid spectral mode/trainable layout')
    grams = []
    for pos in positions:
        matrix = np.stack([delta[pos].reshape(-1) for delta in clipped])
        grams.append(matrix @ matrix.T)
    gram = sum(grams)
    rows = score_gram(gram, calibration['f'])
    for i, row in enumerate(rows):
        row['usable'] = bool(row['valid'] and row['gap'] >= calibration['minimum_gap'])
        row['flagged'] = bool(row['usable'] and row['cosine'] < calibration['cosine_threshold'])
        if 'corroboration' in calibration:
            from defenses.rtc.corroboration import corroboration
            row['base_flagged'] = row['flagged']
            row['corroboration'] = corroboration(gram, i, row['reference_indices'], calibration['corroboration']['pairwise_threshold'])
            row['flagged'] = row['base_flagged'] and row['corroboration']['passes']
        row['q'] = calibration['cap_multiplier'] if mode == 'cap' and row['flagged'] else 1.
    return {'gram': gram, 'grams': grams, 'rows': rows, 'positions': positions,
            'mode': mode, 'calibration': calibration}


def diagnostics(measured, clipped, anchors, actual_deltas, weights, anchor_mass, client_ids, layer_names):
    """Signed projections use a DIFFERENT leave-one-out axis per client.

    Keep missing projections missing. Their sum is only a diagnostic proxy;
    it is not a projection of the total vector on one common direction.
    """
    gram, rows = measured['gram'], measured['rows']
    n = len(rows)
    norm = np.sqrt(np.maximum(np.diag(gram), 0))
    w = np.asarray(weights, dtype=float)
    anchor_dots, actual_dots = np.zeros(n), np.zeros(n)
    anchor_sq = actual_sq = error_sq = 0.
    for pos in measured['positions']:
        matrix = np.stack([delta[pos].reshape(-1) for delta in clipped])
        a, actual = anchors[pos].reshape(-1), actual_deltas[pos].reshape(-1)
        anchor_dots += matrix @ a
        actual_dots += matrix @ actual
        anchor_sq += float(a @ a)
        actual_sq += float(actual @ actual)
        error = actual - (w @ matrix + anchor_mass * a)
        error_sq += float(error @ error)
    records = []
    for i, row in enumerate(rows):
        beta = np.zeros(n)
        if row['valid']:
            ref = row['reference_indices']
            beta[ref] = np.asarray(row['reference_weights']) / norm[ref]
        rn = float(np.sqrt(max(0, beta @ gram @ beta)))
        projection = float(norm[i] * row['cosine']) if row['usable'] else None
        layers = {}
        for name, lg in zip(layer_names, measured['grams']):
            ln, lr = np.sqrt(max(0, lg[i, i])), np.sqrt(max(0, beta @ lg @ beta))
            layers[name] = {'norm': float(ln), 'cosine': float(np.clip((lg @ beta)[i] / (ln * lr), -1, 1))
                            if row['usable'] and ln > 1e-12 and lr > 1e-12 else None}
        records.append({'rtc_r1_valid': row['valid'], 'rtc_r1_usable': row['usable'],
            'rtc_r1_reason': 'low_gap' if row['valid'] and not row['usable'] else row['reason'],
            'rtc_r1_cosine': row['cosine'], 'rtc_r1_gap': row['gap'],
            'rtc_r1_flagged': row['flagged'], 'rtc_r1_q': row['q'],
            'rtc_r1_reference_ids_json': json.dumps([str(client_ids[j]) for j in row['reference_indices']]),
            'rtc_r1_reference_weights_json': json.dumps(row['reference_weights']),
            'rtc_r1_trainable_clipped_norm': float(norm[i]), 'rtc_r1_projection': projection,
            'rtc_r1_weighted_projection': float(w[i] * projection) if projection is not None else None,
            'rtc_r1_negative_contribution': float(w[i] * max(0, -projection)) if projection is not None else None,
            'rtc_r1_anchor_projection': float(anchor_mass * beta @ anchor_dots / rn) if row['usable'] else None,
            'rtc_r1_actual_projection': float(beta @ actual_dots / rn) if row['usable'] else None,
            'rtc_r1_layers_json': json.dumps(layers, allow_nan=False)})
        if 'corroboration' in row:
            records[-1].update(rtc_r1_base_flagged=row['base_flagged'],
                rtc_r1_corroboration_votes=row['corroboration']['votes'],
                rtc_r1_corroboration_required=row['corroboration']['required'],
                rtc_r1_corroboration_passes=row['corroboration']['passes'])
    return records, {'rtc_r1_version': VERSION, 'rtc_r1_mode': measured['mode'],
        'rtc_r1_calibration_hash': calibration_hash(measured['calibration']),
        'rtc_r1_client_ids_json': json.dumps(list(client_ids)), 'rtc_r1_gram_json': json.dumps(gram.tolist(), allow_nan=False),
        'rtc_r1_client_sum_norm': float(np.sqrt(max(0, w @ gram @ w))),
        'rtc_r1_anchor_contribution_norm': float(anchor_mass * np.sqrt(anchor_sq)),
        'rtc_r1_actual_aggregate_norm': float(np.sqrt(actual_sq)), 'rtc_r1_reconstruction_error': float(np.sqrt(error_sq))}
